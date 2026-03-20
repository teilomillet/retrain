"""Adam invariance test — gradient-level measurement.

Instead of training to convergence and comparing final accuracy (too many
confounds: saturation, LoRA, task ceiling), this measures the MECHANISM
directly:

  1. Same batch of data, same model state
  2. Compute gradients under different advantage modes
  3. Compare: raw gradients g, Adam-preconditioned updates m/√v, cosine sim

If Adam absorbs the difference:
  → raw gradients differ (cosine < 1.0, magnitude ratio ≠ 1.0)
  → preconditioned updates converge (cosine ≈ 1.0, magnitude ratio ≈ 1.0)

If the interventions are genuinely different:
  → both raw and preconditioned differ

No training needed — just forward-backward passes. Runs in ~30 seconds on CPU.

    uv run python experiments/adam_invariance_test.py
"""

import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tiny model (50K params, CPU) ────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=64, dim=64, heads=4, layers=2, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(dim, vocab)
        self.vocab = vocab

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        h = self.embed(x) + self.pos(pos)
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)


# ── Synthetic batch ─────────────────────────────────────────────────────

START, SEP, END = 0, 1, 12
DIGITS = list(range(2, 12))


def make_batch(model, group_size=16):
    """Generate one prompt + group_size completions with synthetic binary rewards.

    We assign reward=1.0 to the best half (by likelihood) and 0.0 to the rest.
    This guarantees mixed rewards regardless of task difficulty.
    """
    n = random.randint(2, 4)
    digits = [random.choice(DIGITS) for _ in range(n)]
    prompt = [START] + digits + [SEP]

    completions = []
    with torch.no_grad():
        for _ in range(group_size):
            ids = list(prompt)
            logprobs = []
            for _ in range(8):
                x = torch.tensor([ids])
                logits = model(x)[0, -1]
                probs = F.softmax(logits / 0.8, dim=-1)
                tok = torch.multinomial(probs, 1).item()
                lp = F.log_softmax(logits, dim=-1)[tok].item()
                logprobs.append(lp)
                ids.append(tok)
                if tok == END:
                    break
            gen = ids[len(prompt):]
            # Reward assigned below
            completions.append((gen, logprobs, 0.0))

    # Assign binary rewards: top half by total logprob = correct
    total_lps = [sum(c[1]) for c in completions]
    median_lp = sorted(total_lps)[group_size // 2]
    completions = [
        (gen, lps, 1.0 if sum(lps) >= median_lp else 0.0)
        for gen, lps, _ in completions
    ]

    return prompt, completions


# ── Compute gradients for a given advantage vector ──────────────────────

def compute_grad(model, prompt, completions, token_advs_list):
    """Forward-backward, return flat gradient vector."""
    model.zero_grad()
    total_loss = 0.0
    n_tok = 0

    for i, (gen_ids, _, _) in enumerate(completions):
        if not gen_ids:
            continue
        tok_advs = token_advs_list[i]
        full = prompt + gen_ids
        x = torch.tensor([full])
        logits = model(x)[0, len(prompt)-1:-1]
        gen_t = torch.tensor(gen_ids)
        n = min(len(gen_ids), logits.shape[0])
        lps = F.log_softmax(logits[:n], dim=-1)
        current_lps = lps[torch.arange(n), gen_t[:n]]
        adv_t = torch.tensor(tok_advs[:n], dtype=torch.float32)
        loss = -(adv_t * current_lps).sum() / max(n, 1)
        loss.backward()
        total_loss += loss.item()
        n_tok += n

    # Collect flat gradient
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel()))
    return torch.cat(grads)


# ── Adam preconditioner ─────────────────────────────────────────────────

class AdamState:
    """Mimics Adam's preconditioning without actually updating params."""
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = torch.zeros(sum(p.numel() for p in model.parameters()))
        self.v = torch.zeros_like(self.m)
        self.t = 0

    def precondition(self, g):
        """Given raw gradient g, return what Adam's update would be."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update = self.lr * m_hat / (v_hat.sqrt() + self.eps)
        return update


# ── Advantage modes ─────────────────────────────────────────────────────

def grpo_advantages(rewards):
    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


def maxrl_advantages(rewards):
    mean_r = sum(rewards) / len(rewards)
    return [(r - mean_r) / (mean_r + 0.01) for r in rewards]


def dg_token_advs(advantage, logprobs):
    """DG with scale normalization, eta=0.5, lambda=0.5."""
    n = len(logprobs)
    if n < 2 or advantage == 0.0:
        return [advantage] * n
    surps = [-lp for lp in logprobs]
    mean_s = sum(surps) / n
    var_s = sum((s - mean_s) ** 2 for s in surps) / n
    std_s = max(var_s ** 0.5, 1e-8)
    result = []
    for s in surps:
        z = s / std_s
        x = advantage * z / 0.5
        gate = 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))
        weight = 0.5 + 0.5 * gate
        result.append(advantage * weight)
    return result


# ── Metrics ─────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    dot = (a * b).sum()
    na = a.norm()
    nb = b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return (dot / (na * nb)).item()


def magnitude_ratio(a, b):
    na = a.norm().item()
    nb = b.norm().item()
    if nb < 1e-12:
        return float('inf')
    return na / nb


# ── Main experiment ─────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    random.seed(42)

    model = TinyLM()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Warm up Adam state with a few batches so m and v are non-zero
    adam_grpo = AdamState(model)
    adam_maxrl = AdamState(model)
    adam_pg = AdamState(model)
    adam_dg = AdamState(model)

    N_BATCHES = 20  # measure over 20 batches, report statistics

    metrics = {
        "grpo_vs_maxrl_raw_cosine": [],
        "grpo_vs_maxrl_raw_mag_ratio": [],
        "grpo_vs_maxrl_adam_cosine": [],
        "grpo_vs_maxrl_adam_mag_ratio": [],
        "pg_vs_dg_raw_cosine": [],
        "pg_vs_dg_raw_mag_ratio": [],
        "pg_vs_dg_adam_cosine": [],
        "pg_vs_dg_adam_mag_ratio": [],
    }

    for batch_i in range(N_BATCHES):
        prompt, completions = make_batch(model)
        rewards = [r for _, _, r in completions]

        # Skip uniform reward batches
        if all(r == rewards[0] for r in rewards):
            continue

        # ── Test 1: GRPO vs MaxRL (same tokens, different advantage scale) ──
        grpo_advs = grpo_advantages(rewards)
        maxrl_advs = maxrl_advantages(rewards)

        # Uniform token weighting for both
        grpo_tok = [[a] * len(c[0]) for a, c in zip(grpo_advs, completions)]
        maxrl_tok = [[a] * len(c[0]) for a, c in zip(maxrl_advs, completions)]

        g_grpo = compute_grad(model, prompt, completions, grpo_tok)
        g_maxrl = compute_grad(model, prompt, completions, maxrl_tok)

        u_grpo = adam_grpo.precondition(g_grpo)
        u_maxrl = adam_maxrl.precondition(g_maxrl)

        metrics["grpo_vs_maxrl_raw_cosine"].append(cosine_sim(g_grpo, g_maxrl))
        metrics["grpo_vs_maxrl_raw_mag_ratio"].append(magnitude_ratio(g_maxrl, g_grpo))
        metrics["grpo_vs_maxrl_adam_cosine"].append(cosine_sim(u_grpo, u_maxrl))
        metrics["grpo_vs_maxrl_adam_mag_ratio"].append(magnitude_ratio(u_maxrl, u_grpo))

        # ── Test 2: PG vs DG (same advantages, different token weighting) ──
        pg_tok = [[a] * len(c[0]) for a, c in zip(grpo_advs, completions)]
        dg_tok = [dg_token_advs(a, c[1]) for a, c in zip(grpo_advs, completions)]

        g_pg = compute_grad(model, prompt, completions, pg_tok)
        g_dg = compute_grad(model, prompt, completions, dg_tok)

        u_pg = adam_pg.precondition(g_pg)
        u_dg = adam_dg.precondition(g_dg)

        metrics["pg_vs_dg_raw_cosine"].append(cosine_sim(g_pg, g_dg))
        metrics["pg_vs_dg_raw_mag_ratio"].append(magnitude_ratio(g_dg, g_pg))
        metrics["pg_vs_dg_adam_cosine"].append(cosine_sim(u_pg, u_dg))
        metrics["pg_vs_dg_adam_mag_ratio"].append(magnitude_ratio(u_dg, u_pg))

        print(f"  batch {batch_i:2d} | "
              f"GRPO/MaxRL raw_cos={metrics['grpo_vs_maxrl_raw_cosine'][-1]:.4f} "
              f"adam_cos={metrics['grpo_vs_maxrl_adam_cosine'][-1]:.4f} | "
              f"PG/DG raw_cos={metrics['pg_vs_dg_raw_cosine'][-1]:.4f} "
              f"adam_cos={metrics['pg_vs_dg_adam_cosine'][-1]:.4f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GRADIENT-LEVEL MEASUREMENTS")
    print("=" * 70)

    def summarize(values, label):
        if not values:
            print(f"  {label}: no data")
            return
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        print(f"  {label}: {mean:.4f} ± {std:.4f}")

    print("\n1. GRPO vs MaxRL (pure advantage scaling)")
    print("-" * 55)
    summarize(metrics["grpo_vs_maxrl_raw_cosine"], "Raw gradient cosine    ")
    summarize(metrics["grpo_vs_maxrl_raw_mag_ratio"], "Raw gradient mag ratio ")
    summarize(metrics["grpo_vs_maxrl_adam_cosine"], "Adam update cosine     ")
    summarize(metrics["grpo_vs_maxrl_adam_mag_ratio"], "Adam update mag ratio  ")

    print("\n2. PG vs DG (token-level reweighting)")
    print("-" * 55)
    summarize(metrics["pg_vs_dg_raw_cosine"], "Raw gradient cosine    ")
    summarize(metrics["pg_vs_dg_raw_mag_ratio"], "Raw gradient mag ratio ")
    summarize(metrics["pg_vs_dg_adam_cosine"], "Adam update cosine     ")
    summarize(metrics["pg_vs_dg_adam_mag_ratio"], "Adam update mag ratio  ")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  GRPO vs MaxRL: raw cosine should be ~1.0 (same direction,")
    print("    different scale). Adam update cosine also ~1.0 and mag ratio")
    print("    closer to 1.0 = Adam absorbs the scale difference.")
    print()
    print("  PG vs DG: raw cosine < 1.0 = DG changes gradient direction.")
    print("    If Adam cosine ≈ raw cosine: Adam does NOT absorb the")
    print("    directional change (Adam is not the explanation).")
    print("    If Adam cosine > raw cosine: Adam partially absorbs it.")
    print("=" * 70)


if __name__ == "__main__":
    main()
