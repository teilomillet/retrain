"""Integration test: TL-GRPO + EMA baseline against the real Soma kernel.

Verifies that:
1. Branching produces diverse reward_deltas for accept_customer turns
2. set_price turns get zero local advantage (confirmed kernel behavior)
3. EMA baseline restores nonzero outcome signal to ALL turns
4. The full pipeline (branching + advantage computation) is consistent
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

# ---------------------------------------------------------------------------
# Detect whether the Soma kernel binary is available
# ---------------------------------------------------------------------------

_SOMA_ROOT = Path(__file__).resolve().parent.parent.parent
_CARGO_TOML = _SOMA_ROOT / "Cargo.toml"
_HAS_KERNEL = _CARGO_TOML.is_file()
_KERNEL_CMD: list[str] | None = None
if _HAS_KERNEL:
    _KERNEL_CMD = ["cargo", "run", "--quiet", "--bin", "soma_json"]

SEED = 7
CONFIG = {
    "machine_id": "test-tl",
    "sku": "cola",
    "starting_inventory": 5,
    "starting_price_cents": 175,
    "restock_quantity": 5,
    "restock_delay_ticks": 2,
    "low_inventory_threshold": 2,
    "base_demand_cents": 200,
    "demand_spread_cents": 80,
}
EVALUATOR = {
    "horizon_ticks": 12,
    "min_revenue_cents": 300,
    "min_service_level": 0.55,
    "revenue_weight": 1.0,
    "service_level_weight": 5.0,
    "lost_sale_penalty": 1.5,
    "restock_in_flight_penalty": 0.5,
}


def _execute(operations: list[dict], seed: int = SEED) -> dict:
    """Run operations through the Soma kernel."""
    if _KERNEL_CMD is None:
        pytest.skip("Soma kernel binary not available")
    payload = {
        "seed": seed,
        "operations": operations,
        "config": CONFIG,
        "evaluator": EVALUATOR,
    }
    result = subprocess.run(
        _KERNEL_CMD, cwd=str(_SOMA_ROOT),
        input=json.dumps(payload), text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"Kernel failed: {result.stderr}")
    return json.loads(result.stdout)


def _cumulative(resp: dict) -> float:
    return float(resp.get("run", {}).get("cumulative_reward", 0.0))


def _customer_waiting(resp: dict) -> bool:
    mv = resp.get("model_view", {})
    run = mv.get("run", {})
    obs = run.get("observation", {})
    return bool(obs.get("customer_waiting", False))


def _normalize_reward(resp: dict) -> float:
    """Replicate verifiers_env._normalize_reward for consistency."""
    report = resp["evaluation"]
    summary = report.get("summary", {})
    revenue = float(summary.get("revenue_cents", 0))
    service_level = float(summary.get("service_level", 0.0))
    revenue_target = float(EVALUATOR.get("min_revenue_cents", 1))
    service_target = float(EVALUATOR.get("min_service_level", 1.0))
    reward = 0.55 * min(1.0, revenue / max(1.0, revenue_target))
    reward += 0.45 * min(1.0, service_level / max(0.01, service_target))
    if bool(report.get("passed")):
        reward = 1.0
    return max(0.0, min(1.0, reward))


# ---------------------------------------------------------------------------
# Kernel reward_delta verification
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_KERNEL, reason="Soma kernel not available")
class TestKernelRewardDeltas:
    """Verify kernel reward_delta behavior that TL-GRPO depends on."""

    def test_set_price_has_zero_delta(self):
        """set_price produces zero cumulative_reward change."""
        r0 = _execute([])
        cum_before = _cumulative(r0)

        r1 = _execute([{"kind": "act", "action": {"type": "set_price", "cents": 150}}])
        cum_after = _cumulative(r1)
        assert abs(cum_after - cum_before) < 1e-10

    def test_different_prices_same_immediate_delta(self):
        """All set_price variants produce zero delta."""
        cums = []
        for price in [100, 150, 200, 250, 300]:
            r = _execute([{"kind": "act", "action": {"type": "set_price", "cents": price}}])
            cums.append(_cumulative(r))
        for c in cums:
            assert abs(c - cums[0]) < 1e-10

    def test_schedule_restock_has_zero_delta(self):
        """schedule_restock produces zero cumulative_reward change."""
        r0 = _execute([])
        r1 = _execute([{"kind": "act", "action": {"type": "schedule_restock"}}])
        assert abs(_cumulative(r1) - _cumulative(r0)) < 1e-10

    def test_accept_customer_has_nonzero_delta(self):
        """accept_customer produces nonzero reward when customer is waiting."""
        # seed=7 has a customer waiting after first wait
        r_wait = _execute([{"kind": "wait"}])
        assert _customer_waiting(r_wait), "Expected customer waiting at seed=7 tick=1"
        cum_before = _cumulative(r_wait)

        r_accept = _execute([
            {"kind": "wait"},
            {"kind": "act", "action": {"type": "accept_customer"}},
        ])
        delta = _cumulative(r_accept) - cum_before
        assert abs(delta) > 0.01, f"accept_customer delta should be nonzero, got {delta}"
        print(f"\naccept_customer delta = {delta} (price=175, revenue=1.75)")


# ---------------------------------------------------------------------------
# TL-GRPO branching against real kernel
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_KERNEL, reason="Soma kernel not available")
class TestTlGrpoBranchingIntegration:

    def test_branching_set_price_zero_variance(self):
        """Branching at set_price: all alternatives produce delta=0."""
        pre_cum = _cumulative(_execute([]))
        deltas = []
        for price in [100, 150, 200, 250]:
            post = _execute([{"kind": "act", "action": {"type": "set_price", "cents": price}}])
            deltas.append(_cumulative(post) - pre_cum)

        variance = sum((d - sum(deltas)/len(deltas))**2 for d in deltas) / len(deltas)
        print(f"\nset_price branch deltas: {deltas}, variance: {variance}")
        assert variance < 1e-10

    def test_branching_accept_reject_nonzero_variance(self):
        """Branching at customer turn: accept vs reject vs wait produce different deltas."""
        # After one wait, customer is waiting (seed=7)
        ops_before = [{"kind": "wait"}]
        r_before = _execute(ops_before)
        assert _customer_waiting(r_before)
        pre_cum = _cumulative(r_before)

        alternatives = [
            {"kind": "act", "action": {"type": "accept_customer"}},
            {"kind": "act", "action": {"type": "reject_customer"}},
            {"kind": "wait"},
        ]
        deltas = []
        for alt in alternatives:
            post = _execute(ops_before + [alt])
            deltas.append(_cumulative(post) - pre_cum)

        variance = sum((d - sum(deltas)/len(deltas))**2 for d in deltas) / len(deltas)
        print(f"\naccept/reject/wait branch deltas: {deltas}, variance: {variance}")
        assert variance > 0.01, f"Expected nonzero variance, got {variance}"

    def test_ema_baseline_restores_signal_to_delayed_reward_turns(self):
        """Full pipeline: EMA baseline gives nonzero advantages to set_price turns."""
        from retrain.verifiers_bridge import _compute_tl_grpo_advantages

        # Play a short episode: set_price → wait → accept → advance to end
        ops_episode = [
            {"kind": "act", "action": {"type": "set_price", "cents": 175}},
            {"kind": "wait"},
            {"kind": "act", "action": {"type": "accept_customer"}},
            {"kind": "advance_time", "ticks": 100},
        ]

        # Compute per-turn reward deltas
        turn_ops = ops_episode[:3]  # skip advance_time (not a "turn")
        turn_log = []
        prev_cum = 0.0
        for i in range(len(turn_ops)):
            r = _execute(turn_ops[: i + 1])
            cum = _cumulative(r)
            delta = cum - prev_cum
            turn_log.append({
                "op": turn_ops[i],
                "delta": delta,
                "cumulative": cum,
            })
            prev_cum = cum

        # Get normalized episode reward
        r_final = _execute(ops_episode)
        episode_reward = _normalize_reward(r_final)

        print(f"\nEpisode reward (normalized): {episode_reward:.4f}")
        for entry in turn_log:
            op = entry["op"]
            kind = op.get("kind", "")
            action = op.get("action", {}).get("type", "") if "action" in op else ""
            print(f"  {kind}.{action:<20} delta={entry['delta']:.4f}")

        # Build branch_rewards: primary delta + 3 zero alternatives
        branch_rewards = [[entry["delta"], 0.0, 0.0, 0.0] for entry in turn_log]

        # --- Without EMA baseline (group_size=1 behavior) ---
        state_no_ema: dict[str, object] = {
            "advantage": 0.0,
            "reward": episode_reward,
        }
        _compute_tl_grpo_advantages([state_no_ema], [branch_rewards], turn_weight=0.5)
        advs_no_ema = cast(list[float], state_no_ema["turn_advantages"])

        # --- With EMA baseline ---
        state_ema: dict[str, object] = {
            "advantage": 0.0,
            "reward": episode_reward,
        }
        ema_baseline = 0.5
        _compute_tl_grpo_advantages(
            [state_ema], [branch_rewards],
            turn_weight=0.5, outcome_baseline=ema_baseline,
        )
        advs_ema = cast(list[float], state_ema["turn_advantages"])

        print(f"\nAdvantages per turn (episode_reward={episode_reward:.4f}, baseline={ema_baseline}):")
        print(f"  {'Operation':<25} {'Delta':>8} {'No-EMA':>10} {'With-EMA':>10}")
        for i, entry in enumerate(turn_log):
            op = entry["op"]
            kind = op.get("kind", "")
            action = op.get("action", {}).get("type", "") if "action" in op else ""
            label = f"{kind}.{action}" if action else kind
            print(f"  {label:<25} {entry['delta']:>8.4f} {advs_no_ema[i]:>10.4f} {advs_ema[i]:>10.4f}")

        # --- Assertions ---
        expected_outcome = 0.5 * (episode_reward - ema_baseline)

        # 1. set_price (turn 0): delta=0, no-EMA advantage MUST be 0
        assert abs(turn_log[0]["delta"]) < 1e-10, "set_price delta should be 0"
        assert abs(advs_no_ema[0]) < 1e-6, (
            f"set_price advantage without EMA should be 0, got {advs_no_ema[0]}"
        )

        # 2. set_price with EMA: advantage MUST be nonzero (= turn_weight * outcome)
        assert abs(advs_ema[0] - expected_outcome) < 1e-6, (
            f"set_price advantage with EMA should be {expected_outcome:.6f}, "
            f"got {advs_ema[0]:.6f}"
        )
        assert abs(advs_ema[0]) > 0.001, (
            f"set_price advantage with EMA should be nonzero, got {advs_ema[0]}"
        )

        # 3. accept_customer (turn 2): should have BOTH local + outcome signal
        assert abs(turn_log[2]["delta"]) > 0.01, "accept_customer delta should be nonzero"
        assert abs(advs_ema[2]) > abs(advs_ema[0]), (
            "accept_customer advantage should be larger than set_price "
            "(local + outcome > outcome alone)"
        )

        # 4. All EMA advantages should be nonzero when episode_reward != baseline
        if abs(episode_reward - ema_baseline) > 0.01:
            for i in range(len(turn_log)):
                assert abs(advs_ema[i]) > 0.001, (
                    f"Turn {i}: advantage should be nonzero with EMA, got {advs_ema[i]}"
                )

        print(f"\n  set_price no-EMA advantage: {advs_no_ema[0]:.6f} (expected: 0) ✓")
        print(f"  set_price EMA advantage:    {advs_ema[0]:.6f} (expected: {expected_outcome:.6f}) ✓")
        print(f"  accept advantage > set_price advantage: {abs(advs_ema[2]):.4f} > {abs(advs_ema[0]):.4f} ✓")
        print(f"  All turns have nonzero EMA advantage ✓")

    def test_ema_update_adapts_baseline(self):
        """Verify EMA baseline math: decay * old + (1-decay) * reward."""
        ema = 0.5
        decay = 0.9
        rewards = [0.8, 0.3, 0.9, 0.1, 0.6]

        for r in rewards:
            ema = decay * ema + (1 - decay) * r

        # Manual computation
        expected = 0.5
        for r in rewards:
            expected = 0.9 * expected + 0.1 * r

        assert abs(ema - expected) < 1e-12
        # After these rewards, EMA should have moved from 0.5 toward the mean (~0.54)
        assert 0.4 < ema < 0.6, f"EMA should be near 0.5, got {ema}"
        print(f"\nEMA after 5 updates: {ema:.6f} (started at 0.5)")
