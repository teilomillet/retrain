#!/usr/bin/env python3
"""Semantic planning token diagnostic.

Compares the current hardcoded strategic grams with a semantic (embedding-based)
approach using sentence-transformers. Analyzes actual completions from SEPA
experiment runs to determine how many tokens are classified as "planning".

Usage:
    # Run from an isolated venv (to avoid torchvision conflicts):
    uv run --isolated --with sentence-transformers python3 scripts/semantic_planning_diagnostic.py
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Current hardcoded grams (copied from retrain/advantages.py)
# ---------------------------------------------------------------------------

DEFAULT_STRATEGIC_GRAMS = [
    "wait let me",
    "let me think",
    "on second thought",
    "let me check",
    "let me verify",
    "is this right",
    "double check",
    "try another approach",
    "go back and",
    "start over",
    "that's not right",
    "that doesn't work",
    "another way to",
    "or we could",
    "what if we",
    "notice that",
    "the key is",
    "the key insight",
]


def count_gram_matches(text: str, grams: list[str]) -> dict[str, int]:
    """Count how many times each gram fires in the text."""
    counts: dict[str, int] = {}
    for gram in grams:
        pat = re.compile(r"\b" + re.escape(gram) + r"\b", re.IGNORECASE)
        n = len(pat.findall(text))
        if n > 0:
            counts[gram] = n
    return counts


# ---------------------------------------------------------------------------
# 2. Semantic planning detection via sentence-transformers
# ---------------------------------------------------------------------------

# Planning concept anchors: semantic descriptions of what planning looks like
PLANNING_ANCHORS = [
    "reconsidering the approach and trying a different method",
    "pausing to verify whether the previous step was correct",
    "realizing an error and going back to fix it",
    "thinking about which strategy would work best here",
    "checking the work by substituting the answer back",
    "noticing a pattern or key insight that changes the approach",
    "deciding between two possible solution paths",
    "reflecting on whether the current direction is productive",
    "stepping back to reconsider the problem from scratch",
    "planning the next sequence of algebraic manipulations",
]

# Execution concept anchors: what straightforward computation looks like
EXECUTION_ANCHORS = [
    "substituting a value into an equation and simplifying",
    "performing algebraic manipulation step by step",
    "computing a numerical result from a formula",
    "applying a known theorem or identity directly",
    "writing the final answer in the required format",
    "expanding brackets and collecting like terms",
    "taking the derivative or integral of a function",
    "solving a system of linear equations",
    "factoring a polynomial expression",
    "evaluating a sum or product",
]


def run_semantic_analysis(completions: list[str], window_words: int = 12):
    """Run semantic planning detection on completions."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed anchors
    plan_embs = model.encode(PLANNING_ANCHORS, normalize_embeddings=True)
    exec_embs = model.encode(EXECUTION_ANCHORS, normalize_embeddings=True)
    plan_centroid = np.mean(plan_embs, axis=0)
    plan_centroid /= np.linalg.norm(plan_centroid)
    exec_centroid = np.mean(exec_embs, axis=0)
    exec_centroid /= np.linalg.norm(exec_centroid)

    # Stats
    total_windows = 0
    planning_windows = 0
    total_completions = len(completions)
    completions_with_planning = 0
    planning_examples: list[tuple[str, float, float]] = []  # (text, plan_sim, exec_sim)

    for comp_idx, text in enumerate(completions):
        words = text.split()
        if len(words) < window_words:
            continue

        comp_planning_count = 0
        windows_in_comp = 0

        # Slide a window across the text
        step_size = window_words // 2  # 50% overlap
        window_texts = []
        for start in range(0, len(words) - window_words + 1, step_size):
            window_text = " ".join(words[start : start + window_words])
            window_texts.append(window_text)

        if not window_texts:
            continue

        # Batch encode all windows at once
        window_embs = model.encode(window_texts, normalize_embeddings=True, batch_size=256)
        plan_sims = window_embs @ plan_centroid
        exec_sims = window_embs @ exec_centroid

        for i, (ps, es) in enumerate(zip(plan_sims, exec_sims)):
            total_windows += 1
            windows_in_comp += 1
            # Planning if: closer to planning centroid AND above absolute threshold
            if ps > es and ps > 0.35:
                planning_windows += 1
                comp_planning_count += 1
                if len(planning_examples) < 30:
                    planning_examples.append((window_texts[i], float(ps), float(es)))

        if comp_planning_count > 0:
            completions_with_planning += 1

        if (comp_idx + 1) % 100 == 0:
            print(f"  Processed {comp_idx + 1}/{total_completions} completions...")

    return {
        "total_windows": total_windows,
        "planning_windows": planning_windows,
        "planning_rate": planning_windows / max(total_windows, 1),
        "total_completions": total_completions,
        "completions_with_planning": completions_with_planning,
        "comp_with_planning_rate": completions_with_planning / max(total_completions, 1),
        "examples": planning_examples,
    }


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    # Load completions from the baseline run
    gen_path = Path("logs/campaign_20260222_085036/runs/grpo+none_s42/emergence/generations.jsonl")
    if not gen_path.exists():
        print(f"ERROR: {gen_path} not found")
        sys.exit(1)

    print(f"Loading completions from {gen_path}...")
    completions = []
    with open(gen_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                completions.append(obj["completion"])

    # Sample 500 completions (spread across steps)
    sample_size = min(500, len(completions))
    step = max(1, len(completions) // sample_size)
    sampled = completions[::step][:sample_size]
    print(f"Sampled {len(sampled)} completions from {len(completions)} total")

    # --- Part A: Hardcoded gram analysis ---
    print("\n" + "=" * 60)
    print("PART A: Hardcoded Strategic Grams (current system)")
    print("=" * 60)

    total_matches = 0
    comps_with_match = 0
    gram_totals: dict[str, int] = {}
    for text in sampled:
        counts = count_gram_matches(text, DEFAULT_STRATEGIC_GRAMS)
        if counts:
            comps_with_match += 1
        for gram, n in counts.items():
            total_matches += n
            gram_totals[gram] = gram_totals.get(gram, 0) + n

    print(f"\nCompletions analyzed: {len(sampled)}")
    print(f"Completions with >= 1 gram match: {comps_with_match} ({100*comps_with_match/len(sampled):.1f}%)")
    print(f"Total gram matches: {total_matches}")
    print("\nPer-gram breakdown:")
    for gram in sorted(gram_totals, key=gram_totals.get, reverse=True):
        print(f"  '{gram}': {gram_totals[gram]} matches")
    unmatched = [g for g in DEFAULT_STRATEGIC_GRAMS if g not in gram_totals]
    if unmatched:
        print(f"\nGrams that NEVER fired ({len(unmatched)}/{len(DEFAULT_STRATEGIC_GRAMS)}):")
        for g in unmatched:
            print(f"  '{g}'")

    # --- Part B: Semantic analysis ---
    print("\n" + "=" * 60)
    print("PART B: Semantic Embedding Analysis (proposed replacement)")
    print("=" * 60)

    results = run_semantic_analysis(sampled)

    print(f"\nTotal windows analyzed: {results['total_windows']}")
    print(f"Planning windows: {results['planning_windows']} ({100*results['planning_rate']:.2f}%)")
    print(f"Completions with planning: {results['completions_with_planning']}/{results['total_completions']} ({100*results['comp_with_planning_rate']:.1f}%)")

    print(f"\nImprovement over hardcoded grams:")
    print(f"  Hardcoded: {comps_with_match}/{len(sampled)} completions ({100*comps_with_match/len(sampled):.1f}%)")
    print(f"  Semantic:  {results['completions_with_planning']}/{results['total_completions']} completions ({100*results['comp_with_planning_rate']:.1f}%)")

    if results["examples"]:
        print(f"\nTop planning window examples (showing up to 15):")
        sorted_examples = sorted(results["examples"], key=lambda x: x[1], reverse=True)
        for text, ps, es in sorted_examples[:15]:
            print(f"  plan_sim={ps:.3f} exec_sim={es:.3f} | \"{text[:80]}...\"")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    multiplier = (results['comp_with_planning_rate'] / max(comps_with_match / len(sampled), 0.001))
    print(f"Semantic detection finds {multiplier:.1f}x more planning-containing completions than hardcoded grams")
    print(f"Planning window rate: {100*results['planning_rate']:.2f}% of all windows")

    if results['planning_rate'] > 0.05:
        print("\n-> Semantic detection finds substantial planning signal.")
        print("-> SEPA/HICRA experiments likely failed because the hardcoded grams missed 95%+ of planning tokens.")
        print("-> RECOMMENDATION: Replace hardcoded grams with semantic detection and re-run SEPA experiment.")
    elif results['planning_rate'] > 0.01:
        print("\n-> Semantic detection finds moderate planning signal.")
        print("-> Worth re-running SEPA with semantic detection to see if it makes a difference.")
    else:
        print("\n-> Even semantic detection finds very little planning signal.")
        print("-> The model may genuinely not produce planning tokens, or the anchors need refinement.")


if __name__ == "__main__":
    main()
