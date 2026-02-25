#!/usr/bin/env python3
"""Semantic planning diagnostic v2 — data-driven anchor discovery.

Instead of predefined anchors, this script:
1. Extracts all sentence-level chunks from completions
2. Clusters them by embedding similarity
3. Shows the top clusters to understand what patterns exist
4. Tests broader planning concepts tuned to math reasoning

Usage:
    uv run --isolated --with sentence-transformers python3 scripts/semantic_planning_v2.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np


def extract_sentences(text: str, min_words: int = 5) -> list[str]:
    """Split text into sentence-like chunks, filtering very short ones."""
    # Split on sentence boundaries, markdown headers, "---", etc.
    chunks = re.split(r'(?:\n\n+|---+|\n#{1,4}\s|(?<=[.!?])\s+(?=[A-Z]))', text)
    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        # Remove markdown artifacts
        chunk = re.sub(r'[#*_`]+', ' ', chunk)
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if len(chunk.split()) >= min_words:
            result.append(chunk)
    return result


def main():
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

    # Sample 200 completions
    sample_size = min(200, len(completions))
    step = max(1, len(completions) // sample_size)
    sampled = completions[::step][:sample_size]
    print(f"Sampled {len(sampled)} completions")

    # Extract all sentence chunks
    all_sentences = []
    sentence_sources = []  # (comp_idx, chunk_idx)
    for ci, text in enumerate(sampled):
        chunks = extract_sentences(text)
        for si, sent in enumerate(chunks):
            all_sentences.append(sent[:200])  # cap length for embedding
            sentence_sources.append((ci, si))

    print(f"Extracted {len(all_sentences)} sentence chunks")

    # Load model
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed everything
    print("Embedding all chunks...")
    embs = model.encode(all_sentences, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

    # --- Part 1: Broader planning anchors tuned for math ---
    print("\n" + "=" * 60)
    print("PART 1: Math-tuned planning anchors")
    print("=" * 60)

    # These anchors reflect how math solutions express planning/metacognition
    math_planning_anchors = [
        # Metacognitive / reflective
        "let me reconsider this approach",
        "this doesn't seem right, let me try again",
        "wait, I made an error in the previous step",
        "let me verify this result by checking",
        "going back to reconsider the strategy",
        # Strategic / planning
        "the key insight here is that",
        "notice that we can use the fact that",
        "a clever trick is to substitute",
        "we can simplify by observing that",
        "the idea is to use symmetry",
        "let's try a different approach",
        "instead of computing directly, we can",
        "we should consider what happens when",
        "to solve this, the strategy is",
        "the crucial observation is that",
        # Structural / organizing
        "step 1: understand the problem",
        "first, let's set up the equation",
        "now we need to find",
        "the goal is to compute",
        "our plan is to first find and then substitute",
    ]

    math_execution_anchors = [
        "substituting x equals 2 into the equation",
        "expanding the left side gives",
        "simplifying the fraction yields",
        "therefore the answer is 42",
        "computing 3 plus 5 equals 8",
        "factoring out x from both terms",
        "taking the derivative of x squared gives 2x",
        "by the quadratic formula x equals",
        "multiplying both sides by 2",
        "solving for x we get x equals 7",
        "the discriminant is b squared minus 4ac",
        "so the sum is equal to 15",
        "plugging in the values we obtain",
        "cross-multiplying gives us",
        "collecting like terms on the left",
    ]

    plan_embs = model.encode(math_planning_anchors, normalize_embeddings=True)
    exec_embs = model.encode(math_execution_anchors, normalize_embeddings=True)
    plan_centroid = np.mean(plan_embs, axis=0)
    plan_centroid /= np.linalg.norm(plan_centroid)
    exec_centroid = np.mean(exec_embs, axis=0)
    exec_centroid /= np.linalg.norm(exec_centroid)

    # Classify each chunk
    plan_sims = embs @ plan_centroid
    exec_sims = embs @ exec_centroid

    # Relative classification: planning if plan_sim > exec_sim + margin
    margin = 0.02
    planning_chunks = []
    execution_chunks = []
    for i in range(len(all_sentences)):
        if plan_sims[i] > exec_sims[i] + margin and plan_sims[i] > 0.25:
            planning_chunks.append(i)
        else:
            execution_chunks.append(i)

    print(f"\nPlanning chunks: {len(planning_chunks)}/{len(all_sentences)} ({100*len(planning_chunks)/len(all_sentences):.1f}%)")
    print(f"Execution chunks: {len(execution_chunks)}/{len(all_sentences)} ({100*len(execution_chunks)/len(all_sentences):.1f}%)")

    # How many completions have at least one planning chunk?
    comps_with_planning = set()
    for idx in planning_chunks:
        comps_with_planning.add(sentence_sources[idx][0])
    print(f"Completions with planning: {len(comps_with_planning)}/{len(sampled)} ({100*len(comps_with_planning)/len(sampled):.1f}%)")

    # Show top planning examples
    planning_scored = [(i, plan_sims[i], exec_sims[i]) for i in planning_chunks]
    planning_scored.sort(key=lambda x: x[1] - x[2], reverse=True)

    print(f"\nTop 20 most 'planning-like' chunks:")
    for idx, ps, es in planning_scored[:20]:
        text = all_sentences[idx][:100]
        print(f"  plan={ps:.3f} exec={es:.3f} delta={ps-es:.3f} | \"{text}\"")

    # Show top execution examples for contrast
    exec_scored = [(i, exec_sims[i], plan_sims[i]) for i in execution_chunks]
    exec_scored.sort(key=lambda x: x[1] - x[2], reverse=True)
    print(f"\nTop 10 most 'execution-like' chunks:")
    for idx, es, ps in exec_scored[:10]:
        text = all_sentences[idx][:100]
        print(f"  exec={es:.3f} plan={ps:.3f} delta={es-ps:.3f} | \"{text}\"")

    # --- Part 2: Distribution analysis ---
    print("\n" + "=" * 60)
    print("PART 2: Similarity distribution")
    print("=" * 60)

    deltas = plan_sims - exec_sims
    print(f"plan_sim - exec_sim distribution:")
    print(f"  min={deltas.min():.3f}  p10={np.percentile(deltas, 10):.3f}  median={np.median(deltas):.3f}  p90={np.percentile(deltas, 90):.3f}  max={deltas.max():.3f}")
    print(f"  mean={deltas.mean():.3f}  std={deltas.std():.3f}")

    # What fraction is clearly planning (delta > 0.05)?
    strong_plan = np.sum(deltas > 0.05) / len(deltas)
    moderate_plan = np.sum((deltas > 0.02) & (deltas <= 0.05)) / len(deltas)
    neutral = np.sum(np.abs(deltas) <= 0.02) / len(deltas)
    moderate_exec = np.sum((deltas < -0.02) & (deltas >= -0.05)) / len(deltas)
    strong_exec = np.sum(deltas < -0.05) / len(deltas)

    print(f"\n  Strong planning (delta > 0.05): {100*strong_plan:.1f}%")
    print(f"  Moderate planning (0.02 < delta <= 0.05): {100*moderate_plan:.1f}%")
    print(f"  Neutral (|delta| <= 0.02): {100*neutral:.1f}%")
    print(f"  Moderate execution (-0.05 <= delta < -0.02): {100*moderate_exec:.1f}%")
    print(f"  Strong execution (delta < -0.05): {100*strong_exec:.1f}%")

    # --- Part 3: What does the model's "planning" actually look like? ---
    print("\n" + "=" * 60)
    print("PART 3: Common planning phrases in completions")
    print("=" * 60)

    # Check for common math-reasoning metacognitive patterns via regex
    meta_patterns = {
        "Step N:": r"(?:Step|STEP)\s+\d+",
        "Let's / Let us": r"\b[Ll]et(?:'s| us)\b",
        "We need to": r"\b[Ww]e (?:need|want|must) to\b",
        "Note/Notice/Observe that": r"\b(?:[Nn]ote|[Nn]otice|[Oo]bserve) that\b",
        "Recall/Remember that": r"\b(?:[Rr]ecall|[Rr]emember) that\b",
        "The key/idea/trick is": r"\b[Tt]he (?:key|idea|trick|insight|crucial)\b",
        "Try/Consider": r"\b(?:[Tt]ry|[Cc]onsider)(?:ing)?\b",
        "Wait/Hmm/Hm": r"\b(?:[Ww]ait|[Hh]mm+|[Hh]m)\b",
        "First/Now/Next/Finally": r"\b(?:First|Now|Next|Finally|Then),?\s",
        "This means/implies/gives": r"\b[Tt]his (?:means|implies|gives|suggests|tells)\b",
        "We can/could/should": r"\b[Ww]e (?:can|could|should|might)\b",
        "So/Therefore/Thus/Hence": r"\b(?:So|Therefore|Thus|Hence),?\s",
        "In other words": r"\bin other words\b",
        "Alternatively": r"\b[Aa]lternatively\b",
    }

    print(f"\nMath-reasoning metacognitive patterns in {len(sampled)} completions:")
    for label, pat_str in meta_patterns.items():
        pat = re.compile(pat_str)
        match_count = sum(len(pat.findall(text)) for text in sampled)
        comp_count = sum(1 for text in sampled if pat.search(text))
        print(f"  {label:35s}: {match_count:5d} matches in {comp_count:3d}/{len(sampled)} completions")


if __name__ == "__main__":
    main()
