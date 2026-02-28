# Experiment Tree

The experiment tech tree organises campaigns into a dependency graph. Each node wraps a campaign TOML and has a success condition — when a node completes, its outcome (success or failure) unlocks the next set of experiments automatically.

Use it when you have a branching research plan: "if ratio clipping works, try a sweep; if it fails, fall back to KL penalty."

## Quick start

Define a tree in TOML:

```toml
[tree]
name = "post-sepa"
description = "Experiments after the SEPA null result"

[[tree.nodes]]
id = "ratio-clip"
label = "PPO ratio clipping"
campaign = "campaigns/ratio-clip.toml"
success = "correct_rate > 0.55"
on_success = ["clip-sweep"]
on_failure = ["kl-penalty"]

[[tree.nodes]]
id = "clip-sweep"
label = "Clip epsilon sweep"
campaign = "campaigns/clip-sweep.toml"
success = "correct_rate > 0.60"

[[tree.nodes]]
id = "kl-penalty"
label = "KL penalty baseline"
campaign = "campaigns/kl-penalty.toml"
success = "correct_rate > 0.50"
```

View it:

```bash
retrain tree campaigns/tree.toml
```

```
post-sepa: Experiments after the SEPA null result

[ ] ratio-clip  "PPO ratio clipping"
     success -> [#] clip-sweep  "Clip epsilon sweep"
     failure -> [#] kl-penalty  "KL penalty baseline"
```

## TOML reference

### `[tree]` section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | str | file stem | Tree name shown in header |
| `description` | str | `""` | One-line description |

### `[[tree.nodes]]`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `id` | str | yes | Unique short identifier |
| `label` | str | no | Human-readable description (defaults to id) |
| `campaign` | str | yes | Path to campaign TOML |
| `notes` | str | no | Free-text notes |
| `refs` | list[str] | no | Reference strings (papers, URLs) |
| `success` | str | no | Condition: `"metric op value"` |
| `on_success` | list[str] | no | Nodes to unlock on success |
| `on_failure` | list[str] | no | Nodes to unlock on failure |
| `related` | list[str] | no | Informational cross-references |

**Success condition format:** `"metric op threshold"` where metric is one of `correct_rate`, `loss`, `mean_reward`; op is `>`, `>=`, `<`, `<=`, `==`; threshold is a float.

## CLI commands

### View the tree

```bash
retrain tree campaigns/tree.toml          # ASCII art
retrain tree campaigns/tree.toml --json   # JSON output
```

### See what's ready to run

```bash
retrain tree next campaigns/tree.toml
```

```
Ready to run:
  ratio-clip: campaigns/ratio-clip.toml
    "PPO ratio clipping" — Highest priority. clip_eps=0.2 on local GPU.
```

### Inspect a node

```bash
retrain tree show ratio-clip campaigns/tree.toml
retrain tree show ratio-clip campaigns/tree.toml --json
```

### Run a node's campaign

```bash
retrain tree run ratio-clip campaigns/tree.toml
```

### Evaluate results

```bash
retrain tree eval campaigns/tree.toml
```

Averages the success metric across all completed campaign runs and compares against the threshold.

### Add a note

```bash
retrain tree note ratio-clip "Plateau broken!" campaigns/tree.toml
```

Annotations are timestamped and preserved across resets.

### Reset a node

```bash
retrain tree reset ratio-clip campaigns/tree.toml
```

Resets status to pending, clears outcome/result/timestamps. Annotations are preserved.

## State management

Tree state lives in `tree_state.json` beside the tree TOML (e.g. `campaigns/tree_state.json`). It tracks:

- Per-node status, outcome, campaign directory, results
- Timestamps (started_at, completed_at)
- Annotations (timestamped lab notebook entries)

Writes are atomic (temp file + rename). Delete the state file to start fresh — all nodes revert to pending.

### Status icons

| Icon | Status | Meaning |
|------|--------|---------|
| `[ ]` | pending | Ready to run |
| `[#]` | locked | Waiting for parent to complete |
| `[~]` | running | Campaign in progress |
| `[✓]` | done/success | Succeeded |
| `[✗]` | done/failure | Failed |
| `[-]` | skipped | Parent completed on wrong branch |

## Workflow

1. **Define** your experiment tree in TOML
2. **View** the tree: `retrain tree campaigns/tree.toml`
3. **Run** the next pending node: `retrain tree run ratio-clip campaigns/tree.toml`
4. **Evaluate** results: `retrain tree eval campaigns/tree.toml`
5. **Check next** steps: `retrain tree next campaigns/tree.toml` — children are now unlocked based on outcome
6. **Repeat** until all branches are explored
