# SEPA Experiment Loop

You are running SEPA experiments for the retrain project. Your goal is to collect
data that answers: **does SEPA at full pooling strength (λ=1.0) separate from baseline?**

Reference: teilo.xyz/posts/sepa — the key finding was λ never exceeded 0.10.

## Rules

- NEVER modify core retrain code (retrain/, src/, tests/)
- ONLY create/edit files in campaigns/, logs/, and this file
- NEVER run experiments beyond what's defined in campaign TOMLs
- Budget is controlled by the TOML — do not override max_steps or add seeds via CLI
- Always check git status and existing logs before doing anything

## Phase detection

Check the current state and pick the right phase:

### Phase 1: Run sepa-ramp (if no campaign logs exist yet)

```bash
retrain campaigns/sepa-ramp.toml
```

This runs 2 conditions × 3 seeds × 100 steps = 6 runs.
Auto-squeeze runs after the first run (rank 128 → finds optimal rank).

When done, write results to `logs/sepa-ramp-summary.md` with:
- Per-condition mean correct_rate at steps 20, 50, 80, 100
- Squeeze recommended rank
- Whether conditions separated (>2% gap at step 100)
- exec_entropy_var comparison between conditions

Then move to Phase 2.

### Phase 2: Analyze results (if campaign logs exist but no summary)

Read the campaign's metrics.jsonl files and write `logs/sepa-ramp-summary.md`.
Compare grpo+none vs maxrl+gtpo_sepa on:
- correct_rate trajectory (do they diverge after λ ramps up?)
- exec_entropy_var (SEPA should reduce this by >50%)
- sepa_lambda values (confirm λ reached >0.8)

### Phase 3: Decide next step (if summary exists)

Read `logs/sepa-ramp-summary.md` and decide:

**If separation found (>2% correct_rate gap, p<0.10 on Fisher exact):**
→ Create `campaigns/sepa-ablation.toml` adding the HICRA condition (3 conditions × 3 seeds × 100 steps)
  Use the squeeze-recommended rank instead of 128.

**If no separation but mechanism confirmed (exec_entropy_var reduced but no correct_rate lift):**
→ Create `campaigns/sepa-longer.toml` with same 2 conditions but max_steps=200
  and sepa_steps=200 to see if the effect needs more time.
  Use the squeeze-recommended rank.

**If nothing works (no entropy reduction, no separation):**
→ Write conclusion to summary. Something is wrong with the setup.

### Phase 4: Run follow-up (if follow-up TOML exists but hasn't been run)

Run the follow-up campaign, analyze, write updated summary.

### Phase 5: Done

When you have enough data to answer the question, output:

<promise>SEPA EXPERIMENTS COMPLETE</promise>

## Budget guardrails

- sepa-ramp: 6 runs (76k episodes) — the cheap first experiment
- sepa-ablation: 9 runs (115k episodes) — only if Phase 1 shows signal
- sepa-longer: 6 runs (153k episodes) — only if mechanism works but needs time
- Maximum total: 2 campaigns. Never create a third without human approval.
