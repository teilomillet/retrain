# Tinker Forecasting Note

Reference note for soma / retrain based on Thinking Machines' post on training LLMs to predict world events on Tinker.

## What transfers cleanly

- **Backend alignment matters.** Their observation about sampling and training backends disagreeing on token probabilities maps directly onto retrain's backend concerns. Using Tinker for both sampling and training reduces that mismatch compared with split vLLM/FSDP setups.
- **Bounded rewards are good default engineering.** The exact Brier-score argument is forecasting-specific, but the variance argument carries over: bounded, monotone rewards are easier to optimize than unbounded rewards.
- **Small groups are plausible when ties are rare.** The useful question for soma is not "should we use Brier?" but "how often do rollouts inside one prompt group collapse onto the same reward?"

## What does not transfer directly

- **Brier vs log score is not the main lesson for soma.** That is specific to single-turn probability forecasting. Soma is a sequential control problem with delayed credit assignment.
- **This is not evidence for our current REINFORCE++ normalization choice.** The warehouse REINFORCE++ path still applies batch standard-deviation normalization after group centering, so this post should not be read as validating that exact setup.
- **Energy can still saturate early.** Safety-violation rollouts in energy can hit the same floor reward, which makes tie statistics more important than they are in the forecasting setting.

## Practical use in soma

Current group sizes in repo configs:

- `configs/retrain/soma-energy-tinker.toml`: `group_size = 4`
- `retrain/campaigns/warehouse-rl.toml`: `group_size = 8`
- Generic retrain default: `group_size = 16`

When sweeping `group_size`, watch these metrics in `metrics.jsonl` or wandb:

- `reward_tie_group_rate`: fraction of eligible prompt groups with any repeated reward
- `reward_uniform_group_rate`: fraction of eligible prompt groups where all rewards tie
- `reward_tie_pair_rate`: fraction of within-group reward pairs that tie
- `reward_unique_fraction_mean`: mean distinct-reward fraction inside a group

Recommended ablation:

1. Sweep `group_size` across `{4, 8, 16}`.
2. Compare reward-tie metrics alongside `mean_reward`, `correct_rate`, and wall-clock throughput.
3. For energy, separately inspect how much mass sits on the safety floor before concluding that ties are intrinsically common.
