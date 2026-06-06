# Stage 2 Baseline Experiment Pipeline

The baseline experiment runner creates a reproducible supervised RL workflow in
one command. It generates JSONL data, prepends deterministic canonical depth 1-2
states, removes duplicate cube states, trains the linear and MLP policies, runs
both checkpoints against identical seeded environment rollouts, and writes a
single JSON report.

## Command

Run from `rl/` after installing the backend and RL packages:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.training.baseline_experiment --depths 1 2 3 --samples-per-depth 100 --evaluation-samples-per-depth 100 --evaluation-max-steps 30 --seed 20260531
```

Default outputs are ignored by source control:

- `datasets/baseline-depth-1-2-3.jsonl`
- `checkpoints/linear-baseline-depth-1-2-3.npz`
- `checkpoints/lightweight-linear-axis-samples-28-heldout/linear-policy.npz`
- `reports/baseline-experiment.json`

## Report Shape

The report contains:

- `dataset`: record counts, trainable records, random and canonical depths,
  deduplicated record count, and seed.
- `training.linear` and `training.mlp`: hyperparameters, model paths, and train
  or validation metrics.
- `comparison`: solve-rate summaries and ranking for both checkpoints.
- `backend`: top-ranked checkpoint label plus recommended `RUBIC_RL_MODEL_PATH`
  and `RUBIC_RL_MAX_STEPS` values.

## Backend Use

Use the top-ranked checkpoint from the report for `/solve/rl`. With the current
default hyperparameters and canonical depth 1-2 data, the MLP checkpoint is
ranked first:

```powershell
$env:RUBIC_RL_MODEL_PATH="checkpoints\lightweight-linear-axis-samples-28-heldout\linear-policy.npz"
$env:RUBIC_RL_MAX_STEPS="30"
```

Retrain with higher `--samples-per-depth`, `--mlp-hidden-units`, or
`--mlp-epochs` before testing deeper scrambles in the UI. Pass
`--canonical-depths none` when you need a random-only baseline comparison.
