# RL Research Track

This directory contains the DeepCubeA-inspired research track.

## Current Contents

- `rubic_rl/envs/rubiks_cube.py` implements `RubiksCubeEnv`.
- `rubic_rl/encoding.py` converts cube stickers to one-hot or index observations.
- `rubic_rl/datasets/generator.py` creates supervised JSONL training data.
- `rubic_rl/datasets/canonical.py` creates exhaustive short-depth records.
- `rubic_rl/policies/linear_policy.py` trains a NumPy softmax baseline policy.
- `rubic_rl/policies/mlp_policy.py` trains a NumPy MLP baseline policy.
- `rubic_rl/policies/torch_policy.py` adapts optional PyTorch policy/value
  checkpoints for evaluation and search.
- `rubic_rl/models/policy_value.py` defines the optional PyTorch residual
  policy/value network.
- `rubic_rl/evaluation/policy_eval.py` evaluates policy checkpoints in the environment.
- `rubic_rl/evaluation/compare_policies.py` compares and ranks checkpoints.
- `rubic_rl/evaluation/search_eval.py` benchmarks greedy rollout against beam search.
- `rubic_rl/training/baseline_experiment.py` runs dataset generation, training,
  comparison, and report writing in one command.
- `rubic_rl/training/adi.py` trains the optional PyTorch policy/value baseline.
- `tests/` verifies the environment, datasets, supervised policy, and evaluator.

The RL track should consume the shared cube engine behavior but remain isolated
from the production API until baseline performance is measurable.

## Setup

Install the backend first so the shared cube engine is importable, then install
the RL package:

```powershell
cd backend
python -m pip install -e ".[dev]"
cd ..\rl
python -m pip install -e ".[dev]"
```

Install the optional PyTorch training path when you need the policy/value
network:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pip install -e ".[torch,dev]"
```

Run the RL tests:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

Run a tracked experiment from JSON config:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
.\.venv\Scripts\python.exe -m rubic_rl.training.run_experiment --config rl\experiments\baseline-smoke.json
```

This writes standardized outputs under `datasets/<run-name>/`,
`checkpoints/<run-name>/`, and `reports/<run-name>/`, including a
`manifest.json` with the run config, outputs, git commit, and headline metrics.

For the recommended lightweight path, start with the small imitation-style
policy track instead of ADI or DAVI:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
.\.venv\Scripts\python.exe -m rubic_rl.training.run_experiment --config rl\experiments\lightweight-mlp-smoke.json
```

Then run the broader shallow benchmark:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
.\.venv\Scripts\python.exe -m rubic_rl.training.run_experiment --config rl\experiments\lightweight-mlp-shallow.json
```

These configs keep the model small, keep the dataset cheap to generate, and
benchmark against existing baseline runs when those checkpoints are present.

Use the lightweight smoke linear checkpoint as the initial reference candidate:

```text
checkpoints/lightweight-mlp-smoke/linear-policy.npz
```

Use the held-out shallow case set at:

```text
rl/benchmarks/heldout-shallow.json
```

when you want to validate a candidate without reusing the earlier smoke or
shallow benchmark cases.

When the smoke run becomes the incumbent, use the promotion-oriented linear
variant to try to beat it on the shallow suite:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
.\.venv\Scripts\python.exe -m rubic_rl.training.run_experiment --config rl\experiments\lightweight-linear-promote.json
```

If you want the same benchmark and manifest flow without paying for MLP
training, use the linear-only experiment type:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
.\.venv\Scripts\python.exe -m rubic_rl.training.run_experiment --config rl\experiments\lightweight-linear-only-promote.json
```

To compare a few cheap linear-only candidates in one pass, run the sweep
entrypoint with the curated configs:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
.\.venv\Scripts\python.exe -m rubic_rl.training.run_sweep --name lightweight-linear-sweep --config rl\experiments\lightweight-linear-sweep-balanced.json --config rl\experiments\lightweight-linear-sweep-data-heavy.json --config rl\experiments\lightweight-linear-sweep-depth-4.json
```

This writes `reports/<sweep-name>/summary.json`, ranks the candidates by the
promotion gate first, then by solve-rate delta and latency, and gives you one
place to inspect the current best linear checkpoint.

For narrower tuning, keep the held-out cases fixed and change only one training
axis at a time:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.training.run_sweep --name lightweight-linear-heldout-axis --config experiments\lightweight-linear-axis-samples-28-heldout.json --config experiments\lightweight-linear-axis-depth-4-heldout.json
```

Run a shared benchmark suite on persisted benchmark cases:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
.\.venv\Scripts\python.exe -m rubic_rl.evaluation.benchmark_suite --suite smoke --policy-model baseline=checkpoints\baseline-smoke\linear-policy.npz --cases-out reports\benchmark-smoke-cases.json --out reports\benchmark-smoke.json
```

This writes a common case set and a single report that ranks policy rollout,
policy-guided beam search, and optional DAVI weighted-A* entrants on the same
scrambles. Add `--davi-model label=path\to\checkpoint.pt` to include the value
track in the same benchmark run.

Experiment configs can also declare `benchmark.compare_runs` so a new run can
automatically compare against checkpoints from prior tracked runs. Missing
reference runs are skipped by default; set `require_compare_runs` when a
promotion gate should fail on missing incumbents.

Generate a small supervised dataset:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.datasets.generator --out ..\datasets\smoke.jsonl --depths 1 2 3 --samples-per-depth 10 --seed 20260531 --include-solved
```

Generate deterministic canonical depth 1-2 coverage:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.datasets.canonical --out ..\datasets\canonical-depth-1-2.jsonl --depths 1 2
```

Train the first supervised baseline:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.training.supervised --dataset ..\datasets\smoke.jsonl --model-out ..\checkpoints\linear-policy-smoke.npz --epochs 200 --learning-rate 0.2 --batch-size 128 --validation-split 0.2 --seed 20260531
```

Train the MLP supervised baseline:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.training.mlp_supervised --dataset ..\datasets\smoke.jsonl --model-out ..\checkpoints\mlp-policy-smoke.npz --hidden-units 64 --epochs 300 --learning-rate 0.05 --batch-size 128 --validation-split 0.2 --seed 20260531
```

Evaluate the checkpoint:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.policy_eval --model ..\checkpoints\mlp-policy-smoke.npz --policy-type auto --depths 1 2 3 --samples-per-depth 100 --max-steps 30 --seed 20260531
```

Compare checkpoints on identical seeded rollouts:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.compare_policies --model linear=..\checkpoints\linear-policy-smoke.npz --model mlp=..\checkpoints\mlp-policy-smoke.npz --depths 1 2 3 --samples-per-depth 100 --max-steps 30 --seed 20260531 --out ..\reports\policy-comparison.json
```

Benchmark greedy rollout against policy-guided beam search:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.search_eval --model ..\checkpoints\mlp-baseline-depth-1-2-3.npz --policy-type auto --depths 1 2 3 --samples-per-depth 100 --max-depth 30 --beam-width 5 --top-k 5 --seed 20260531 --out ..\reports\search-evaluation.json
```

Run the full supervised baseline experiment pipeline:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.training.baseline_experiment --depths 1 2 3 --samples-per-depth 100 --evaluation-samples-per-depth 100 --evaluation-max-steps 30 --seed 20260531
```

Train the PyTorch ADI-style policy/value baseline:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.training.adi --depths 1 2 3 --samples-per-depth 100 --iterations 1 --epochs-per-iteration 5 --seed 20260531
```

Evaluate the PyTorch ADI checkpoint:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.search_eval --model ..\checkpoints\torch-policy-value-adi.pt --policy-type auto --depths 1 2 3 --samples-per-depth 100 --max-depth 30 --beam-width 5 --top-k 5 --seed 20260531 --out ..\reports\adi-search-evaluation.json
```

Use the environment directly:

```python
from rubic_rl import RubiksCubeEnv

env = RubiksCubeEnv(scramble_depth=10, max_steps=100)
observation, info = env.reset(seed=20260531)
observation, reward, terminated, truncated, info = env.step(0)
```

## Action Space

The action space is `Discrete(18)` with this fixed move order:

```text
U, D, L, R, F, B,
U', D', L', R', F', B',
U2, D2, L2, R2, F2, B2
```

## Observation Space

Default observations are one-hot sticker tensors with shape `(54, 6)`.
Set `observation_mode="indices"` to receive a `(54,)` integer vector instead.

## Dataset Records

JSONL records include the cube `stickers`, integer `sticker_indices`,
`scramble`, inverse `target_moves`, and first-move `target_action`. Use
`target_action` for the first supervised policy baseline, then verify a full
sequence by applying `target_moves` to the source cube.

## Canonical Dataset

`generate_canonical_dataset` enumerates all axis-pruned depth 1-2 action
sequences from the solved state, deduplicates by sticker state, and writes the
same `DatasetRecord` format as the random generator. Use it to cover short
horizon states deterministically before adding random deeper scrambles.

## Supervised Baseline

`LinearPolicy` is a multinomial logistic-regression baseline implemented with
NumPy. It converts each cube state into a 324-value one-hot vector and predicts
one of the fixed 18 actions. This is intentionally small and deterministic so
later neural policies have a clear baseline to beat.

## Neural Baseline

`MLPPolicy` is a one-hidden-layer ReLU classifier implemented with NumPy. It
uses the same features, labels, and checkpoint format family as `LinearPolicy`,
which makes solve-rate comparisons reproducible before adding deeper RL
algorithms or external ML dependencies.

## PyTorch Policy/Value Baseline

`RubiksPolicyValueNet` is an optional residual PyTorch model with a policy head
over the fixed 18 actions and a scalar value head for normalized distance to the
solved state. `rubic_rl.training.adi` generates reverse-scrambled states,
trains policy and value losses together, and writes a `.pt` checkpoint with
model version metadata. See `docs/stage-2-adi-policy-value-training.md` for the
command and report format.

## Policy Evaluation

`evaluate_policy_on_env` runs a policy against seeded random scrambles and
`evaluate_policy_on_scrambles` runs explicit scrambles for controlled tests. The
CLI prints JSON solve-rate summaries grouped by scramble depth. Add
`--include-episodes` when you need per-rollout predicted moves and rewards.
Use `--policy-type auto` to detect linear, MLP, and Torch checkpoints
automatically.

## Policy Comparison

`compare_policies` evaluates multiple labeled checkpoints with the same
`PolicyEvaluationConfig`, then ranks them by solve rate, step count, and average
reward. Use `--model label=path` once per checkpoint and write reports under
`reports/` or another ignored output directory.

## Search Evaluation

`search_eval` runs greedy rollout and policy-guided beam search against the same
seeded scramble cases. The report includes solve rate, move count, latency,
expanded states, visited states, depth reached, and a strategy ranking. Use
`--include-episodes` when debugging individual failures or candidate paths.

## Baseline Experiment Pipeline

`baseline_experiment` is the preferred reproducible command for Stage 2
experiments. It writes the dataset, both checkpoint files, and a JSON report with
training metrics, rollout summaries, policy ranking, and backend environment
variables. It prepends canonical depth 1-2 records and deduplicates by cube state
by default; pass `--canonical-depths none` to compare against random-only data.
See `docs/stage-2-baseline-experiment-pipeline.md` for details.

## Backend Integration

The backend `/solve/rl` endpoint can load selected `.npz` policies and `.pt`
Torch policy/value checkpoints, then run bounded policy-guided beam search. Set
`RUBIC_RL_MODEL_PATH`, `RUBIC_RL_POLICY_TYPE=auto`, and
`RUBIC_RL_POLICY_DEVICE=cpu` before starting Uvicorn. For the current Torch
baseline, point the model path at
`checkpoints\torch-policy-value-adi-depth-1-10-regularized.pt` and set
`RUBIC_RL_POLICY_TYPE=torch`. Tune `RUBIC_RL_SEARCH_WIDTH` and
`RUBIC_RL_SEARCH_TOP_K` when comparing latency against solve rate. Keep the RL
package installed in the same virtual environment, with the Torch extra for
`.pt` checkpoints. See `docs/stage-2-rl-backend-integration.md` and
`docs/stage-3-rl-inference-search.md` for the API contract and search behavior.

Use `GET /solve/rl/status` to inspect the configured runtime without solving.
Add `?load=true` to force-load the checkpoint and verify Torch or NumPy policy
metadata before a deployment or demo.
