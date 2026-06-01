# RL Research Track

This directory contains the DeepCubeA-inspired research track.

## Current Contents

- `rubic_rl/envs/rubiks_cube.py` implements `RubiksCubeEnv`.
- `rubic_rl/encoding.py` converts cube stickers to one-hot or index observations.
- `rubic_rl/datasets/generator.py` creates supervised JSONL training data.
- `rubic_rl/datasets/canonical.py` creates exhaustive short-depth records.
- `rubic_rl/policies/linear_policy.py` trains a NumPy softmax baseline policy.
- `rubic_rl/policies/mlp_policy.py` trains a NumPy MLP baseline policy.
- `rubic_rl/evaluation/policy_eval.py` evaluates policy checkpoints in the environment.
- `rubic_rl/evaluation/compare_policies.py` compares and ranks checkpoints.
- `rubic_rl/evaluation/search_eval.py` benchmarks greedy rollout against beam search.
- `rubic_rl/training/baseline_experiment.py` runs dataset generation, training,
  comparison, and report writing in one command.
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

Run the RL tests:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

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

## Policy Evaluation

`evaluate_policy_on_env` runs a policy against seeded random scrambles and
`evaluate_policy_on_scrambles` runs explicit scrambles for controlled tests. The
CLI prints JSON solve-rate summaries grouped by scramble depth. Add
`--include-episodes` when you need per-rollout predicted moves and rewards.
Use `--policy-type auto` to detect linear and MLP checkpoints automatically.

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

The backend `/solve/rl` endpoint can load a selected `.npz` checkpoint and run a
bounded policy-guided beam search. Set `RUBIC_RL_MODEL_PATH` before starting
Uvicorn, for example `checkpoints\mlp-baseline-depth-1-2-3.npz`, or use the
top-ranked path from `reports\baseline-experiment.json`. Tune
`RUBIC_RL_SEARCH_WIDTH` and `RUBIC_RL_SEARCH_TOP_K` when comparing latency
against solve rate. Keep the RL package installed in the same virtual
environment. See `docs/stage-2-rl-backend-integration.md` and
`docs/stage-3-rl-inference-search.md` for the API contract and search behavior.
