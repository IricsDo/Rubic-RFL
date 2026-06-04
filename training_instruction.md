# RL Training Instructions

## How to Create the Data

Training data is synthetic. The code starts from a solved cube, applies scrambles, then labels each state with the inverse move sequence needed to solve it.

Use the random generator for scalable datasets:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
python -m rubic_rl.datasets.generator --out ..\datasets\random-depth-1-30.jsonl --depths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 --samples-per-depth 2000 --seed 20260603 --include-solved
```

Output: `datasets/random-depth-1-30.jsonl`. Each row contains `stickers`, `sticker_indices`, `scramble`, `depth`, `target_moves`, `target_action`, and metadata.

Use the canonical generator for small deterministic shallow-depth anchors:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
python -m rubic_rl.datasets.canonical --out ..\datasets\canonical-depth-1-2.jsonl --depths 1 2 --include-solved
```

Output: `datasets/canonical-depth-1-2.jsonl`.

## How to Train the Data

The main RL training command is ADI-style policy/value training. It generates and deduplicates records internally, trains a PyTorch residual network, then writes the exact dataset, checkpoint, and report.

Smoke test:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
python -m rubic_rl.training.adi --depths 1 2 3 --samples-per-depth 100 --iterations 1 --epochs-per-iteration 5 --device cuda --dataset-out ..\datasets\adi-smoke.jsonl --checkpoint-out ..\checkpoints\torch-policy-value-adi-smoke.pt --report-out ..\reports\adi-training-smoke.json
```

Depth 1-30 candidate:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
python -m rubic_rl.training.adi --depths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 --samples-per-depth 3000 --canonical-depths 1 2 --iterations 6 --epochs-per-iteration 5 --hidden-dim 512 --residual-blocks 4 --dropout 0.1 --batch-size 1024 --learning-rate 0.0003 --weight-decay 0.001 --value-loss-weight 0.25 --seed 20260603 --device cuda --dataset-out ..\datasets\adi-depth-1-30-curriculum.jsonl --checkpoint-out ..\checkpoints\torch-policy-value-adi-depth-1-30.pt --report-out ..\reports\adi-training-depth-1-30.json
```

Outputs:

- `datasets/adi-depth-1-30-curriculum.jsonl`: deduplicated training records.
- `checkpoints/torch-policy-value-adi-depth-1-30.pt`: PyTorch policy/value checkpoint.
- `reports/adi-training-depth-1-30.json`: training config, metrics, final losses, and checkpoint metadata.

Older NumPy baselines can still be trained with `rubic_rl.training.supervised`, `rubic_rl.training.mlp_supervised`, or `rubic_rl.training.baseline_experiment`, but the UI should use the Torch ADI checkpoint when possible.

For the DAVI value-model track, continue from the existing `.resume` file instead of restarting. Use this when ADI/beam search is not strong enough for deeper scrambles:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
python -m rubic_rl.training.davi --device cuda --iterations 60000 --batch-size 1000 --hidden-dim 512 --residual-blocks 6 --dropout 0.0 --learning-rate 0.001 --weight-decay 0.00001 --loss-type smooth_l1 --huber-delta 1.0 --grad-clip-norm 5.0 --target-update-interval 200 --checkpoint-interval 200 --curriculum-start 1 --curriculum-interval 150 --max-scramble-depth 30 --hard-depth-fraction 0.5 --seed 20260603 --model-version torch-value-davi-v0.2 --checkpoint-out ..\checkpoints\torch-value-davi.pt --report-out ..\reports\davi-training.json
```

Outputs:

- `checkpoints/torch-value-davi.pt`: value checkpoint for DAVI evaluation and solver use.
- `checkpoints/torch-value-davi.pt.resume`: full training state for continuing after interruption.
- `reports/davi-training.json`: loss, sampled depth range, target/prediction means, MAE, and gradient diagnostics.

`--hard-depth-fraction 0.5` means half of each batch is sampled exactly at the current curriculum depth, which focuses learning on the deepest states after the curriculum reaches 30. Lower it if shallow-depth performance regresses; increase it if deeper depths stay weak.

## How to Evaluation Data

Evaluate a model with policy-guided beam search before using it in the UI:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
python -m rubic_rl.evaluation.search_eval --model ..\checkpoints\torch-policy-value-adi-depth-1-30.pt --policy-type auto --depths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 --samples-per-depth 50 --max-depth 30 --beam-width 10 --top-k 8 --seed 20260603 --out ..\reports\adi-search-evaluation-depth-1-30.json
```

Output: `reports/adi-search-evaluation-depth-1-30.json`.

Read these fields first:

- `summary.overall.solve_rate`: total solve percentage.
- `summary.by_depth`: solve rate per scramble depth.
- `avg_duration_ms`: average latency.
- `avg_expanded_states`: search cost.

Use `rubic_rl.evaluation.policy_eval` for direct rollout checks without beam search. Use `rubic_rl.evaluation.compare_policies` when comparing multiple checkpoints with the same seeded evaluation setup.

Evaluate the DAVI checkpoint with a fixed search budget so model comparisons are fair:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
python -m rubic_rl.evaluation.davi_eval --model ..\checkpoints\torch-value-davi.pt --depths 12,13,14,15,16 --samples-per-depth 10 --weight 0.6 --batch-expansion 1000 --max-nodes 1000000 --device cuda --out ..\reports\davi-eval-12-16.json
```

For the official promotion gate, use the same `weight`, `batch-expansion`, `max-nodes`, and `seed`, then raise `--samples-per-depth` to `50`. Changing search parameters changes the solver budget, so do not compare two checkpoints unless these values match.

## How to Config This Model to UI

The frontend calls the backend, and the backend loads the RL model from environment variables. Start the backend with the selected checkpoint:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
$env:RUBIC_RL_MODEL_PATH="checkpoints\torch-policy-value-adi-depth-1-30.pt"
$env:RUBIC_RL_POLICY_TYPE="torch"
$env:RUBIC_RL_POLICY_DEVICE="cpu"
$env:RUBIC_RL_MAX_STEPS="30"
$env:RUBIC_RL_SEARCH_WIDTH="10"
$env:RUBIC_RL_SEARCH_TOP_K="8"
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Verify the backend loaded the model:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/solve/rl/status?load=true"
```

For Docker or production compose, set these values in `deploy/production.env`:

```text
RUBIC_RL_MODEL_FILE=torch-policy-value-adi-depth-1-30.pt
RUBIC_RL_POLICY_TYPE=torch
RUBIC_RL_POLICY_DEVICE=cpu
RUBIC_RL_MAX_STEPS=30
RUBIC_RL_SEARCH_WIDTH=10
RUBIC_RL_SEARCH_TOP_K=8
```

The model file must exist under `checkpoints/`. Promote a model to the UI only when its evaluation report shows acceptable `summary.by_depth` solve rate for the scramble depths exposed in the UI.

## Python File Reference and Outputs

| File | What it does | Output file |
| --- | --- | --- |
| `rl/rubic_rl/datasets/generator.py` | Generates random scramble training records from solved cubes. | JSONL dataset from `--out`, usually under `datasets/`. |
| `rl/rubic_rl/datasets/canonical.py` | Enumerates deterministic shallow move sequences, prunes same-axis repeats, and deduplicates cube states. | JSONL dataset from `--out`. |
| `rl/rubic_rl/datasets/__init__.py` | Exposes dataset helpers for import by training scripts. | No direct output. |
| `rl/rubic_rl/training/adi.py` | Main PyTorch ADI-style policy/value trainer. Creates data, trains, saves model, writes report. | `--dataset-out` JSONL, `--checkpoint-out` `.pt`, `--report-out` JSON. |
| `rl/rubic_rl/training/supervised.py` | Trains older NumPy linear policy from an existing JSONL dataset. | Optional `.npz` from `--model-out`; metrics printed to console. |
| `rl/rubic_rl/training/mlp_supervised.py` | Trains older NumPy MLP policy from an existing JSONL dataset. | Optional `.npz` from `--model-out`; metrics printed to console. |
| `rl/rubic_rl/training/baseline_experiment.py` | Builds a baseline dataset, trains linear and MLP baselines, evaluates both, and writes a report. | JSONL dataset, two `.npz` checkpoints, JSON report. |
| `rl/rubic_rl/training/__init__.py` | Exposes training package symbols. | No direct output. |
| `rl/rubic_rl/evaluation/search_eval.py` | Evaluates a checkpoint with greedy rollout and policy-guided beam search. This is the UI promotion gate. | Optional JSON report from `--out`; otherwise prints JSON. |
| `rl/rubic_rl/evaluation/policy_eval.py` | Evaluates one checkpoint with direct Gymnasium-style rollouts. | JSON printed to console. |
| `rl/rubic_rl/evaluation/compare_policies.py` | Compares multiple checkpoints on the same seeded rollout cases. | Optional JSON report from `--out`; otherwise prints JSON. |
| `rl/rubic_rl/evaluation/__init__.py` | Exposes evaluation helpers. | No direct output. |
| `rl/rubic_rl/envs/rubiks_cube.py` | Defines the Rubik environment, action mapping, reset, step, and observations. | No direct output. |
| `rl/rubic_rl/envs/__init__.py` | Exposes environment package symbols. | No direct output. |
| `rl/rubic_rl/models/policy_value.py` | Defines the Torch policy/value network and checkpoint save/load helpers. | Writes `.pt` checkpoints when called by training code. |
| `rl/rubic_rl/models/__init__.py` | Exposes model package symbols. | No direct output. |
| `rl/rubic_rl/policies/loaders.py` | Loads policy checkpoints and auto-detects policy type. | No direct output. |
| `rl/rubic_rl/policies/torch_policy.py` | Wraps Torch checkpoints for inference. | No direct output. |
| `rl/rubic_rl/policies/linear_policy.py` | Implements the older NumPy linear policy and feature encoding. | Writes `.npz` only when used by training scripts. |
| `rl/rubic_rl/policies/mlp_policy.py` | Implements the older NumPy MLP policy. | Writes `.npz` only when used by training scripts. |
| `rl/rubic_rl/policies/__init__.py` | Exposes policy package symbols. | No direct output. |
| `rl/rubic_rl/encoding.py` | Converts cube stickers into numeric indices/features. | No direct output. |
| `rl/rubic_rl/__init__.py` | Package metadata and imports. | No direct output. |
