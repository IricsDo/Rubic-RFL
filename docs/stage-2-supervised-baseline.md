# Stage 2 Supervised Baseline

## Scope

The first learning baseline is a dependency-light NumPy softmax policy. It
predicts the next solving action from the flattened one-hot cube sticker state.
This is a warm-start target for later neural and reinforcement-learning work,
not the final solver.

## Training Data

Use the JSONL dataset generator from `rubic_rl.datasets.generator`. Solved
records are ignored during training because they do not have `target_action`.
Each trainable record contributes:

- input: `sticker_indices` converted to a 324-value one-hot feature vector.
- label: `target_action`, the first move in the inverse solution.

## Commands

Generate a smoke dataset:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.datasets.generator --out ..\datasets\smoke.jsonl --depths 1 2 3 --samples-per-depth 100 --seed 20260531 --include-solved
```

Train a checkpoint:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.training.supervised --dataset ..\datasets\smoke.jsonl --model-out ..\checkpoints\linear-policy-smoke.npz --epochs 200 --learning-rate 0.2 --batch-size 128 --validation-split 0.2 --seed 20260531
```

The command prints JSON metrics with train and validation loss/accuracy.
Checkpoints are written under `checkpoints/`, which is ignored by source
control.

## Verification

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

The tests verify feature conversion, supervised learning on canonical one-move
states, and model save/load behavior.
