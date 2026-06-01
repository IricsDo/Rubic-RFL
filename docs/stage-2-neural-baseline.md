# Stage 2 Neural Baseline

## Scope

The neural baseline adds a dependency-light NumPy multilayer perceptron for
supervised warm-start training. It uses the same JSONL records and 324-value
one-hot cube features as `LinearPolicy`, then predicts one of the fixed 18 cube
actions.

This is not yet reinforcement learning. It is a stronger supervised policy to
compare against the linear baseline before adding value search or RL updates.

## Command

Train from the RL package directory:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.training.mlp_supervised --dataset ..\datasets\smoke.jsonl --model-out ..\checkpoints\mlp-policy-smoke.npz --hidden-units 64 --epochs 300 --learning-rate 0.05 --batch-size 128 --validation-split 0.2 --seed 20260531
```

Evaluate with the shared policy evaluator:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.policy_eval --model ..\checkpoints\mlp-policy-smoke.npz --policy-type auto --depths 1 2 3 --samples-per-depth 100 --max-steps 30 --seed 20260531
```

## Output

The trainer prints JSON with training loss, accuracy, validation metrics,
hidden-unit count, and optional checkpoint path. The evaluator prints solve-rate
metrics grouped by scramble depth.

## Verification

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

Tests cover MLP training, invalid config handling, checkpoint round-tripping,
auto checkpoint loading, and explicit one-move rollout solving.
