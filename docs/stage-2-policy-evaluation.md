# Stage 2 Policy Evaluation

## Scope

The policy evaluator measures whether a trained action policy can solve
scrambled cubes through real Gymnasium rollouts. It loads a checkpoint, predicts
one move at a time from the current cube state, applies that action in
`RubiksCubeEnv`, and reports solve rates by scramble depth.

This evaluator is intentionally separate from training. Use it to compare
checkpoints and to define the baseline a future neural or RL policy must beat.

## Command

Run evaluation from the RL package directory:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.policy_eval --model ..\checkpoints\linear-policy-smoke.npz --policy-type auto --depths 1 2 3 --samples-per-depth 100 --max-steps 30 --seed 20260531
```

Add `--include-episodes` to include per-rollout scrambles, predicted moves,
rewards, and termination status in the JSON output. Use `--policy-type auto` to
detect linear and MLP checkpoints from saved `.npz` files.

## Output

The command prints JSON with:

- `config`: depths, sample count, step limit, and seed.
- `summary.overall`: total attempts, solved count, solve rate, average steps,
  and average reward.
- `summary.by_depth`: the same metrics grouped by scramble depth.

## Verification

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

The tests cover explicit scramble rollouts, deterministic seeded evaluation,
invalid config handling, and solve-rate aggregation.
