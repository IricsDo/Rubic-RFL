# Stage 2 Policy Comparison

## Scope

The policy comparison runner evaluates multiple trained checkpoints against the
same seeded scramble set, then ranks them by solve rate and step efficiency. Use
it when deciding whether the MLP baseline actually beats the linear baseline
before moving toward deeper RL algorithms or backend integration.

The runner reuses `evaluate_policy_on_env`, so each model receives identical
depths, sample counts, step limits, and random seeds.

## Command

Run comparison from the RL package directory:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.compare_policies --model linear=..\checkpoints\linear-policy-smoke.npz --model mlp=..\checkpoints\mlp-policy-smoke.npz --depths 1 2 3 --samples-per-depth 100 --max-steps 30 --seed 20260531 --out ..\reports\policy-comparison.json
```

Each `--model` value uses `label=path`. Checkpoint type detection is automatic
for linear and MLP `.npz` files.

## Output

The JSON report contains:

- `config`: shared evaluation depths, sample count, step limit, and seed.
- `policies`: each labeled model with overall and per-depth solve metrics.
- `ranking`: models ordered by solve rate, then fewer solved/all steps, then
  higher average reward.

Add `--include-episodes` when you need per-rollout scrambles, moves, rewards,
and termination status for debugging.

## Verification

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_policy_comparison.py
```

The tests cover labeled model parsing, duplicate-label rejection, ranking, and
optional episode details.
