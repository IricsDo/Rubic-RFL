# Stage 2 RL Environment

## Scope

The first research milestone adds a Gymnasium-compatible Rubik's Cube
environment. It does not train a neural network yet.

## Environment

`rubic_rl.envs.RubiksCubeEnv` wraps the shared backend cube engine:

- `reset(seed=None, options=None)` returns `(observation, info)`.
- `step(action)` returns `(observation, reward, terminated, truncated, info)`.
- `render()` returns an ANSI cube net unless `render_mode="human"`.
- `close()` is a no-op for Gymnasium compatibility.

## Action Space

The environment uses `Discrete(18)` with stable move IDs:

```text
U, D, L, R, F, B,
U', D', L', R', F', B',
U2, D2, L2, R2, F2, B2
```

Use `MOVE_TO_ACTION` for reproducible conversion from cube moves to actions.

## Observations

The default observation is a one-hot sticker tensor with shape `(54, 6)`.
Set `observation_mode="indices"` to receive a `(54,)` integer vector.

## Rewards

The baseline reward strategy is intentionally simple:

- solved state after a step: `+1.0`
- unsolved step: `-0.01`

DeepCubeA-style training should primarily rely on generated backward scrambles
and search, not on naive sparse-reward exploration.

## Verification

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

Expected result: all RL environment tests pass.
