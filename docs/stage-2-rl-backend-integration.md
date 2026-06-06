# Stage 2 RL Backend Integration

## Scope

The FastAPI backend exposes `/solve/rl` as a policy-guided RL endpoint. It
validates the cube, loads a trained `.npz` policy or `.pt` Torch policy/value
checkpoint lazily, uses the policy to guide a bounded beam search, and returns
the same top-level solver response shape as the classical solver.

The endpoint is intended for measured checkpoint demos, not full-depth
production solving. If search cannot solve within the configured depth limit,
the response status is `failed` and the best explored move prefix is returned.

## Configuration

By default, the backend looks for:

```text
checkpoints/lightweight-linear-axis-samples-28-heldout/linear-policy.npz
```

Override the checkpoint, search depth, beam width, and candidate count before
starting Uvicorn:

```powershell
$env:RUBIC_RL_MODEL_PATH="checkpoints\lightweight-linear-axis-samples-28-heldout\linear-policy.npz"
$env:RUBIC_RL_POLICY_TYPE="auto"
$env:RUBIC_RL_POLICY_DEVICE="cpu"
$env:RUBIC_RL_MAX_STEPS="30"
$env:RUBIC_RL_SEARCH_WIDTH="5"
$env:RUBIC_RL_SEARCH_TOP_K="5"
cd backend
..\.venv\Scripts\uvicorn.exe app.main:app --reload
```

Install the RL package in the same virtual environment when loading real
checkpoints. Use the Torch extra when serving `.pt` checkpoints:

```powershell
cd rl
python -m pip install -e ".[dev]"
python -m pip install -e ".[torch,dev]"
```

## API Response

`POST /solve/rl` accepts the standard cube payload and returns:

- `status`: `solved`, `failed`, `invalid`, or `unavailable`.
- `moves`: predicted move sequence.
- `move_count`: number of predicted moves.
- `message`: solver outcome or checkpoint loading issue.
- `details`: present for RL search responses and includes strategy, max depth,
  beam width, top-K size, expanded state count, visited state count, depth
  reached, policy type, policy device, and per-step top candidates.

The frontend Replay panel includes a Classical/RL mode selector. RL mode calls
`/solve/rl`; Classical mode keeps the Kociemba endpoint and local inverse-history
fallback.

`GET /solve/rl/status` reports the configured RL runtime before solving. It
includes checkpoint existence, current loaded state, model version, policy
type/device, search settings, and the last policy load error. Add `?load=true`
to force-load the configured checkpoint during deployment or smoke checks.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests
```

The tests cover runtime status reporting, solved-state handling, in-memory
policy rollout, missing checkpoint behavior, policy-guided beam search, invalid
cubes, and existing classical endpoints.
