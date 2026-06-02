# Stage 3 RL Inference Search

## Scope

The RL backend now uses policy-guided beam search instead of a pure greedy
rollout. The trained policy still ranks actions, but the solver explores several
high-probability branches before declaring failure. This matches the project
plan's Phase 7 recommendation that neural policies should guide search rather
than directly replace search.

## Search Strategy

`backend/app/solvers/rl_search.py` implements:

- `SearchConfig(max_depth, beam_width, top_k, trace_limit)` for bounded search control.
- `PolicyGuidedSearch` for expanding top policy actions at each cube state.
- Repeated-state filtering to avoid cycling through already-seen stickers.
- Immediate inverse-move filtering to avoid one-step undo loops.
- Per-step decision logs with selected move, confidence, and top candidates.
- Optional bounded branch traces with kept, pruned, skipped, and solution outcomes.

When a policy exposes `predict_proba`, search uses ranked action probabilities.
For older test policies that only expose `predict`, the solver falls back to a
single-action probability distribution.

## Backend Configuration

```powershell
$env:RUBIC_RL_MODEL_PATH="checkpoints\mlp-baseline-depth-1-2-3.npz"
$env:RUBIC_RL_POLICY_TYPE="auto"
$env:RUBIC_RL_POLICY_DEVICE="cpu"
$env:RUBIC_RL_MAX_STEPS="30"
$env:RUBIC_RL_SEARCH_WIDTH="5"
$env:RUBIC_RL_SEARCH_TOP_K="5"
$env:RUBIC_RL_SEARCH_TRACE_LIMIT="200"
cd backend
..\.venv\Scripts\uvicorn.exe app.main:app --reload
```

Use a wider beam or larger top-K when solve quality matters more than latency.
Keep both small for interactive demos unless the search benchmark shows the
extra solve rate is worth the added latency.

For Torch policy/value checkpoints, set `RUBIC_RL_MODEL_PATH` to the `.pt`
checkpoint and `RUBIC_RL_POLICY_TYPE=torch`. Keep `RUBIC_RL_POLICY_DEVICE=cpu`
unless the backend runtime has a working CUDA PyTorch install.

## Runtime Status

`GET /solve/rl/status` reports the configured RL runtime without solving a cube:
model path, checkpoint name, checkpoint existence, configured and effective
policy type, device, loaded state, last load error, and search settings.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/solve/rl/status
Invoke-RestMethod "http://127.0.0.1:8000/solve/rl/status?load=true"
```

The default check is lightweight and does not load the policy. Use `?load=true`
for deployment verification because it forces the configured checkpoint to load
and returns Torch metadata such as the checkpoint model version when available.

## Search Benchmark

`rl/rubic_rl/evaluation/search_eval.py` compares greedy policy rollout against
the backend's policy-guided beam search on identical seeded scrambles.

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.search_eval --model ..\checkpoints\mlp-baseline-depth-1-2-3.npz --policy-type auto --depths 1 2 3 --samples-per-depth 100 --max-depth 30 --beam-width 5 --top-k 5 --seed 20260531 --out ..\reports\search-evaluation.json
```

The JSON report stores the benchmark config, generated cases, per-strategy
summaries by depth, solve rate, average move count, latency, expanded states,
visited states, depth reached, and an overall ranking. Add `--include-episodes`
to inspect individual successes and failures.

## Phase 8 Decision Trace

The static frontend renders the `/solve/rl` `details` payload as an RL decision
panel. The panel shows model version/checkpoint metadata, search strategy,
policy type/device, depth, expanded and visited states, beam width, Top-K size,
current cube stickers, and each selected move with Top-K confidence bars.

During replay, the active decision step is highlighted in the trace and the
affected cube layer is outlined in the 3D viewport. This is the first
explainability slice from Phase 8.

The backend now also returns `details.search_trace` for RL searches. Each record
captures the expanded node id, parent id, depth, path, sticker fingerprint,
score, beam rank, and candidate outcomes:

- `kept_in_beam` for generated branches retained after beam pruning.
- `pruned_by_beam` for generated branches dropped by the beam width.
- `skipped_inverse` and `skipped_visited` for filtered actions.
- `solution` for the branch that solved the cube.

The frontend renders these records under `Search Trace` in the RL panel, with
the active replay path highlighted as the user steps through the solve.

## Phase 8 WebSocket Streaming

`/ws/solve/{session_id}` now accepts the same cube payload as the REST solve
endpoints plus `solver: "rl"` or `solver_mode: "rl"`. The stream emits:

- `started` with session, solver, status, and initial stickers.
- `decision` with selected move, confidence, Top-K candidates, search metrics,
  model and policy metadata, and the cube stickers before the move.
- `move` with the applied move and resulting stickers.
- `completed` with the normal solver result payload and replay package.

The frontend RL mode opens this WebSocket first. Streamed moves are applied live
so the cube animates toward the solution, and the final replay index is set to
the solved state. If the WebSocket cannot connect, the UI restores the original
cube and falls back to the REST `/solve/rl` response.

## Phase 8 Replay Packages

`POST /solve/rl/replay-package` returns a reproducible solve artifact with:

- schema version, session id, creation timestamp, solver status, duration, and moves.
- initial and final cube snapshots with validation details.
- model version/checkpoint, policy type/device, search metrics, and bounded
  branch trace records.
- per-step before/after stickers plus decision confidence and Top-K candidates.
- the original solver result payload for API compatibility.

The WebSocket `completed` event includes the same package, so streamed solves do
not need a second solve request. The frontend stores the package after an RL
solve, enables `Export Replay`, and can import `rubic-rfl-replay-v1` JSON back to
the initial cube state for replay, including the branch trace viewer.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests
cd rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

The focused search test proves beam search can solve a two-move scramble even
when the correct moves are ranked below the top policy action. Additional
coverage verifies branch trace outcomes and replay package trace serialization.
