# Handoff

## Current State

The DAVI work has been implemented on `feature/davi` and is being merged back to `main`. The branch contains the DAVI value-iteration track, policy-aware DAVI v0.3 training, resumable DAVI evaluation reports, frontend animation polish, and deployment docs cleanup.

The latest depth-15 DAVI evaluation from Colab used:

```bash
python -m rubic_rl.evaluation.davi_eval \
  --model /content/drive/MyDrive/rubic-rfl/checkpoints/torch-value-davi.pt \
  --depths 15 \
  --samples-per-depth 50 \
  --weight 0.6 \
  --policy-weight 0.25 \
  --batch-expansion 1000 \
  --max-nodes 1000000 \
  --device cuda \
  --out /content/drive/MyDrive/rubic-rfl/reports/davi-eval-depth-15-policy-v03-s50.json \
  --resume \
  --flush-every 1
```

Result: depth 15 solved `31/50`, `solve_rate=0.62`, `avg_len_solved=11.77`, `avg_expanded=490682`, `avg_ms=103599`. This is better than earlier runs but not close to the target of depth 1-30 at 90%.

## What Was Built

- Added DAVI value training in `rl/rubic_rl/training/davi.py`.
- Added weighted A* search in `rl/rubic_rl/search/weighted_astar.py`.
- Added DAVI evaluation in `rl/rubic_rl/evaluation/davi_eval.py`.
- Added policy-guided search support with `--policy-weight`.
- Added resumable evaluation with `--resume` and `--flush-every`.
- Updated `training_instruction.md` with data, training, evaluation, and UI model selection commands.
- Updated frontend cube animation/replay behavior.
- Switched Docker docs and CI references toward `deploy/compose.production.yml`.

## DAVI Assessment

Training at step 100000 shows the model is learning:

- `policy_accuracy` is around `0.41-0.47`, much better than random `~0.056`.
- `value_loss` is around `0.07-0.09`.
- `mae` is around `0.28-0.33`.

However, depth-15 evaluation still fails many cases by hitting `max_nodes`. The current bottleneck is search efficiency and CPU/RAM frontier growth, not GPU memory. Continuing to train DAVI v0.3 may help slightly, but it is unlikely to reach the target alone.

## Recommended Next Steps

1. Run policy-weight ablations on the same checkpoint:
   - `--policy-weight 0.0`
   - `--policy-weight 0.1`
   - `--policy-weight 0.25`
   - `--policy-weight 0.5`
2. Add `--policy-top-k` to Weighted A* so the search can prune low-probability actions.
3. Record failed evaluation cases and use them for targeted replay/evaluation.
4. Only consider a larger model after search improvements are measured.

## Useful Commands

Run DAVI tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider rl\tests\test_davi_training.py
```

Run all RL tests:

```powershell
cd rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

Serve the frontend locally:

```powershell
python -m http.server 5173 -d frontend
```

Start the production-style stack:

```powershell
.\.venv\Scripts\python.exe tools\prepare_production_env.py --generate-secrets --force
docker compose --env-file deploy\production.env -f deploy\compose.production.yml up -d
```

## Notes

Generated local files such as `.venv/`, `node_modules/`, `datasets/`, `checkpoints/`, `reports/`, `.env`, and `deploy/production.env` are intentionally ignored and should stay out of Git.
