# Phase 12 RL Roadmap

This roadmap is based on the current `rl/` implementation in this repository,
not on a generic RL backlog.

## Current State

The RL track already has more than placeholders:

- `rl/rubic_rl/envs/rubiks_cube.py` provides a tested Gymnasium environment.
- `rl/rubic_rl/training/baseline_experiment.py` provides reproducible
  supervised baselines for linear and MLP policies.
- `rl/rubic_rl/training/adi.py` provides a PyTorch policy/value training path.
- `rl/rubic_rl/training/davi.py` provides a value-learning path for
  weighted-A* search.
- `rl/rubic_rl/evaluation/policy_eval.py` and
  `rl/rubic_rl/evaluation/search_eval.py` provide seeded evaluation flows.
- `backend/app/solvers/rl_policy.py` already consumes selected RL checkpoints.

The track is therefore past "set up RL". The next work should improve
experiment discipline, benchmark quality, and training iteration speed.

## Gaps

The main gaps in the current RL workflow are:

1. No experiment config layer.
   Training and evaluation are mostly driven by long CLI commands in docs.

2. No standard benchmark suite across all RL approaches.
   The repo has evaluation tools, but not a single source of truth for
   benchmark cases, depth bands, and ranking criteria.

3. Curriculum exists inside trainers, but not as a tracked experiment plan.
   `adi.py` and `davi.py` support curriculum-like progression, but there is no
   standard preset system, no promotion criteria, and no stable naming for runs.

4. Weak artifact bookkeeping.
   Checkpoints and reports are produced, but there is no manifest that says
   which checkpoint is best for which benchmark and backend runtime profile.

5. Product-facing RL validation is still shallow.
   The backend can serve RL checkpoints, but there is no repeatable gate such as
   "only promote a checkpoint if depth 1-5 solve rate and latency thresholds are
   met".

## Recommended Order

Work in this order:

1. Standardize experiments.
2. Standardize benchmarks.
3. Improve DAVI and ADI with tracked curricula.
4. Add a promotion path from research checkpoint to backend runtime.

That order matters. This repo already has enough training code. It does not yet
have enough structure to compare runs cleanly.

## Phase A: Standardize Experiments

Goal: make every serious RL run reproducible from a small config file.

### Deliverables

- Add `rl/experiments/` with JSON or YAML configs for:
  - `baseline-smoke`
  - `adi-depth-1-5`
  - `adi-depth-1-10`
  - `davi-smoke`
  - `davi-depth-10`
- Add a small runner such as `rubic_rl.training.run_experiment`.
- Standardize output layout:
  - `datasets/<run-name>/...`
  - `checkpoints/<run-name>/...`
  - `reports/<run-name>/...`
- Write a manifest file per run with:
  - git commit
  - seed
  - trainer name
  - config
  - output paths
  - headline metrics

### Why this first

Right now the knowledge of how to run the best ADI and DAVI jobs lives mostly
in docs and ad hoc command lines. That is fragile and slows iteration.

### Suggested implementation targets

- New package area: `rl/rubic_rl/experiments/`
- New CLI entry: `python -m rubic_rl.training.run_experiment --config ...`
- Tests:
  - config parsing
  - output directory creation
  - manifest generation

## Phase B: Standardize Benchmarks

Goal: compare all RL approaches on the same seeded cases and the same ranking
rules.

### Deliverables

- Add benchmark presets such as:
  - `smoke`: depths 1-3
  - `shallow`: depths 1-5
  - `mid`: depths 1-10
  - `search-heavy`: depths 5-10
- Persist benchmark cases to JSON so every model sees the same scrambles.
- Add a unified benchmark runner that can evaluate:
  - greedy policy rollout
  - policy-guided beam search
  - weighted A* with DAVI value estimates
- Produce one report schema with:
  - solve rate by depth
  - average solution length
  - average expanded states
  - average latency
  - recommended backend operating point

### Why this second

`policy_eval.py`, `search_eval.py`, and DAVI search already exist, but they are
still separate tools. The missing piece is one benchmark contract.

### Suggested implementation targets

- New area: `rl/rubic_rl/benchmarks/`
- New CLI entry: `python -m rubic_rl.evaluation.benchmark_suite --suite shallow`
- Tests:
  - benchmark case persistence
  - report ranking
  - deterministic replay with fixed seeds

## Phase C: Improve DAVI and ADI

Goal: improve training quality after measurement is stable.

### DAVI priorities

1. Add fixed validation case sets by depth.
   Current training reports track optimization statistics well, but model
   selection should also use search outcomes on held-out cases.

2. Add curriculum presets.
   Promote from depth bands only when previous bands meet thresholds.

3. Run ablations on:
   - `loss_type`
   - `hard_depth_fraction`
   - `solved_fraction`
   - `policy_loss_weight`
   - `target_update_interval`

4. Store best-by-metric checkpoints, not only latest checkpoints.

### ADI priorities

1. Track overfit explicitly.
   The docs already show that larger depth 1-10 runs overfit.

2. Add training presets for:
   - regularized shallow curriculum
   - cumulative depth 1-10
   - lower-capacity mobile/backend-serving model

3. Compare ADI checkpoints with the same benchmark suites used for DAVI.

### Why this third

Without Phases A and B, these changes produce more runs, not more clarity.

## Phase D: Promote RL Into the Backend Path

Goal: make RL checkpoint promotion deliberate instead of manual.

### Deliverables

- Add a promotion report that declares:
  - best checkpoint
  - benchmark suite used
  - backend env values
  - expected latency and solve rate
- Add a verification command that checks whether a promoted checkpoint still
  loads and meets a minimum solve-rate smoke target.
- Record model metadata in a simple manifest that the backend can read or that
  deployment can reference directly.

### Suggested implementation targets

- Extend `backend/app/solvers/rl_policy.py` integration only after the benchmark
  contract is stable.
- Reuse `/solve/rl/status` as the runtime verification endpoint.

## Concrete Next Sprint

If only one sprint is available, do this:

1. Add `rl/experiments/` configs.
2. Add `run_experiment` to launch baseline, ADI, and DAVI from config.
3. Add one benchmark suite named `shallow` for depths 1-5.
4. Add one manifest file per run.
5. Add tests for config parsing and deterministic benchmark case generation.

That sprint gives the highest leverage. It will make every later RL change
cheaper to run, compare, and promote.

## Promotion Criteria

A practical promotion rule for this repo:

- Depth 1-3 solve rate must be near-saturated.
- Depth 1-5 solve rate must beat the current backend default checkpoint.
- Average latency must stay within the local backend budget.
- Search width increases must justify their cost in solve-rate gain.

Do not promote a checkpoint based only on training loss or policy accuracy.

## Immediate Recommendation

The next implementation task should be:

`Add experiment configs and a unified RL experiment runner.`

That is the smallest change that improves the entire RL track instead of only a
single model family.
