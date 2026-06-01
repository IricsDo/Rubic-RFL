# ADR 0001: Solver Architecture

## Status

Accepted and updated for the Stage 1 classical solver.

## Decision

The product will treat every solver as a replaceable module behind a shared result
shape. The Stage 1 classical implementation uses Kociemba two-phase search for
arbitrary valid 3x3 cube states. An RL-guided solver can be added later without
changing the frontend contract.

## Rationale

The plan identifies the RL solver as high-risk research work. The UI, API,
validation, replay, and session workflows should become useful before a trained
model exists. Kociemba gives deterministic, fast, and well-understood baseline
solutions that can later be compared against RL inference.

## Consequences

- Arbitrary imported cube states can be solved by the classical endpoint when
  they pass cube legality validation.
- The backend already exposes `/solve/classical` and `/solve/rl`; the RL endpoint
  currently returns an explicit unavailable status.
- Future solver modules should return `solver`, `status`, `moves`, `move_count`,
  `duration_ms`, and a human-readable `message`.
