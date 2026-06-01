-- Phase 9 persistence baseline for Rubic RFL.
-- Runtime still supports the JSON replay store; this migration defines the
-- production PostgreSQL target schema for sessions, solves, model outputs,
-- replay packages, and historical metrics.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id text UNIQUE,
    display_name text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id text NOT NULL UNIQUE,
    user_id uuid REFERENCES users(id) ON DELETE SET NULL,
    source text NOT NULL DEFAULT 'api',
    replay_schema_version text NOT NULL,
    replay_package jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cube_states (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('initial', 'final', 'intermediate')),
    stickers char(54) NOT NULL,
    faces jsonb,
    history jsonb NOT NULL DEFAULT '[]'::jsonb,
    validation jsonb,
    is_solved boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    version text NOT NULL,
    checkpoint text,
    policy_type text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (version, checkpoint)
);

CREATE TABLE IF NOT EXISTS solves (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    solver text NOT NULL,
    status text NOT NULL,
    move_count integer NOT NULL DEFAULT 0 CHECK (move_count >= 0),
    duration_ms integer NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
    moves jsonb NOT NULL DEFAULT '[]'::jsonb,
    message text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS moves (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    solve_id uuid NOT NULL REFERENCES solves(id) ON DELETE CASCADE,
    step integer NOT NULL CHECK (step > 0),
    move text NOT NULL,
    before_state_id uuid REFERENCES cube_states(id) ON DELETE SET NULL,
    after_state_id uuid REFERENCES cube_states(id) ON DELETE SET NULL,
    decision jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (solve_id, step)
);

CREATE TABLE IF NOT EXISTS solver_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    solve_id uuid NOT NULL REFERENCES solves(id) ON DELETE CASCADE,
    model_version_id uuid REFERENCES model_versions(id) ON DELETE SET NULL,
    strategy text,
    max_depth integer,
    beam_width integer,
    top_k integer,
    depth_reached integer,
    expanded_states integer,
    visited_states integer,
    search_trace jsonb,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS metrics (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id uuid REFERENCES sessions(id) ON DELETE CASCADE,
    solve_id uuid REFERENCES solves(id) ON DELETE CASCADE,
    model_version_id uuid REFERENCES model_versions(id) ON DELETE SET NULL,
    name text NOT NULL,
    value double precision NOT NULL,
    unit text,
    tags jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_created_at
    ON sessions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id_created_at
    ON sessions (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_cube_states_session_role
    ON cube_states (session_id, role);

CREATE INDEX IF NOT EXISTS idx_solves_session_id
    ON solves (session_id);

CREATE INDEX IF NOT EXISTS idx_solves_solver_status_created_at
    ON solves (solver, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_moves_solve_step
    ON moves (solve_id, step);

CREATE INDEX IF NOT EXISTS idx_solver_runs_model_version_id
    ON solver_runs (model_version_id);

CREATE INDEX IF NOT EXISTS idx_solver_runs_strategy
    ON solver_runs (strategy);

CREATE INDEX IF NOT EXISTS idx_metrics_name_created_at
    ON metrics (name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_metrics_session_id
    ON metrics (session_id);
