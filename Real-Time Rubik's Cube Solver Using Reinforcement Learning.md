# Real-Time Rubik's Cube Solver Using Reinforcement Learning

## Revised Project Execution Plan

---

## 1. Project Overview

### 1.1 Goal

Build a modern web application that allows users to interact with a 3D Rubik's Cube, submit cube states, and visualize the solving process in real time.

The long-term goal is to support a Reinforcement Learning based Rubik's Cube solver. However, because training a reliable RL solver for a full 3x3 Rubik's Cube is a difficult research problem, the project should be developed in two tracks:

1. **Product Track**
   Build a fully working Rubik's Cube web application with 3D interaction, validation, replay, and real-time visualization.

2. **Research Track**
   Train and evaluate a DeepCubeA-inspired RL solver and integrate it into the product when it reaches acceptable performance.

---

## 2. Feasibility Assessment

### 2.1 Overall Feasibility

The project is feasible, but it should not be treated as a simple web application. It combines:

* 3D frontend engineering
* Backend API development
* WebSocket-based real-time communication
* Rubik's Cube state validation
* Machine learning model training
* Reinforcement learning research
* Model serving and monitoring

The web application part is highly feasible. The RL solver part is feasible as a research project, but it has higher technical risk and should be developed incrementally.

---

### 2.2 Main Technical Risks

| Area                  | Risk Level | Explanation                                                                | Mitigation                                                     |
| --------------------- | ---------: | -------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 3D Cube UI            |     Medium | Correct cube rotation and animation logic can become complex.              | Start with a simple cube model and add animation gradually.    |
| Cube State Validation |       High | Invalid cube states must be detected correctly.                            | Use cubie representation and strict parity/orientation checks. |
| RL Solver             |  Very High | Rubik's Cube has sparse rewards and a huge state space.                    | Use DeepCubeA/ADI-inspired training instead of simple DQN/PPO. |
| Real-Time Inference   |     Medium | Model inference and search may be too slow for real-time UI.               | Stream intermediate steps and use async backend workers.       |
| Production Deployment |     Medium | ML model serving, database, Redis, and frontend deployment add complexity. | Deploy MVP first, then add production infrastructure.          |

---

## 3. Recommended Development Strategy

### 3.1 Build the Project in Three Stages

Instead of trying to build the full production system immediately, the project should be divided into three clear stages.

---

### Stage 1: MVP Solver Platform

The goal of this stage is to build a working Rubik's Cube application without depending on a fully trained RL model.

The MVP should include:

* Interactive 3D Rubik's Cube
* Manual cube manipulation
* Scramble generation
* Cube state validation
* Classical solver fallback
* Step-by-step replay
* Move timeline
* Basic backend API

Recommended fallback solver:

* Kociemba two-phase solver or another deterministic cube-solving algorithm

This ensures the application is useful even before the RL model becomes reliable.

---

### Stage 2: RL Research and Training

The goal of this stage is to develop and evaluate the RL model independently from the production frontend.

This stage should include:

* Custom Rubik's Cube environment
* Dataset generation from solved-state backward scrambles
* DeepCubeA-inspired neural network
* Autodidactic Iteration training
* Evaluation pipeline
* Model checkpointing
* Experiment tracking
* Baseline comparison against classical solver

---

### Stage 3: Real-Time RL Integration

The goal of this stage is to integrate the trained model into the web application.

This stage should include:

* Model inference API
* WebSocket streaming
* Top-K action visualization
* Confidence logging
* Search tree visualization if MCTS is used
* Replay package generation
* Historical solve analytics

---

## 4. Revised System Architecture

```text
┌──────────────────────────────────┐
│          React Frontend           │
│                                  │
│  - 3D Rubik's Cube                │
│  - Manual Cube Editor             │
│  - Move Timeline                  │
│  - Replay Viewer                  │
│  - RL Decision Panel              │
│  - Analytics Dashboard            │
└───────────────┬──────────────────┘
                │
        REST API / WebSocket
                │
┌───────────────▼──────────────────┐
│          FastAPI Backend          │
│                                  │
│  - Solver API                     │
│  - Cube Validator                 │
│  - WebSocket Streaming            │
│  - Replay Generator               │
│  - Session Management             │
│  - Metrics Service                │
└───────────────┬──────────────────┘
                │
┌───────────────▼──────────────────┐
│          Solver Layer             │
│                                  │
│  - Classical Solver Fallback      │
│  - RL Inference Engine            │
│  - Search Module                  │
│  - Model Version Manager          │
└───────────────┬──────────────────┘
                │
┌───────────────▼──────────────────┐
│          Training Pipeline        │
│                                  │
│  - Cube Environment               │
│  - ADI Training                   │
│  - Evaluation Pipeline            │
│  - Experiment Tracking            │
└──────────────────────────────────┘
```

---

## 5. Technology Stack

## 5.1 Frontend

### Core

* React
* TypeScript
* Vite

### 3D Rendering

* Three.js
* React Three Fiber
* Drei

### UI

* TailwindCSS
* Radix UI
* Framer Motion

### State Management

* Zustand

### Communication

* Axios for REST APIs
* Native WebSocket or Socket.IO-compatible wrapper for real-time updates

### Testing

* Vitest
* React Testing Library
* Playwright for end-to-end testing

---

## 5.2 Backend

### Core API

* Python
* FastAPI
* Pydantic

### Real-Time Communication

* FastAPI WebSocket

### ML and RL

* PyTorch
* Gymnasium
* NumPy

### Task Processing

* Celery or RQ
* Redis as message broker and cache

### Storage

* PostgreSQL for sessions, solves, metrics, and model metadata
* Object storage for model checkpoints and replay packages

### Testing

* Pytest
* FastAPI TestClient
* WebSocket integration tests

---

## 5.3 DevOps

* Docker
* Docker Compose for local development
* GitHub Actions for CI/CD
* Prometheus and Grafana for monitoring
* Sentry or OpenTelemetry for application error tracking

---

# 6. Phase 1: Research and Architecture Decision

## 6.1 Objectives

Define the correct solving strategy before implementation becomes too large.

---

## 6.2 Tasks

### Task 1.1: Literature Review

Study the following topics:

* DeepCubeA
* Autodidactic Iteration
* Approximate Policy Iteration
* MCTS with neural networks
* AlphaZero-like search
* Classical Rubik's Cube solving algorithms
* Kociemba two-phase algorithm

---

### Task 1.2: Algorithm Comparison

| Algorithm             |    Difficulty | Expected Performance | Suitability                  |
| --------------------- | ------------: | -------------------: | ---------------------------- |
| DQN                   |        Medium |                  Low | Not recommended for full 3x3 |
| PPO                   |        Medium |        Low to Medium | Useful for experiments only  |
| A3C                   |        Medium |               Medium | Possible but not ideal       |
| DeepCubeA / ADI       |          High |                 High | Best RL candidate            |
| AlphaZero-like Search |     Very High |            Very High | Future enhancement           |
| Kociemba Solver       | Low to Medium |            Very High | Best MVP fallback            |

---

### Task 1.3: Architecture Decision

Recommended decision:

* Use a deterministic classical solver for MVP.
* Use DeepCubeA-inspired ADI as the main RL research direction.
* Treat RL as a replaceable solver module.
* Keep frontend/backend independent from the internal solver implementation.

---

## 6.3 Deliverables

* Literature review document
* Solver architecture decision record
* Baseline solver selection
* RL research roadmap

---

# 7. Phase 2: Cube Simulator and State Validator

## 7.1 Objectives

Create a mathematically correct Rubik's Cube engine.

---

## 7.2 Recommended Representation

Use **cubie representation** internally.

The cube should track:

* Corner permutation
* Corner orientation
* Edge permutation
* Edge orientation

This representation is better for legality checks than raw 54-sticker representation.

---

## 7.3 Supported Moves

```text
U, D, L, R, F, B
U', D', L', R', F', B'
U2, D2, L2, R2, F2, B2
```

---

## 7.4 Tasks

### Task 2.1: Implement Cube Core

* Define cube state data structure
* Implement solved state
* Implement move application
* Implement inverse moves
* Implement move sequence parser

---

### Task 2.2: Implement Scrambling

Features:

* Random scramble generation
* Configurable scramble depth
* Avoid redundant consecutive moves on the same axis
* Reproducible scrambles using random seed

---

### Task 2.3: Implement Validation

Validation should detect:

* Invalid sticker counts
* Invalid corner orientation
* Invalid edge orientation
* Invalid permutation parity
* Impossible cube states

---

### Task 2.4: Implement Testing

Target test coverage:

* 95%+ for cube core
* 100% for critical move and validation logic

Test cases:

* Applying a move and its inverse returns to the original state
* Applying the same quarter-turn four times returns to the original state
* Scramble followed by inverse scramble returns solved state
* Invalid parity states are rejected

---

## 7.5 Deliverables

* Cube simulator
* Move engine
* Scramble generator
* Cube state validator
* Unit test suite

---

# 8. Phase 3: MVP Solver Backend

## 8.1 Objectives

Build a functional backend before RL integration.

---

## 8.2 Tasks

### Task 3.1: Backend Project Structure

```text
backend/
├── app/
│   ├── api/
│   ├── core/
│   ├── cube/
│   ├── solvers/
│   ├── services/
│   ├── websocket/
│   ├── database/
│   └── tests/
├── pyproject.toml
└── Dockerfile
```

---

### Task 3.2: REST API Endpoints

```text
GET  /health
POST /cube/validate
POST /cube/scramble
POST /cube/apply-move
POST /solve/classical
POST /solve/rl
GET  /sessions/{session_id}
```

---

### Task 3.3: WebSocket Endpoints

```text
/ws/solve/{session_id}
/ws/logs/{session_id}
```

---

### Task 3.4: Classical Solver Integration

Implement a deterministic solver as the first working solver.

The backend should return:

```json
{
  "solver": "classical",
  "status": "solved",
  "moves": ["R", "U", "R'", "U'"],
  "move_count": 4,
  "duration_ms": 12
}
```

---

## 8.3 Deliverables

* FastAPI backend
* REST API
* WebSocket API
* Classical solver fallback
* API tests

---

# 9. Phase 4: Frontend MVP

## 9.1 Objectives

Build a polished and usable 3D Rubik's Cube interface.

---

## 9.2 Frontend Structure

```text
frontend/
├── src/
│   ├── components/
│   ├── features/
│   │   ├── cube/
│   │   ├── solver/
│   │   ├── replay/
│   │   └── analytics/
│   ├── stores/
│   ├── api/
│   ├── types/
│   └── pages/
├── package.json
└── vite.config.ts
```

---

## 9.3 Tasks

### Task 4.1: Build 3D Cube

Features:

* Render 3x3 cube
* Rotate camera
* Select faces
* Apply moves visually

---

### Task 4.2: Build Cube Controls

Controls:

* Scramble
* Solve
* Reset
* Pause
* Replay
* Step forward
* Step backward

---

### Task 4.3: Build Move Timeline

Display:

```text
1. R
2. U
3. R'
4. U'
```

---

### Task 4.4: Build Cube State Editor

Input methods:

* Manual face color editing
* Move sequence input
* Scramble depth selector
* Import/export cube state as JSON

---

### Task 4.5: Build Solver Result Panel

Display:

* Solver type
* Move count
* Solve duration
* Current move
* Progress percentage

---

## 9.4 Deliverables

* Interactive 3D cube
* Cube control panel
* Move timeline
* Replay system
* MVP solver UI

---

# 10. Phase 5: RL Environment

## 10.1 Objectives

Transform the cube simulator into a reinforcement learning environment.

---

## 10.2 State Space

Recommended options:

### Option A: Cubie Encoding

Use:

* Corner permutation
* Corner orientation
* Edge permutation
* Edge orientation

### Option B: One-Hot Sticker Encoding

Use:

* 54 sticker positions
* 6 possible colors per sticker

Recommended initial choice:

* Use cubie encoding for validation.
* Use one-hot or learned embedding for neural network input.

---

## 10.3 Action Space

Use 18 discrete actions:

```text
U, D, L, R, F, B,
U', D', L', R', F', B',
U2, D2, L2, R2, F2, B2
```

---

## 10.4 Reward Design

A naive reward design is not enough for a full 3x3 cube.

Basic reward:

```text
Solved state: +1
Unsolved state: 0
Step penalty: small negative value
```

Recommended DeepCubeA-style approach:

* Generate training states by applying random moves backward from the solved state.
* Train the network to estimate value and policy from these generated states.
* Use search during inference instead of relying only on direct policy output.

---

## 10.5 Gymnasium API

The environment should implement:

```python
reset(seed=None, options=None)
step(action)
render()
close()
```

The `step()` method should return:

```python
observation, reward, terminated, truncated, info
```

---

## 10.6 Deliverables

* Gymnasium-compatible Rubik's Cube environment
* Observation encoder
* Action mapping
* Reward strategy
* Environment tests

---

# 11. Phase 6: RL Training Pipeline

## 11.1 Objectives

Train a DeepCubeA-inspired baseline model.

---

## 11.2 Training Data Generation

Generate states by scrambling backward from the solved cube.

Example depths:

```text
1 move away
2 moves away
...
30 moves away
...
100 moves away
```

Each generated state should store:

* Cube state
* Scramble depth
* Last move
* Inverse solution candidate
* Whether the state is solved

---

## 11.3 Neural Network Design

Recommended architecture:

```text
Input Encoding
↓
Fully Connected Layer
↓
Residual Blocks
↓
Policy Head
↓
Value Head
```

Outputs:

* Policy distribution over 18 moves
* Value estimate for distance-to-solution or expected cost-to-go

---

## 11.4 Training Tasks

### Task 6.1: Build Dataset Generator

* Generate states from solved cube
* Support multiple scramble depths
* Store generated batches efficiently

---

### Task 6.2: Implement Network

* PyTorch model
* Residual blocks
* Policy head
* Value head

---

### Task 6.3: Implement ADI Training Loop

* Generate training states
* Estimate cost-to-go
* Update value and policy networks
* Save checkpoints

---

### Task 6.4: Hyperparameter Tuning

Tune:

* Learning rate
* Batch size
* Number of residual blocks
* Hidden dimension
* Search depth
* Number of generated states
* Optimizer
* Training curriculum

---

### Task 6.5: Evaluation

Evaluate by scramble depth:

| Scramble Depth | Solve Rate | Average Moves | Average Time |
| -------------: | ---------: | ------------: | -----------: |
|              5 |        TBD |           TBD |          TBD |
|             10 |        TBD |           TBD |          TBD |
|             20 |        TBD |           TBD |          TBD |
|             30 |        TBD |           TBD |          TBD |
|             50 |        TBD |           TBD |          TBD |
|            100 |        TBD |           TBD |          TBD |

---

## 11.5 Deliverables

* Dataset generator
* PyTorch model
* ADI training pipeline
* Model checkpoints
* Evaluation report
* Baseline comparison against classical solver

---

# 12. Phase 7: RL Inference and Search

## 12.1 Objectives

Use the trained neural network to solve cube states.

---

## 12.2 Recommended Inference Strategy

Do not rely only on the highest-probability policy move.

Use a search strategy such as:

* Weighted A*
* Beam search
* MCTS
* Policy-guided search

The neural network should guide the search instead of directly replacing search.

---

## 12.3 Inference Output Format

```json
{
  "solver": "rl",
  "model_version": "deepcubea-v0.1",
  "status": "solved",
  "moves": ["R", "U", "R'", "U'"],
  "move_count": 4,
  "duration_ms": 85,
  "steps": [
    {
      "step": 1,
      "selected_move": "R",
      "confidence": 0.91,
      "top_candidates": [
        {"move": "R", "probability": 0.91},
        {"move": "U", "probability": 0.05},
        {"move": "F", "probability": 0.03}
      ]
    }
  ]
}
```

---

## 12.4 Deliverables

* RL inference module
* Search module
* Model loading system
* Model version metadata
* Inference benchmark report

---

# 13. Phase 8: Explainability and Real-Time Visualization

## 13.1 Objectives

Make the solving process transparent to users.

---

## 13.2 Explainability Features

Display:

* Selected move
* Confidence score
* Top-K candidate moves
* Search depth
* Number of expanded states
* Current cube state
* Estimated distance-to-solution
* Model version

---

## 13.3 Real-Time Visualization Features

* Stream selected moves through WebSocket
* Animate every cube move
* Highlight selected face
* Highlight affected cubies
* Show confidence graph
* Show decision history timeline
* Allow pause, resume, and replay

---

## 13.4 Deliverables

* RL decision panel
* Real-time move streaming
* Confidence visualization
* Replay package generator
* Search trace viewer

---

# 14. Phase 9: Data Persistence

## 14.1 Objectives

Store solving sessions, model outputs, and replay data.

---

## 14.2 PostgreSQL Schema

Recommended tables:

```text
users
sessions
cube_states
solves
moves
solver_runs
model_versions
metrics
```

---

## 14.3 Data to Store

For each solve:

* Initial cube state
* Solver type
* Model version
* Move sequence
* Solve status
* Solve duration
* Move count
* Decision logs
* Replay data
* Created timestamp

---

## 14.4 Deliverables

* Database schema
* Migration scripts
* Replay storage
* Historical analytics API

---

# 15. Phase 10: Testing Strategy

## 15.1 Objectives

Guarantee correctness, stability, and reproducibility.

---

## 15.2 Test Categories

### Cube Engine Tests

* Move correctness
* Inverse move correctness
* Scramble reversibility
* Invalid state detection
* Parity checks

### Backend Tests

* REST API tests
* WebSocket tests
* Solver service tests
* Database integration tests

### Frontend Tests

* Component tests
* Cube interaction tests
* Replay UI tests
* End-to-end tests

### RL Tests

* Environment API tests
* Training smoke tests
* Deterministic seed tests
* Model checkpoint loading tests
* Evaluation benchmark tests

### Performance Tests

* Inference latency
* WebSocket streaming latency
* Frontend animation FPS
* Backend load test

---

## 15.3 Deliverables

* Automated test suite
* CI test pipeline
* Test coverage report
* Performance benchmark report

---

# 16. Phase 11: Deployment

## 16.1 Objectives

Deploy the application in a reproducible environment.

---

## 16.2 Local Development

Use Docker Compose:

```text
frontend
backend
postgres
redis
worker
```

---

## 16.3 Production Deployment

Recommended deployment options:

* Frontend: Vercel, Netlify, or static hosting
* Backend: AWS ECS, GCP Cloud Run, Azure Container Apps, or a VPS
* Database: Managed PostgreSQL
* Redis: Managed Redis
* Model storage: S3-compatible object storage

---

## 16.4 CI/CD Pipeline

GitHub Actions should run:

* Linting
* Type checking
* Unit tests
* Backend tests
* Frontend build
* Docker build
* Deployment workflow

---

## 16.5 Monitoring

Track:

* API latency
* WebSocket connection count
* Solver success rate
* Inference latency
* Error rate
* Model version usage
* Average solve duration

---

## 16.6 Deliverables

* Dockerized frontend
* Dockerized backend
* CI/CD pipeline
* Production deployment
* Monitoring dashboard

---

# 17. Milestone Plan

## Milestone 1: Cube Engine

Expected outcome:

* Correct cube representation
* Move engine
* Scramble generator
* Validation logic
* Unit tests

Success criteria:

* 95%+ test coverage for cube engine
* All inverse move tests pass
* Invalid cube states are rejected

---

## Milestone 2: Backend MVP

Expected outcome:

* FastAPI backend
* Validation API
* Scramble API
* Classical solve API
* WebSocket streaming prototype

Success criteria:

* Backend can validate and solve a cube state
* API responses are deterministic
* WebSocket can stream move-by-move updates

---

## Milestone 3: Frontend MVP

Expected outcome:

* Interactive 3D cube
* Scramble and solve buttons
* Move timeline
* Replay animation

Success criteria:

* User can scramble and solve a cube visually
* Cube animation remains smooth
* Move timeline matches cube state

---

## Milestone 4: RL Environment

Expected outcome:

* Gymnasium-compatible environment
* Observation encoder
* Action mapping
* Reward function
* Environment tests

Success criteria:

* Environment passes API tests
* Random actions produce valid cube states
* Scramble and inverse scramble are reproducible

---

## Milestone 5: RL Baseline

Expected outcome:

* DeepCubeA-inspired training pipeline
* Initial trained model
* Evaluation report

Success criteria:

* Model solves shallow scrambles reliably
* Training is reproducible
* Evaluation metrics are stored

---

## Milestone 6: RL Integration

Expected outcome:

* RL inference API
* Model versioning
* Decision logs
* Real-time RL visualization

Success criteria:

* Frontend can display RL-selected moves
* Backend streams Top-K action candidates
* Replay package can be saved and reloaded

---

## Milestone 7: Production Release

Expected outcome:

* Deployed application
* Monitoring
* CI/CD
* Database persistence

Success criteria:

* Application is publicly accessible
* Solve sessions are stored
* Errors and performance metrics are monitored

---

# 18. Revised Success Criteria

## MVP Success Criteria

The MVP is successful if:

* Users can interact with a 3D Rubik's Cube.
* Users can generate valid scrambles.
* Users can solve the cube using a deterministic fallback solver.
* The solution is visualized step by step.
* The backend validates cube states correctly.
* The frontend runs smoothly at around 60 FPS on modern devices.

---

## RL Research Success Criteria

The RL research track is successful if:

* The environment is mathematically correct.
* The model can solve shallow scrambles consistently.
* The model improves as scramble depth increases during curriculum training.
* Evaluation is reproducible.
* The RL solver can be compared against the classical solver.

---

## Full Product Success Criteria

The full product is successful if:

* RL solver achieves a high solve rate on random scrambles.
* Average inference latency is acceptable for interactive use.
* Real-time move streaming works reliably.
* Decision logs are visible to users.
* Solve sessions are reproducible.
* Model versions can be tracked and compared.

---

# 19. Missing Items Added to the Original Plan

The following items were missing or under-specified in the original project plan and should be included:

* MVP scope definition
* Classical solver fallback
* Clear distinction between product development and RL research
* Risk assessment
* Milestone-based delivery
* Cube legality validation details
* Solver interface abstraction
* Model versioning
* Search-based inference strategy
* Reproducible training pipeline
* Evaluation by scramble depth
* Frontend cube state editor
* End-to-end replay package format
* WebSocket error handling
* Authentication plan for future user accounts
* CI/CD pipeline details
* Monitoring and observability
* Performance benchmarks
* Data schema for sessions and solves

---

# 20. Final Recommendation

The project should be implemented incrementally.

The recommended order is:

1. Build the cube engine.
2. Build the backend MVP.
3. Build the frontend MVP.
4. Add a classical solver fallback.
5. Build the RL environment.
6. Train a DeepCubeA-inspired baseline.
7. Add RL inference and search.
8. Add real-time explainability.
9. Add persistence, monitoring, and deployment.

This approach reduces risk because the application becomes useful early, even before the RL model reaches production-level performance.

The most important architectural principle is:

> The web application should not depend on the RL model being successful.
> The RL solver should be a replaceable solver module inside a broader Rubik's Cube solving platform.
