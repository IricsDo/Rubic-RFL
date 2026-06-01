import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.cube import Cube
from app.solvers import RLSolver
from app.storage import ReplaySessionStore
from app.main import app
import app.main as main_module


class ConstantPolicy:
    def __init__(self, action: int):
        self.action = action

    def predict(self, features):
        return [self.action for _ in range(features.shape[0])]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.original_session_store = main_module.session_store
        temp_root = Path(__file__).resolve().parents[2] / ".tmp" / "test-sessions"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.store_dir = temp_root / f"case-{uuid4().hex}"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        main_module.session_store = ReplaySessionStore(self.store_dir)
        main_module.metrics.reset()
        self.client = TestClient(app)

    def tearDown(self):
        main_module.session_store = self.original_session_store
        main_module.metrics.reset()

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_metrics_endpoint_exposes_http_request_metrics(self):
        self.client.get("/health")

        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("# HELP rubic_http_requests_total", body)
        self.assertIn(
            'rubic_http_requests_total{method="GET",path="/health",status="200"} 1',
            body,
        )
        self.assertIn(
            'rubic_http_request_duration_seconds_count{method="GET",path="/health",status="200"} 1',
            body,
        )

    def test_validate_endpoint_accepts_solved_cube(self):
        response = self.client.post(
            "/cube/validate", json={"stickers": Cube.solved().to_string()}
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["validation"]["valid"])
        self.assertTrue(payload["is_solved"])

    def test_scramble_endpoint_returns_valid_history(self):
        response = self.client.post(
            "/cube/scramble", json={"depth": 8, "seed": "api-test"}
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["scramble"]), 8)
        self.assertEqual(payload["history"], payload["scramble"])
        self.assertTrue(payload["validation"]["valid"])
        self.assertFalse(payload["is_solved"])

    def test_apply_move_endpoint_updates_history(self):
        response = self.client.post(
            "/cube/apply-move",
            json={"stickers": Cube.solved().to_string(), "history": [], "move": "R"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["history"], ["R"])
        self.assertFalse(payload["is_solved"])
        self.assertTrue(payload["validation"]["valid"])

    def test_classical_solver_solves_scramble_without_history(self):
        scramble = self.client.post(
            "/cube/scramble", json={"depth": 10, "seed": "solve-test"}
        ).json()

        response = self.client.post(
            "/solve/classical",
            json={
                "stickers": scramble["stickers"],
                "session_id": "classical-session",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "solved")
        self.assertEqual(payload["solver"], "classical-kociemba")
        self.assertEqual(payload["move_count"], len(payload["moves"]))
        self.assertLessEqual(payload["move_count"], 24)

        solved = Cube.from_string(scramble["stickers"]).apply_sequence(
            payload["moves"], record_history=False
        )
        self.assertTrue(solved.is_solved())
        self.assertEqual(payload["session_id"], "classical-session")

        stored = self.client.get("/sessions/classical-session")
        self.assertEqual(stored.status_code, 200)
        self.assertEqual(stored.json()["solver"], "classical-kociemba")

    def test_classical_solver_rejects_invalid_state(self):
        stickers = list(Cube.solved().stickers)
        stickers[0] = "R"

        response = self.client.post(
            "/solve/classical",
            json={"stickers": "".join(stickers)},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "invalid")
        self.assertIn("Sticker U appears 8 times instead of 9", payload["message"])

    def test_rl_solver_reports_solved_for_solved_cube_without_policy(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(model_path="missing-policy.npz")
        try:
            response = self.client.post(
                "/solve/rl", json={"stickers": Cube.solved().to_string()}
            )
        finally:
            main_module.rl_solver = original_solver

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["solver"], "rl-policy")
        self.assertEqual(payload["status"], "solved")
        self.assertEqual(payload["moves"], [])

    def test_rl_solver_solves_one_move_state_with_configured_policy(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(
            policy=ConstantPolicy(9),
            model_path="in-memory-test-policy",
            max_steps=1,
        )
        scrambled = Cube.solved().apply_move("R")
        try:
            response = self.client.post(
                "/solve/rl", json={"stickers": scrambled.to_string()}
            )
        finally:
            main_module.rl_solver = original_solver

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["solver"], "rl-policy")
        self.assertEqual(payload["status"], "solved")
        self.assertEqual(payload["moves"], ["R'"])
        self.assertEqual(payload["move_count"], 1)

        solved = scrambled.apply_sequence(payload["moves"], record_history=False)
        self.assertTrue(solved.is_solved())

    def test_metrics_endpoint_exposes_solver_run_metrics(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(
            policy=ConstantPolicy(9),
            model_path="in-memory-metrics-policy",
            max_steps=1,
        )
        scrambled = Cube.solved().apply_move("R")
        try:
            solve_response = self.client.post(
                "/solve/rl", json={"stickers": scrambled.to_string()}
            )
        finally:
            main_module.rl_solver = original_solver

        self.assertEqual(solve_response.status_code, 200)

        response = self.client.get("/metrics")
        body = response.text
        self.assertIn(
            'rubic_solver_runs_total{model_version="in-memory-metrics-policy",solver="rl-policy",status="solved"} 1',
            body,
        )
        self.assertIn(
            'rubic_solver_success_total{model_version="in-memory-metrics-policy",solver="rl-policy"} 1',
            body,
        )
        self.assertIn(
            'rubic_solver_duration_seconds_count{model_version="in-memory-metrics-policy",solver="rl-policy",status="solved"} 1',
            body,
        )
        self.assertIn(
            'rubic_solver_moves_total{model_version="in-memory-metrics-policy",solver="rl-policy",status="solved"} 1',
            body,
        )

    def test_rl_replay_package_captures_reproducible_step_trace(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(
            policy=ConstantPolicy(9),
            model_path="in-memory-replay-policy",
            max_steps=1,
        )
        scrambled = Cube.solved().apply_move("R")
        try:
            response = self.client.post(
                "/solve/rl/replay-package",
                json={
                    "session_id": "replay-test",
                    "stickers": scrambled.to_string(),
                },
            )
        finally:
            main_module.rl_solver = original_solver

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schema_version"], "rubic-rfl-replay-v1")
        self.assertEqual(payload["session_id"], "replay-test")
        self.assertEqual(payload["solver"], "rl-policy")
        self.assertEqual(payload["status"], "solved")
        self.assertEqual(payload["moves"], ["R'"])
        self.assertEqual(payload["initial_state"]["stickers"], scrambled.to_string())
        self.assertTrue(payload["initial_state"]["validation"]["valid"])
        self.assertTrue(payload["final_state"]["is_solved"])
        self.assertEqual(payload["model"]["version"], "in-memory-replay-policy")
        self.assertEqual(payload["search"]["strategy"], "policy-guided-beam-search")
        self.assertEqual(payload["search"]["trace"]["trace_limit"], 200)
        self.assertFalse(payload["search"]["trace"]["truncated"])
        self.assertEqual(payload["search"]["trace"]["records"][0]["node_id"], 1)
        self.assertEqual(
            payload["search"]["trace"]["records"][0]["candidates"][0]["outcome"],
            "solution",
        )

        step = payload["steps"][0]
        self.assertEqual(step["step"], 1)
        self.assertEqual(step["move"], "R'")
        self.assertEqual(step["before_stickers"], scrambled.to_string())
        self.assertTrue(Cube.from_string(step["after_stickers"]).is_solved())
        self.assertEqual(step["decision"]["selected_move"], "R'")
        self.assertEqual(step["decision"]["confidence"], 1.0)
        self.assertEqual(step["decision"]["top_candidates"][0]["move"], "R'")
        self.assertEqual(payload["solver_result"]["moves"], ["R'"])

    def test_replay_package_is_persisted_and_listed_by_session_id(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(
            policy=ConstantPolicy(9),
            model_path="in-memory-persist-policy",
            max_steps=1,
        )
        scrambled = Cube.solved().apply_move("R")
        try:
            response = self.client.post(
                "/solve/rl/replay-package",
                json={
                    "session_id": "persisted-rl-session",
                    "stickers": scrambled.to_string(),
                },
            )
        finally:
            main_module.rl_solver = original_solver

        self.assertEqual(response.status_code, 200)

        stored = self.client.get("/sessions/persisted-rl-session")
        self.assertEqual(stored.status_code, 200)
        stored_payload = stored.json()
        self.assertEqual(stored_payload["session_id"], "persisted-rl-session")
        self.assertEqual(stored_payload["solver"], "rl-policy")
        self.assertEqual(stored_payload["moves"], ["R'"])

        listing = self.client.get("/sessions?limit=10")
        self.assertEqual(listing.status_code, 200)
        sessions = listing.json()["sessions"]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "persisted-rl-session")
        self.assertEqual(sessions[0]["model_version"], "in-memory-persist-policy")

    def test_sessions_can_save_imported_replay_package_and_delete_it(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(
            policy=ConstantPolicy(9),
            model_path="in-memory-import-policy",
            max_steps=1,
        )
        scrambled = Cube.solved().apply_move("R")
        try:
            replay = self.client.post(
                "/solve/rl/replay-package",
                json={
                    "session_id": "importable-session",
                    "stickers": scrambled.to_string(),
                },
            ).json()
        finally:
            main_module.rl_solver = original_solver

        replay["session_id"] = "manual-import"
        save_response = self.client.post("/sessions", json={"replay_package": replay})
        self.assertEqual(save_response.status_code, 200)
        self.assertTrue(save_response.json()["saved"])
        self.assertEqual(save_response.json()["session"]["session_id"], "manual-import")

        delete_response = self.client.delete("/sessions/manual-import")
        self.assertEqual(delete_response.status_code, 200)
        self.assertTrue(delete_response.json()["deleted"])

        missing = self.client.get("/sessions/manual-import")
        self.assertEqual(missing.status_code, 404)

    def test_solve_analytics_summarizes_saved_sessions(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(
            policy=ConstantPolicy(9),
            model_path="in-memory-analytics-policy",
            max_steps=1,
        )
        scrambled = Cube.solved().apply_move("R")
        try:
            solved_response = self.client.post(
                "/solve/rl/replay-package",
                json={
                    "session_id": "analytics-solved",
                    "stickers": scrambled.to_string(),
                },
            )
            self.assertEqual(solved_response.status_code, 200)

            main_module.rl_solver = RLSolver(
                model_path="missing-analytics-policy.npz",
                max_steps=1,
            )
            unavailable_response = self.client.post(
                "/solve/rl",
                json={
                    "session_id": "analytics-unavailable",
                    "stickers": scrambled.to_string(),
                },
            )
            self.assertEqual(unavailable_response.status_code, 200)
        finally:
            main_module.rl_solver = original_solver

        response = self.client.get("/analytics/solves?limit=10")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["totals"]["session_count"], 2)
        self.assertEqual(payload["totals"]["solved_count"], 1)
        self.assertEqual(payload["totals"]["solve_rate"], 0.5)
        self.assertEqual(payload["performance"]["average_move_count"], 0.5)
        self.assertEqual(payload["storage"]["backend"], "json-files")
        self.assertEqual(payload["filters"]["limit"], 10)

        statuses = {item["status"]: item for item in payload["by_status"]}
        self.assertEqual(statuses["solved"]["session_count"], 1)
        self.assertEqual(statuses["unavailable"]["session_count"], 1)

        solvers = {item["solver"]: item for item in payload["by_solver"]}
        self.assertEqual(solvers["rl-policy"]["session_count"], 2)
        self.assertEqual(solvers["rl-policy"]["solved_count"], 1)
        self.assertEqual(payload["recent_sessions"][0]["session_id"], "analytics-unavailable")

        filtered = self.client.get("/analytics/solves?status=solved")
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(filtered.json()["totals"]["session_count"], 1)
        self.assertEqual(filtered.json()["totals"]["solve_rate"], 1.0)

    def test_rl_solver_reports_unavailable_when_checkpoint_is_missing(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(model_path="missing-policy.npz", max_steps=1)
        scrambled = Cube.solved().apply_move("R")
        try:
            response = self.client.post(
                "/solve/rl", json={"stickers": scrambled.to_string()}
            )
        finally:
            main_module.rl_solver = original_solver

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["moves"], [])

    def test_rl_solver_websocket_streams_decision_trace(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(
            policy=ConstantPolicy(9),
            model_path="in-memory-ws-policy",
            max_steps=1,
        )
        scrambled = Cube.solved().apply_move("R")
        try:
            with self.client.websocket_connect("/ws/solve/rl-session") as websocket:
                websocket.send_json(
                    {
                        "solver": "rl",
                        "stickers": scrambled.to_string(),
                    }
                )

                started = websocket.receive_json()
                decision = websocket.receive_json()
                move = websocket.receive_json()
                completed = websocket.receive_json()
        finally:
            main_module.rl_solver = original_solver

        self.assertEqual(started["event"], "started")
        self.assertEqual(started["session_id"], "rl-session")
        self.assertEqual(started["solver"], "rl-policy")

        self.assertEqual(decision["event"], "decision")
        self.assertEqual(decision["step"], 1)
        self.assertEqual(decision["selected_move"], "R'")
        self.assertEqual(decision["confidence"], 1.0)
        self.assertEqual(decision["model_version"], "in-memory-ws-policy")
        self.assertEqual(decision["top_candidates"][0]["move"], "R'")

        self.assertEqual(move["event"], "move")
        self.assertEqual(move["step"], 1)
        self.assertEqual(move["move"], "R'")
        self.assertTrue(Cube.from_string(move["stickers"]).is_solved())

        self.assertEqual(completed["event"], "completed")
        self.assertEqual(completed["result"]["status"], "solved")
        self.assertEqual(completed["result"]["moves"], ["R'"])
        self.assertEqual(
            completed["result"]["details"]["model_version"],
            "in-memory-ws-policy",
        )
        self.assertEqual(
            completed["replay_package"]["schema_version"],
            "rubic-rfl-replay-v1",
        )
        self.assertEqual(completed["replay_package"]["session_id"], "rl-session")
        self.assertEqual(completed["replay_package"]["steps"][0]["move"], "R'")
        self.assertEqual(
            completed["replay_package"]["search"]["trace"]["records"][0]["candidates"][0][
                "outcome"
            ],
            "solution",
        )

        metrics_body = self.client.get("/metrics").text
        self.assertIn(
            'rubic_websocket_connections_total{endpoint="/ws/solve/{session_id}"} 1',
            metrics_body,
        )
        self.assertIn(
            'rubic_websocket_active_connections{endpoint="/ws/solve/{session_id}"} 0',
            metrics_body,
        )
        self.assertIn(
            'rubic_solver_runs_total{model_version="in-memory-ws-policy",solver="rl-policy",status="solved"} 1',
            metrics_body,
        )


if __name__ == "__main__":
    unittest.main()
