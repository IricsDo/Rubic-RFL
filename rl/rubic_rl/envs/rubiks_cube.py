from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from app.cube import Cube, generate_scramble, inverse_sequence, parse_moves, validate_stickers
from rubic_rl.encoding import encode_cube

ACTION_MOVES: tuple[str, ...] = (
    "U",
    "D",
    "L",
    "R",
    "F",
    "B",
    "U'",
    "D'",
    "L'",
    "R'",
    "F'",
    "B'",
    "U2",
    "D2",
    "L2",
    "R2",
    "F2",
    "B2",
)
MOVE_TO_ACTION = {move: index for index, move in enumerate(ACTION_MOVES)}


@dataclass(frozen=True)
class RewardConfig:
    solved_reward: float = 1.0
    step_penalty: float = -0.01


class RubiksCubeEnv(gym.Env):
    """Gymnasium-compatible 3x3 Rubik's Cube environment.

    The environment delegates all cube transitions to the production cube engine,
    which keeps research experiments aligned with backend validation behavior.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        *,
        scramble_depth: int = 0,
        max_steps: int = 100,
        observation_mode: str = "one_hot",
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if scramble_depth < 0:
            raise ValueError("scramble_depth cannot be negative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if observation_mode not in {"one_hot", "indices"}:
            raise ValueError("observation_mode must be 'one_hot' or 'indices'")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode!r}")

        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(ACTION_MOVES))
        if observation_mode == "one_hot":
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(54, 6),
                dtype=np.int8,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=5,
                shape=(54,),
                dtype=np.int64,
            )

        self._cube = Cube.solved()
        self._scramble: tuple[str, ...] = ()
        self._steps = 0

    @property
    def cube(self) -> Cube:
        return self._cube

    @property
    def scramble(self) -> tuple[str, ...]:
        return self._scramble

    @property
    def steps(self) -> int:
        return self._steps

    @staticmethod
    def action_to_move(action: int) -> str:
        try:
            action_index = int(action)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid action: {action!r}") from error
        if action_index < 0 or action_index >= len(ACTION_MOVES):
            raise ValueError(f"Invalid action: {action!r}")
        return ACTION_MOVES[action_index]

    @staticmethod
    def move_to_action(move: str) -> int:
        normalized = parse_moves([move])[0]
        try:
            return MOVE_TO_ACTION[normalized]
        except KeyError as error:
            raise ValueError(f"Move is not in the action space: {move!r}") from error

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self._steps = 0

        if "cube" in options:
            cube = options["cube"]
            if not isinstance(cube, Cube):
                raise ValueError("options['cube'] must be an app.cube.Cube instance")
            self._cube = cube.without_history()
            self._scramble = ()
        elif "stickers" in options:
            stickers = options["stickers"]
            if isinstance(stickers, str):
                self._cube = Cube.from_string(stickers)
            else:
                self._cube = Cube(stickers=tuple(stickers))
            self._scramble = ()
        elif "scramble" in options:
            self._scramble = parse_moves(options["scramble"])
            self._cube = Cube.solved().apply_sequence(self._scramble)
        else:
            depth = int(options.get("scramble_depth", self.scramble_depth))
            if depth < 0:
                raise ValueError("scramble_depth cannot be negative")
            scramble_seed = seed
            if scramble_seed is None:
                scramble_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
            self._scramble = generate_scramble(depth=depth, seed=scramble_seed)
            self._cube = Cube.solved().apply_sequence(self._scramble)

        validation = validate_stickers(self._cube.stickers)
        if not validation.valid:
            raise ValueError(f"Invalid cube state: {'; '.join(validation.errors)}")

        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        move = self.action_to_move(action)
        self._cube = self._cube.apply_move(move)
        self._steps += 1

        terminated = self._cube.is_solved()
        truncated = self._steps >= self.max_steps and not terminated
        reward = (
            self.reward_config.solved_reward
            if terminated
            else self.reward_config.step_penalty
        )
        info = self._info()
        info["move"] = move
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        output = self._render_ansi()
        if self.render_mode == "human":
            print(output)
            return None
        return output

    def close(self) -> None:
        return None

    def inverse_scramble_actions(self) -> tuple[int, ...]:
        return tuple(self.move_to_action(move) for move in inverse_sequence(self._scramble))

    def _observation(self) -> np.ndarray:
        return encode_cube(self._cube, self.observation_mode)

    def _info(self) -> dict[str, Any]:
        return {
            "is_solved": self._cube.is_solved(),
            "scramble": list(self._scramble),
            "history": list(self._cube.history),
            "steps": self._steps,
        }

    def _render_ansi(self) -> str:
        faces = self._cube.to_faces()

        def rows(face: str) -> list[str]:
            values = faces[face]
            return [" ".join(values[index : index + 3]) for index in range(0, 9, 3)]

        up = rows("U")
        left = rows("L")
        front = rows("F")
        right = rows("R")
        back = rows("B")
        down = rows("D")

        lines = [f"      {row}" for row in up]
        lines.extend(
            "  ".join(section[row] for section in (left, front, right, back))
            for row in range(3)
        )
        lines.extend(f"      {row}" for row in down)
        return "\n".join(lines)
