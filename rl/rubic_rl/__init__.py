from __future__ import annotations

from gymnasium.envs.registration import register, registry

from .envs import ACTION_MOVES, MOVE_TO_ACTION, RubiksCubeEnv

ENV_ID = "RubicRFL/RubiksCube-v0"


def register_envs() -> None:
    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="rubic_rl.envs:RubiksCubeEnv",
        )


register_envs()

__all__ = [
    "ACTION_MOVES",
    "ENV_ID",
    "MOVE_TO_ACTION",
    "RubiksCubeEnv",
    "register_envs",
]
