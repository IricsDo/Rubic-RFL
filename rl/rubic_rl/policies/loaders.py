from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np

from rubic_rl.policies.linear_policy import LinearPolicy
from rubic_rl.policies.mlp_policy import MLPPolicy


PolicyType = Literal["auto", "linear", "mlp"]
LoadedPolicy = LinearPolicy | MLPPolicy


def load_policy(
    input_path: str | Path | BinaryIO,
    *,
    policy_type: PolicyType = "auto",
) -> LoadedPolicy:
    if policy_type == "linear":
        return LinearPolicy.load(input_path)
    if policy_type == "mlp":
        return MLPPolicy.load(input_path)
    if policy_type != "auto":
        raise ValueError(f"Unsupported policy type: {policy_type}")

    detected = _detect_policy_type(input_path)
    if detected == "mlp":
        return MLPPolicy.load(input_path)
    return LinearPolicy.load(input_path)


def _detect_policy_type(input_path: str | Path | BinaryIO) -> Literal["linear", "mlp"]:
    _rewind_if_possible(input_path)
    try:
        with np.load(input_path, allow_pickle=False) as data:
            files = set(data.files)
            if "policy_type" in files:
                policy_type = str(np.asarray(data["policy_type"]).item())
                if policy_type in {"linear", "mlp"}:
                    return policy_type  # type: ignore[return-value]
            if {"input_weights", "hidden_bias", "output_weights", "output_bias"} <= files:
                return "mlp"
            return "linear"
    finally:
        _rewind_if_possible(input_path)


def _rewind_if_possible(value: object) -> None:
    seek = getattr(value, "seek", None)
    if callable(seek):
        seek(0)
