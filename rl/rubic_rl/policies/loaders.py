from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal

import numpy as np

from rubic_rl.policies.linear_policy import LinearPolicy
from rubic_rl.policies.mlp_policy import MLPPolicy

if TYPE_CHECKING:
    from rubic_rl.policies.torch_policy import TorchPolicyValuePolicy

PolicyType = Literal["auto", "linear", "mlp", "torch"]
if TYPE_CHECKING:
    LoadedPolicy = LinearPolicy | MLPPolicy | TorchPolicyValuePolicy
else:
    LoadedPolicy = Any


def load_policy(
    input_path: str | Path | BinaryIO,
    *,
    policy_type: PolicyType = "auto",
    torch_map_location: str = "cpu",
) -> LoadedPolicy:
    if policy_type == "linear":
        return LinearPolicy.load(input_path)
    if policy_type == "mlp":
        return MLPPolicy.load(input_path)
    if policy_type == "torch":
        return _load_torch_policy(input_path, map_location=torch_map_location)
    if policy_type != "auto":
        raise ValueError(f"Unsupported policy type: {policy_type}")

    detected = _detect_policy_type(input_path)
    if detected == "mlp":
        return MLPPolicy.load(input_path)
    if detected == "torch":
        return _load_torch_policy(input_path, map_location=torch_map_location)
    return LinearPolicy.load(input_path)


def _detect_policy_type(
    input_path: str | Path | BinaryIO,
) -> Literal["linear", "mlp", "torch"]:
    suffix = _path_suffix(input_path)
    if suffix in {".pt", ".pth"}:
        return "torch"

    _rewind_if_possible(input_path)
    try:
        with np.load(input_path, allow_pickle=False) as data:
            files = set(data.files)
            if "policy_type" in files:
                policy_type = str(np.asarray(data["policy_type"]).item())
                if policy_type in {"linear", "mlp"}:
                    return policy_type  # type: ignore[return-value]
            if (
                {"input_weights", "hidden_bias", "output_weights", "output_bias"}
                <= files
            ):
                return "mlp"
            return "linear"
    finally:
        _rewind_if_possible(input_path)


def _load_torch_policy(
    input_path: str | Path | BinaryIO,
    *,
    map_location: str,
) -> "TorchPolicyValuePolicy":
    from rubic_rl.policies.torch_policy import TorchPolicyValuePolicy

    return TorchPolicyValuePolicy.load(input_path, map_location=map_location)


def _path_suffix(input_path: Any) -> str:
    if isinstance(input_path, (str, Path)):
        return Path(input_path).suffix.lower()
    name = getattr(input_path, "name", None)
    if isinstance(name, str):
        return Path(name).suffix.lower()
    return ""


def _rewind_if_possible(value: object) -> None:
    seek = getattr(value, "seek", None)
    if callable(seek):
        seek(0)
