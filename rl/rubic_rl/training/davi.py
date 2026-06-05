"""Deep Approximate Value Iteration (DAVI) trainer for the Rubik's Cube.

Unlike ``rubic_rl.training.adi`` (supervised on noisy scramble-inverse labels),
DAVI learns a value function ``V(s)`` = estimated moves-to-solve by bootstrapping
backward from the solved state:

    y(s) = 0                                    if s is solved
    y(s) = min_a [ 1 + V_target(child_a) ]      otherwise   (V_target(solved)=0)

A periodically-frozen *target network* supplies the bootstrap value for
stability. States are generated on the fly by scrambling the solved cube, with an
optional depth curriculum so the value horizon grows outward. The learned value
is consumed at solve time by ``rubic_rl.search.weighted_astar``.

The trainer checkpoints full training state (model, target, optimizer, RNG, step)
so a run interrupted by a Colab disconnect can resume — there is no other resume
path, so this is mandatory for long runs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from rubic_rl import cube_ops as C
from rubic_rl.models.policy_value import (
    CheckpointMetadata,
    RubiksPolicyValueNet,
    TorchModelConfig,
    save_policy_value_checkpoint,
)

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError as error:  # pragma: no cover - exercised by install path
    raise ModuleNotFoundError(
        "PyTorch is required for DAVI training. "
        'Install the optional dependency with: python -m pip install -e ".[torch]"'
    ) from error


DEFAULT_CHECKPOINT_OUT = Path("../checkpoints/torch-value-davi.pt")
DEFAULT_REPORT_OUT = Path("../reports/davi-training.json")


@dataclass
class DAVIConfig:
    checkpoint_out: Path = DEFAULT_CHECKPOINT_OUT
    report_out: Path | None = DEFAULT_REPORT_OUT
    resume_from: Path | None = None
    iterations: int = 10_000
    batch_size: int = 1_000
    max_scramble_depth: int = 30
    min_scramble_depth: int = 1
    solved_fraction: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss_type: str = "mse"
    huber_delta: float = 1.0
    policy_loss_weight: float = 0.0
    grad_clip_norm: float | None = None
    target_update_interval: int = 200
    checkpoint_interval: int = 200
    curriculum: bool = True
    curriculum_start: int = 1
    curriculum_interval: int = 150
    hard_depth_fraction: float = 0.0
    hidden_dim: int = 512
    residual_blocks: int = 6
    dropout: float = 0.0
    seed: int | None = 20260603
    model_version: str = "torch-value-davi-v0.1"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when set")
        if self.policy_loss_weight < 0:
            raise ValueError("policy_loss_weight cannot be negative")
        if not 0.0 <= self.hard_depth_fraction <= 1.0:
            raise ValueError("hard_depth_fraction must be in [0, 1]")
        if self.loss_type not in {"mse", "smooth_l1"}:
            raise ValueError("loss_type must be 'mse' or 'smooth_l1'")

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checkpoint_out"] = str(self.checkpoint_out)
        data["report_out"] = None if self.report_out is None else str(self.report_out)
        data["resume_from"] = None if self.resume_from is None else str(self.resume_from)
        return data

    @property
    def model_config(self) -> TorchModelConfig:
        return TorchModelConfig(
            hidden_dim=self.hidden_dim,
            residual_blocks=self.residual_blocks,
            dropout=self.dropout,
        )

    def resume_path(self) -> Path:
        if self.resume_from is not None:
            return Path(self.resume_from)
        return Path(str(self.checkpoint_out) + ".resume")


def _value(model: RubiksPolicyValueNet, features: torch.Tensor) -> torch.Tensor:
    _, value = model(features)
    return value


@torch.no_grad()
def _bootstrap_targets(
    target_model: RubiksPolicyValueNet,
    states: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute value targets and the greedy bootstrap action for an (N,54) batch."""
    n = states.shape[0]
    children = C.expand_all_actions(states)  # (N, A, 54)
    flat = children.reshape(n * C.NUM_ACTIONS, C.NUM_STICKERS)
    features = torch.as_tensor(C.one_hot_features(flat), device=device)
    child_value = _value(target_model, features).reshape(n, C.NUM_ACTIONS)
    child_value = torch.clamp(child_value, min=0.0)
    child_solved = torch.as_tensor(
        C.is_solved(flat).reshape(n, C.NUM_ACTIONS), device=device
    )
    # Cost of one move + future cost (0 if the child is already solved).
    cost = 1.0 + torch.where(child_solved, torch.zeros_like(child_value), child_value)
    targets, best_actions = cost.min(dim=1)
    return (
        targets.detach().cpu().numpy(),
        best_actions.detach().cpu().numpy().astype(np.int64),
    )


def _current_curriculum_depth(config: DAVIConfig, step: int) -> int:
    if not config.curriculum:
        return config.max_scramble_depth
    grown = config.curriculum_start + step // max(1, config.curriculum_interval)
    return int(min(config.max_scramble_depth, grown))


def _sample_training_states(
    config: DAVIConfig,
    depth: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the training batch, optionally focusing part of it at current depth."""
    hard_n = int(round(config.batch_size * config.hard_depth_fraction))
    hard_n = max(0, min(config.batch_size, hard_n))
    uniform_n = config.batch_size - hard_n

    batches: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    if uniform_n > 0:
        states, sampled_depths = C.scramble_batch(
            uniform_n, depth, rng, min_depth=config.min_scramble_depth
        )
        batches.append(states)
        depths.append(sampled_depths)
    if hard_n > 0:
        states, sampled_depths = C.scramble_batch(hard_n, depth, rng, min_depth=depth)
        batches.append(states)
        depths.append(sampled_depths)

    if not batches:
        return C.solved_batch(0), np.zeros(0, dtype=np.int64)
    states = np.vstack(batches)
    sampled_depths = np.concatenate(depths).astype(np.int64)
    order = rng.permutation(len(states))
    return states[order], sampled_depths[order]


def _loss(predicted: torch.Tensor, target: torch.Tensor, config: DAVIConfig) -> torch.Tensor:
    if config.loss_type == "mse":
        return functional.mse_loss(predicted, target)
    if config.loss_type in {"smooth_l1", "huber"}:
        return functional.smooth_l1_loss(predicted, target, beta=config.huber_delta)
    raise ValueError("loss_type must be 'mse' or 'smooth_l1'")


def _policy_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float | None:
    mask = targets >= 0
    if not bool(mask.any()):
        return None
    predicted = torch.argmax(logits[mask], dim=1)
    return float((predicted == targets[mask]).float().mean().item())


def run_davi_training(config: DAVIConfig) -> dict[str, Any]:
    device = torch.device(config.device)
    model = RubiksPolicyValueNet(config.model_config).to(device)
    target_model = RubiksPolicyValueNet(config.model_config).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed)

    start_step = 0
    history: list[dict[str, Any]] = []
    resume_path = config.resume_path()
    if resume_path.exists():
        start_step, history = _load_resume(resume_path, model, target_model, optimizer, rng, device)
        print(f"[davi] resumed from {resume_path} at step {start_step}", flush=True)

    n_solved = max(0, int(round(config.batch_size * config.solved_fraction)))
    started = time.perf_counter()

    for step in range(start_step, config.iterations):
        depth = _current_curriculum_depth(config, step)
        states, sampled_depths = _sample_training_states(config, depth, rng)
        targets, best_actions = _bootstrap_targets(target_model, states, device)

        if n_solved > 0:
            states = np.vstack([states, C.solved_batch(n_solved)])
            targets = np.concatenate([targets, np.zeros(n_solved, dtype=np.float32)])
            best_actions = np.concatenate(
                [best_actions, np.full(n_solved, -100, dtype=np.int64)]
            )

        features = torch.as_tensor(C.one_hot_features(states), device=device)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
        policy_target_tensor = torch.as_tensor(best_actions, dtype=torch.long, device=device)

        model.train()
        optimizer.zero_grad()
        policy_logits, predicted = model(features)
        value_loss = _loss(predicted, target_tensor, config)
        policy_loss = functional.cross_entropy(
            policy_logits,
            policy_target_tensor,
            ignore_index=-100,
        )
        loss = value_loss + config.policy_loss_weight * policy_loss
        loss.backward()
        grad_norm = None
        if config.grad_clip_norm is not None and config.grad_clip_norm > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            )
        optimizer.step()

        if (step + 1) % config.target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        if (step + 1) % 50 == 0 or step == start_step:
            errors = (predicted.detach() - target_tensor).abs()
            entry = {
                "step": step + 1,
                "loss": float(loss.item()),
                "value_loss": float(value_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "policy_accuracy": _policy_accuracy(policy_logits.detach(), policy_target_tensor),
                "curriculum_depth": depth,
                "sampled_depth_min": int(sampled_depths.min()) if len(sampled_depths) else 0,
                "sampled_depth_max": int(sampled_depths.max()) if len(sampled_depths) else 0,
                "sampled_depth_mean": float(sampled_depths.mean()) if len(sampled_depths) else 0.0,
                "target_mean": float(targets.mean()),
                "target_std": float(targets.std()),
                "pred_mean": float(predicted.detach().mean().item()),
                "pred_std": float(predicted.detach().std(unbiased=False).item()),
                "mae": float(errors.mean().item()),
                "grad_norm": grad_norm,
                "elapsed_s": round(time.perf_counter() - started, 1),
            }
            history.append(entry)
            policy_accuracy = entry["policy_accuracy"]
            policy_accuracy_text = (
                "n/a" if policy_accuracy is None else f"{policy_accuracy:.3f}"
            )
            print(
                f"[davi] step {step + 1}/{config.iterations} "
                f"loss={entry['loss']:.4f} K={depth} "
                f"depth_mean={entry['sampled_depth_mean']:.1f} "
                f"y_mean={entry['target_mean']:.2f} "
                f"pred_mean={entry['pred_mean']:.2f} "
                f"mae={entry['mae']:.3f} "
                f"pi_acc={policy_accuracy_text} {entry['elapsed_s']}s",
                flush=True,
            )

        if (step + 1) % config.checkpoint_interval == 0 or (step + 1) == config.iterations:
            _save_checkpoint(config, model)
            _save_resume(resume_path, model, target_model, optimizer, rng, step + 1, history)
            _write_report(config, history, step + 1)

    _save_checkpoint(config, model)
    _write_report(config, history, config.iterations)
    return {"steps": config.iterations, "history_tail": history[-5:]}


def _save_checkpoint(config: DAVIConfig, model: RubiksPolicyValueNet) -> None:
    metadata = CheckpointMetadata(
        model_version=config.model_version,
        value_scale=1.0,
        training=config.to_json_dict(),
    )
    save_policy_value_checkpoint(config.checkpoint_out, model=model, metadata=metadata)


def _save_resume(
    path: Path,
    model: RubiksPolicyValueNet,
    target_model: RubiksPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    step: int,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "target": target_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)  # atomic-ish: avoid corrupt file if interrupted mid-write


def _load_resume(
    path: Path,
    model: RubiksPolicyValueNet,
    target_model: RubiksPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    target_model.load_state_dict(payload["target"])
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"].cpu() if hasattr(payload["torch_rng"], "cpu") else payload["torch_rng"])
    return int(payload["step"]), list(payload.get("history", []))


def _write_report(config: DAVIConfig, history: list[dict[str, Any]], step: int) -> None:
    if config.report_out is None:
        return
    report = {
        "experiment": "davi-value-iteration",
        "config": config.to_json_dict(),
        "completed_steps": step,
        "history": history,
    }
    Path(config.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(config.report_out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_seed(value: str | None) -> int | None:
    if value is None or str(value).lower() == "none":
        return None
    return int(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DAVI value-iteration trainer for Rubik's Cube.")
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume-state file. Default: <checkpoint-out>.resume")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--max-scramble-depth", type=int, default=30)
    parser.add_argument("--min-scramble-depth", type=int, default=1)
    parser.add_argument("--solved-fraction", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss-type", choices=("mse", "smooth_l1"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--policy-loss-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum")
    parser.add_argument("--curriculum-start", type=int, default=1)
    parser.add_argument("--curriculum-interval", type=int, default=150)
    parser.add_argument("--hard-depth-fraction", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", default="20260603")
    parser.add_argument("--model-version", default="torch-value-davi-v0.1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    config = DAVIConfig(
        checkpoint_out=args.checkpoint_out,
        report_out=args.report_out,
        resume_from=args.resume_from,
        iterations=args.iterations,
        batch_size=args.batch_size,
        max_scramble_depth=args.max_scramble_depth,
        min_scramble_depth=args.min_scramble_depth,
        solved_fraction=args.solved_fraction,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        policy_loss_weight=args.policy_loss_weight,
        grad_clip_norm=args.grad_clip_norm,
        target_update_interval=args.target_update_interval,
        checkpoint_interval=args.checkpoint_interval,
        curriculum=args.curriculum,
        curriculum_start=args.curriculum_start,
        curriculum_interval=args.curriculum_interval,
        hard_depth_fraction=args.hard_depth_fraction,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
        dropout=args.dropout,
        seed=_parse_seed(args.seed),
        model_version=args.model_version,
        device=args.device,
    )
    summary = run_davi_training(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
