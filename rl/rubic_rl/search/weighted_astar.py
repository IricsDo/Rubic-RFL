"""Batch Weighted A* (BWAS) search using a learned value heuristic.

Solves a cube by A* over states with cost ``f = g + weight * V(s)``, where ``g``
is the number of moves taken and ``V`` is the DAVI value estimate (moves-to-go).
Children are expanded in batches so the value network is called on many states at
once — essential for GPU throughput. ``weight < 1`` makes the search closer to
optimal (slower); ``weight`` toward 1 is greedier/faster but longer solutions.
"""

from __future__ import annotations

import heapq
from typing import Callable

import numpy as np

from rubic_rl import cube_ops as C

ValueFn = Callable[[np.ndarray], np.ndarray]
ActionPolicyFn = Callable[[np.ndarray], np.ndarray]


def make_value_fn(model, device, *, clamp_min: float = 0.0, batch: int = 8192) -> ValueFn:
    """Wrap a RubiksPolicyValueNet into ``states (M,54) -> values (M,)``."""
    import torch

    def value_fn(states: np.ndarray) -> np.ndarray:
        features = C.one_hot_features(states)
        chunks = []
        for start in range(0, len(features), batch):
            tensor = torch.as_tensor(features[start : start + batch], device=device)
            with torch.no_grad():
                _, value = model(tensor)
            chunks.append(value.detach().cpu().numpy())
        values = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return np.clip(values, clamp_min, None)

    return value_fn


def make_action_policy_fn(model, device, *, batch: int = 8192) -> ActionPolicyFn:
    """Wrap a RubiksPolicyValueNet into ``states (M,54) -> action probabilities``."""
    import torch

    def action_policy_fn(states: np.ndarray) -> np.ndarray:
        features = C.one_hot_features(states)
        chunks = []
        for start in range(0, len(features), batch):
            tensor = torch.as_tensor(features[start : start + batch], device=device)
            with torch.no_grad():
                logits, _ = model(tensor)
                probabilities = torch.softmax(logits, dim=1)
            chunks.append(probabilities.detach().cpu().numpy())
        if not chunks:
            return np.zeros((0, C.NUM_ACTIONS), dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    return action_policy_fn


def weighted_astar_solve(
    state: np.ndarray,
    value_fn: ValueFn,
    *,
    action_policy_fn: ActionPolicyFn | None = None,
    weight: float = 0.6,
    policy_weight: float = 0.0,
    batch_expansion: int = 100,
    max_nodes: int = 1_000_000,
    avoid_inverse: bool = True,
) -> dict:
    """Solve a single ``(54,)`` cube state. Returns a result dict.

    Keys: ``solved`` (bool), ``moves`` (tuple[str]), ``length`` (int),
    ``expanded`` (nodes popped), ``generated`` (children created).
    """
    state = np.asarray(state, dtype=np.int64).reshape(-1)
    if bool(C.is_solved(state[None])[0]):
        return {"solved": True, "moves": (), "length": 0, "expanded": 0, "generated": 0}

    start_h = float(value_fn(state[None])[0])
    counter = 0
    # heap entries: (f, tiebreak, g, state_bytes, path)
    open_heap: list[tuple] = [(weight * start_h, counter, 0, state.tobytes(), ())]
    best_g: dict[bytes, int] = {state.tobytes(): 0}
    expanded = 0
    generated = 0
    depth_reached = 0

    while open_heap and expanded < max_nodes:
        popped = []
        while open_heap and len(popped) < batch_expansion:
            popped.append(heapq.heappop(open_heap))

        parent_states = np.stack(
            [np.frombuffer(sbytes, dtype=np.int64) for _f, _t, _g, sbytes, _path in popped]
        )
        parent_probs = (
            action_policy_fn(parent_states)
            if action_policy_fn is not None and policy_weight > 0.0
            else None
        )

        child_states: list[np.ndarray] = []
        child_meta: list[tuple[int, tuple, bytes, float]] = []
        for parent_index, (_f, _t, g, sbytes, path) in enumerate(popped):
            expanded += 1
            parent = parent_states[parent_index]
            last_action = C.MOVES.index(path[-1]) if path else -1
            kids = parent[C.MOVE_PERMUTATIONS]  # (A, 54)
            if parent_probs is None:
                action_order = range(C.NUM_ACTIONS)
                probabilities = None
            else:
                probabilities = parent_probs[parent_index]
                action_order = np.argsort(-probabilities)
            for action in action_order:
                action = int(action)
                if (
                    avoid_inverse
                    and last_action >= 0
                    and action == C.INVERSE_ACTION[last_action]
                ):
                    continue
                child = kids[action]
                child_bytes = child.tobytes()
                g_child = g + 1
                depth_reached = max(depth_reached, g_child)
                prev = best_g.get(child_bytes)
                if prev is not None and prev <= g_child:
                    continue
                policy_cost = (
                    0.0
                    if probabilities is None
                    else float(-np.log(max(float(probabilities[action]), 1e-8)))
                )
                child_states.append(child)
                child_meta.append((g_child, path + (C.MOVES[action],), child_bytes, policy_cost))

        if not child_states:
            continue

        child_arr = np.stack(child_states)
        generated += len(child_arr)

        solved_mask = C.is_solved(child_arr)
        if solved_mask.any():
            index = int(np.argmax(solved_mask))
            g_child, path, _cb, _policy_cost = child_meta[index]
            return {
                "solved": True,
                "moves": path,
                "length": len(path),
                "expanded": expanded,
                "generated": generated,
                "visited": len(best_g),
                "depth_reached": depth_reached,
            }

        values = value_fn(child_arr)
        for (g_child, path, child_bytes, policy_cost), value in zip(child_meta, values):
            prev = best_g.get(child_bytes)
            if prev is not None and prev <= g_child:
                continue
            best_g[child_bytes] = g_child
            counter += 1
            heapq.heappush(
                open_heap,
                (
                    g_child + weight * float(value) + policy_weight * policy_cost,
                    counter,
                    g_child,
                    child_bytes,
                    path,
                ),
            )

    return {
        "solved": False,
        "moves": (),
        "length": 0,
        "expanded": expanded,
        "generated": generated,
        "visited": len(best_g),
        "depth_reached": depth_reached,
    }
