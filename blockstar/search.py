"""Automatic order discovery with geometric obstacle relocation.

No piece labels, reference solutions, or preselected assembly order are used.
Search can run in either direction because every slide is reversible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .engine import Engine
from .model import Move, PuzzleSpec, replay


@dataclass(frozen=True, slots=True)
class Policy:
    path_weight: float = 1.0
    blocker_weight: float = 2.0
    parking_distance: float = 1.0
    goal_overlap: float = 1.0
    temperature: float = 0.5
    order_noise: float = 3.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class Planner:
    def __init__(self, spec: PuzzleSpec):
        self.spec = spec
        self.engine = Engine(spec)

    def rollout(
        self, seed: int, *, reverse: bool = True, policy: Policy | None = None
    ) -> tuple[Move, ...] | None:
        policy = policy or Policy()
        rng = random.Random(seed)
        spec, engine = self.spec, self.engine
        state, goal = (spec.goal, spec.initial) if reverse else (spec.initial, spec.goal)
        fixed: set[int] = set()
        moves: list[Move] = []
        width = spec.width

        def axes(a: int, b: int) -> int:
            return int(a // width != b // width) + int(a % width != b % width)

        while state != goal:
            fixed.update(i for i in range(len(state)) if state[i] == goal[i])
            fixed_mask = 0
            for i in fixed:
                fixed_mask |= engine.masks[i][state[i]]
            choices = []
            for i in range(len(state)):
                if i in fixed:
                    continue
                relaxed = engine.find_path(i, state[i], goal[i], fixed_mask)
                if relaxed is None:
                    return None
                direct = engine.path(state, i, goal[i])
                path = direct if direct is not None else relaxed
                corridor = engine.corridor(i, state[i], path)
                blockers = [
                    j for j in range(len(state)) if j != i and engine.masks[j][state[j]] & corridor
                ]
                rank = (
                    policy.path_weight * len(path)
                    + policy.blocker_weight * len(blockers)
                    + rng.random() * policy.order_noise
                )
                choices.append((rank, i, path, corridor, blockers))
            choices.sort()
            for _, i, _path, corridor, blockers in choices:
                trial, added = list(state), []
                rng.shuffle(blockers)
                failed = False
                for j in blockers:
                    occupied = engine.occupied(tuple(trial)) ^ engine.masks[j][trial[j]]
                    reachable = engine.reachable(j, trial[j], occupied)
                    parking = []
                    for pos, route in reachable.items():
                        mask = engine.masks[j][pos]
                        if mask & corridor or not route:
                            continue
                        overlap = sum(
                            bool(mask & engine.masks[k][goal[k]])
                            for k in range(len(state))
                            if k != j and k not in fixed
                        )
                        rank = (
                            len(route)
                            + policy.parking_distance * axes(pos, goal[j])
                            + policy.goal_overlap * overlap
                            + rng.random() * policy.temperature
                        )
                        parking.append((rank, pos, route))
                    if not parking:
                        failed = True
                        break
                    _, pos, route = min(parking)
                    added.extend(engine.moves(j, trial[j], route))
                    trial[j] = pos
                if failed:
                    continue
                # Recompute after parking; this may expose a shorter route.
                route = engine.path(tuple(trial), i, goal[i])
                if route is None:
                    continue
                added.extend(engine.moves(i, trial[i], route))
                trial[i] = goal[i]
                frozen = fixed_mask | engine.masks[i][goal[i]]
                if any(
                    engine.find_path(j, trial[j], goal[j], frozen) is None
                    for j in range(len(state))
                    if j != i and j not in fixed
                ):
                    continue
                state = tuple(trial)
                moves.extend(added)
                fixed.add(i)
                break
            else:
                return None
        result = tuple(move.inverse() for move in reversed(moves)) if reverse else tuple(moves)
        replay(spec, result)
        return result
