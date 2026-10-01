"""Seeded evolutionary policy search, with isolated RNGs and replayed incumbents."""

from __future__ import annotations

import platform
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .model import Move, PuzzleSpec, replay
from .optimize import optimize
from .search import Planner, Policy


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    trials: int = 200
    seed: int = 0
    adaptive: bool = True
    optimize_passes: int = 3
    refine_margin: int = 3
    explore: bool = False

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ValueError("trials must be positive")
        if self.optimize_passes < 0 or self.refine_margin < 0:
            raise ValueError("optimize_passes and refine_margin must be nonnegative")


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    moves: tuple[Move, ...] | None
    report: dict


def _random_policy(rng: random.Random) -> Policy:
    return Policy(
        path_weight=rng.uniform(0.2, 3),
        blocker_weight=rng.uniform(0, 5),
        parking_distance=rng.uniform(0, 4),
        goal_overlap=rng.uniform(0, 3),
        temperature=rng.uniform(0, 5),
        order_noise=rng.uniform(0, 12),
    )


def _mutate(policy: Policy, rng: random.Random) -> Policy:
    values = asdict(policy)
    for key, value in values.items():
        if rng.random() < 0.65:
            values[key] = min(
                20.0, max(0.0, value * rng.lognormvariate(0, 0.4) + rng.gauss(0, 0.15))
            )
    return Policy(**values)


def discover(
    spec: PuzzleSpec,
    config: DiscoveryConfig | None = None,
    *,
    progress: Callable[[dict], None] | None = None,
) -> DiscoveryResult:
    """Each policy is assessed on two fresh rollouts; low mean scores breed successors.

    Evolution learns geometric decision weights and search direction. Twenty percent
    of generations explore fresh parameters. All randomness comes from ``seed``;
    stopping uses work counts, not wall time. Neither the upstream dataset nor a
    stored solution is read. Optional refinement only uses newly discovered runs.
    """
    started = time.perf_counter()
    config = config or DiscoveryConfig()
    planner = Planner(spec)
    rng = random.Random(config.seed)
    best: tuple[Move, ...] | None = None
    raw_best: int | None = None
    history, records, elites = [], [], []
    refined: set[tuple[Move, ...]] = set()
    failures, refined_count = 0, 0
    trial = 0
    # Fitness penalty, scaled by problem size, only ranks failed policy samples.
    # Failures are never counted as solutions or omitted from the report.
    failure_cost = max(100, len(spec.pieces) * 20)
    while trial < config.trials:
        if not config.adaptive or trial < 4:
            policy, reverse = Policy(), (trial // 2) % 2 == 0
        elif len(elites) < 6 or rng.random() < 0.2:
            policy, reverse = _random_policy(rng), rng.random() < 0.5
        else:
            parent = min(rng.sample(elites, min(3, len(elites))), key=lambda item: item[0])
            policy = _mutate(parent[2], rng)
            reverse = parent[3] if rng.random() > 0.1 else not parent[3]
        costs = []
        for _ in range(min(2, config.trials - trial)):
            rollout_seed = rng.randrange(2**63)
            moves = planner.rollout(rollout_seed, reverse=reverse, policy=policy)
            raw_score = len(moves) if moves is not None else None
            costs.append(raw_score if raw_score is not None else failure_cost)
            record = {
                "trial": trial,
                "seed": rollout_seed,
                "reverse": reverse,
                "policy": asdict(policy),
                "raw_score": raw_score,
                "refined_score": None,
            }
            if moves is None:
                failures += 1
            else:
                raw_best = raw_score if raw_best is None else min(raw_best, raw_score)
                if (
                    config.optimize_passes
                    and moves not in refined
                    and (best is None or len(moves) <= len(best) + config.refine_margin)
                ):
                    refined.add(moves)
                    refined_count += 1
                    moves = optimize(
                        spec,
                        moves,
                        seed=rollout_seed,
                        passes=config.optimize_passes,
                        engine=planner.engine,
                        explore=config.explore,
                    )
                    record["refined_score"] = len(moves)
                if best is None or len(moves) < len(best):
                    replay(spec, moves)
                    best = moves
                    event = {
                        "trial": trial,
                        "raw_score": raw_score,
                        "score": len(best),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    history.append(event)
                    if progress:
                        progress(event)
            records.append(record)
            trial += 1
        elites.append((sum(costs) / len(costs), trial, policy, reverse))
        elites.sort(key=lambda item: (item[0], item[1]))
        del elites[12:]
    report = {
        "schema_version": 1,
        "puzzle_sha256": spec.fingerprint,
        "puzzle": spec.name,
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "trials": trial,
        "successes": trial - failures,
        "failures": failures,
        "raw_best": raw_best,
        "best_score": len(best) if best is not None else None,
        "lower_bound": spec.lower_bound,
        "refinements": refined_count,
        "incumbents": history,
        "elites": [
            {"mean_raw_cost": score, "reverse": rev, "policy": asdict(pol)}
            for score, _, pol, rev in elites
        ],
        "records": records,
    }
    return DiscoveryResult(best, report)
