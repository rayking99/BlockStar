"""Command-line discovery, benchmarks, strict replay, and dataset audit."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .artifacts import audit_dataset, load_solution, solution_document, write_json
from .discovery import DiscoveryConfig, discover
from .engine import Engine
from .model import PuzzleSpec, replay
from .optimize import optimize


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _progress(event: dict) -> None:
    print(
        f"trial {event['trial']}: {event['score']} moves (raw {event['raw_score']}, "
        f"{event['elapsed_seconds']:.2f}s)",
        file=sys.stderr,
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="BlockStar geometric discovery and independently validated benchmarks"
    )
    result.add_argument(
        "--puzzle", type=Path, help="JSON puzzle; defaults to the original 20x18 board"
    )
    commands = result.add_subparsers(dest="command", required=True)
    find = commands.add_parser("discover", help="discover a solution from the boards alone")
    find.add_argument("--trials", type=_positive, default=200)
    find.add_argument("--seed", type=int, default=0)
    find.add_argument("--fixed-policy", action="store_true", help="ablate evolutionary tuning")
    find.add_argument(
        "--passes", type=_nonnegative, default=3, help="refinement passes; 0 disables"
    )
    find.add_argument("--refine-margin", type=_nonnegative, default=3)
    find.add_argument(
        "--explore", action="store_true", help="explore equal-cost schedules during refinement"
    )
    find.add_argument("--output", type=Path, default=Path("runs/best.json"))
    find.add_argument("--report", type=Path, default=Path("runs/discovery.json"))
    find.add_argument("--quiet", action="store_true")
    validate = commands.add_parser("replay", help="validate every slide and final equality")
    validate.add_argument("solution", type=Path)
    validate.add_argument("--show-board", action="store_true")
    audit = commands.add_parser(
        "audit", help="replay historical results without trusting their labels"
    )
    audit.add_argument("dataset", type=Path)
    audit.add_argument("--output", type=Path)
    audit.add_argument(
        "--strict", action="store_true", help="return a failing exit code if any record is rejected"
    )
    polish = commands.add_parser(
        "optimize", help="reschedule a valid solution without increasing its score"
    )
    polish.add_argument("solution", type=Path)
    polish.add_argument("--restarts", type=_positive, default=20)
    polish.add_argument("--passes", type=_positive, default=12)
    polish.add_argument("--seed", type=int, default=0)
    polish.add_argument("--output", type=Path, default=Path("runs/optimized.json"))
    polish.add_argument("--report", type=Path, default=Path("runs/optimization.json"))
    bench = commands.add_parser(
        "benchmark", help="compare fixed, adaptive, and refined search across seeds"
    )
    bench.add_argument("--trials", type=_positive, default=200)
    bench.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    bench.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    bench.add_argument("--solutions", type=Path, default=Path("runs/solutions"))
    return result


def _benchmark(spec: PuzzleSpec, args: argparse.Namespace) -> int:
    rows = []
    for method, adaptive, passes in (
        ("fixed", False, 0),
        ("adaptive", True, 0),
        ("adaptive_refined", True, 3),
    ):
        for seed in args.seeds:
            result = discover(spec, DiscoveryConfig(args.trials, seed, adaptive, passes))
            row = {
                k: result.report[k]
                for k in (
                    "trials",
                    "successes",
                    "failures",
                    "raw_best",
                    "best_score",
                    "elapsed_seconds",
                )
            }
            row.update(method=method, seed=seed)
            rows.append(row)
            write_json(args.solutions / f"{method}-{seed}.report.json", result.report)
            if result.moves is not None:
                write_json(
                    args.solutions / f"{method}-{seed}.json",
                    solution_document(
                        spec,
                        result.moves,
                        config=asdict(DiscoveryConfig(args.trials, seed, adaptive, passes)),
                    ),
                )
            print(json.dumps(row), file=sys.stderr, flush=True)
    summary = {}
    for method in ("fixed", "adaptive", "adaptive_refined"):
        group = [row for row in rows if row["method"] == method]
        scores = [row["best_score"] for row in group if row["best_score"] is not None]
        summary[method] = {
            "successful_searches": len(scores),
            "searches": len(group),
            "best": min(scores, default=None),
            "worst": max(scores, default=None),
            "median": statistics.median(scores) if scores else None,
            "mean": statistics.mean(scores) if scores else None,
            "rollout_success_rate": sum(row["successes"] for row in group)
            / sum(row["trials"] for row in group),
            "total_seconds": sum(row["elapsed_seconds"] for row in group),
        }
    report = {
        "schema_version": 1,
        "puzzle_sha256": spec.fingerprint,
        "trials_per_search": args.trials,
        "seeds": args.seeds,
        "environment": result.report["environment"],
        "budget_note": "Equal rollout budgets; refinement adds compute. Times include replay and refinement.",
        "summary": summary,
        "runs": rows,
    }
    write_json(args.output, report)
    print(json.dumps(summary, indent=2))
    return 0 if all(row["best_score"] is not None for row in rows) else 1


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        spec = PuzzleSpec.load(args.puzzle)
        if args.command == "discover":
            if args.output.resolve() == args.report.resolve():
                raise ValueError("Solution and report paths must differ")
            config = DiscoveryConfig(
                args.trials,
                args.seed,
                not args.fixed_policy,
                args.passes,
                args.refine_margin,
                args.explore,
            )
            result = discover(spec, config, progress=None if args.quiet else _progress)
            write_json(args.report, result.report)
            if result.moves is None:
                print(
                    "No solution found within this search budget; this does not prove unsolvability.",
                    file=sys.stderr,
                )
                return 1
            write_json(args.output, solution_document(spec, result.moves, config=asdict(config)))
            print(
                json.dumps(
                    {
                        "solved": True,
                        "moves": len(result.moves),
                        "solution": str(args.output),
                        "report": str(args.report),
                        "seconds": result.report["elapsed_seconds"],
                    }
                )
            )
        elif args.command == "replay":
            moves = load_solution(args.solution, spec)
            result = replay(spec, moves)
            print(
                json.dumps(
                    {
                        "solved": result.solved,
                        "moves": result.move_count,
                        "unit_distance": result.unit_distance,
                    }
                )
            )
            if args.show_board:
                print("\n".join(" ".join(row) for row in result.board))
        elif args.command == "audit":
            report, _ = audit_dataset(spec, args.dataset)
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, indent=2))
            return int(args.strict and report["invalid"] > 0)
        elif args.command == "optimize":
            if args.output.resolve() == args.report.resolve():
                raise ValueError("Solution and report paths must differ")
            moves = load_solution(args.solution, spec)
            initial_score, started = len(moves), time.perf_counter()
            engine, history = Engine(spec), []
            for restart in range(args.restarts):
                moves = optimize(
                    spec,
                    moves,
                    seed=args.seed + restart,
                    passes=args.passes,
                    explore=True,
                    engine=engine,
                )
                history.append(
                    {
                        "restart": restart,
                        "score": len(moves),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            metadata = {
                "input": str(args.solution),
                "initial_score": initial_score,
                "seed": args.seed,
                "restarts": args.restarts,
                "passes": args.passes,
            }
            write_json(args.output, solution_document(spec, moves, **metadata))
            write_json(
                args.report,
                {
                    **metadata,
                    "puzzle_sha256": spec.fingerprint,
                    "final_score": len(moves),
                    "elapsed_seconds": time.perf_counter() - started,
                    "history": history,
                },
            )
            print(
                json.dumps(
                    {"before": initial_score, "after": len(moves), "output": str(args.output)}
                )
            )
        elif args.command == "benchmark":
            return _benchmark(spec, args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0
