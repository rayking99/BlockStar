"""Versioned solution files, atomic writes, and untrusted dataset auditing."""

from __future__ import annotations

import ast
import json
import os
import statistics
import tempfile
from pathlib import Path

from .model import Move, PuzzleSpec, replay


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def solution_document(spec: PuzzleSpec, moves: tuple[Move, ...], **metadata) -> dict:
    result = replay(spec, moves)
    return {
        "schema_version": 1,
        "puzzle": spec.name,
        "puzzle_sha256": spec.fingerprint,
        "score": result.move_count,
        "unit_distance": result.unit_distance,
        "lower_bound": spec.lower_bound,
        "solved": True,
        "moves": [m.as_list() for m in moves],
        "metadata": metadata,
    }


def parse_moves(data) -> tuple[Move, ...]:
    if not isinstance(data, (list, tuple)):
        raise ValueError("Moves must be a list of [piece, direction, distance] triples")
    moves = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("Each move must be [piece, direction, distance]")
        moves.append(Move(*item))
    return tuple(moves)


def load_solution(path: str | Path, spec: PuzzleSpec) -> tuple[Move, ...]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Expected a schema_version 1 solution document")
    if document.get("puzzle_sha256") != spec.fingerprint:
        raise ValueError("Solution belongs to a different puzzle")
    moves = parse_moves(document.get("moves"))
    result = replay(spec, moves)
    if document.get("score") != result.move_count:
        raise ValueError("Stored score does not match the replayed move count")
    if "unit_distance" in document and document["unit_distance"] != result.unit_distance:
        raise ValueError("Stored distance does not match the replayed distance")
    return moves


def audit_dataset(spec: PuzzleSpec, path: str | Path) -> tuple[dict, list[tuple[Move, ...]]]:
    """Read historical Python-literal or JSONL trajectories without executing code."""
    valid, rejected, seen = [], [], set()
    records, duplicates = 0, 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            records += 1
            try:
                moves = parse_moves(ast.literal_eval(line))
                replay(spec, moves)
                valid.append(moves)
                if moves in seen:
                    duplicates += 1
                seen.add(moves)
            except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
                rejected.append({"line": line_number, "reason": str(exc)})
    scores = [len(m) for m in valid]
    report = {
        "schema_version": 1,
        "puzzle_sha256": spec.fingerprint,
        "source": str(path),
        "records": records,
        "valid": len(valid),
        "invalid": len(rejected),
        "duplicates": duplicates,
        "best": min(scores, default=None),
        "median": statistics.median(scores) if scores else None,
        "mean": statistics.mean(scores) if scores else None,
        "worst": max(scores, default=None),
        "rejections": rejected,
    }
    return report, valid
