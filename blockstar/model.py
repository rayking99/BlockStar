"""Immutable puzzle definitions and a strict, independent cell-by-cell referee."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
Board = tuple[tuple[str, ...], ...]
State = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Move:
    piece: str
    direction: str
    distance: int

    def __post_init__(self) -> None:
        if self.piece == "#":
            raise ValueError("Fixed obstacles (#) cannot move")
        if not isinstance(self.piece, str) or not self.piece or self.piece == "-":
            raise ValueError("A move must name a piece")
        if not isinstance(self.direction, str) or self.direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {self.direction!r}")
        if type(self.distance) is not int or self.distance <= 0:
            raise ValueError("Move distance must be a positive integer")

    def as_list(self) -> list:
        return [self.piece, self.direction, self.distance]

    def inverse(self) -> Move:
        return Move(self.piece, OPPOSITE[self.direction], self.distance)


@dataclass(frozen=True, slots=True)
class Piece:
    name: str
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class PuzzleSpec:
    name: str
    start: Board
    target: Board
    pieces: tuple[Piece, ...]
    initial: State
    goal: State

    @classmethod
    def from_boards(cls, start: Iterable, target: Iterable, name: str = "puzzle") -> PuzzleSpec:
        start, target = tuple(map(tuple, start)), tuple(map(tuple, target))
        if not start or not start[0]:
            raise ValueError("Boards must be nonempty")
        height, width = len(start), len(start[0])
        if len(target) != height or any(len(row) != width for row in (*start, *target)):
            raise ValueError("Boards must be rectangular and have identical dimensions")
        if any(
            not isinstance(cell, str) or not cell
            for board in (start, target)
            for row in board
            for cell in row
        ):
            raise ValueError("Board cells must be nonempty strings; '-' means empty")
        fixed_start = {
            (r, c) for r, row in enumerate(start) for c, cell in enumerate(row) if cell == "#"
        }
        fixed_target = {
            (r, c) for r, row in enumerate(target) for c, cell in enumerate(row) if cell == "#"
        }
        if fixed_start != fixed_target:
            raise ValueError("Fixed obstacles (#) must occupy identical cells in both boards")
        names = sorted({cell for row in start for cell in row} - {"-", "#"})
        if set(names) != {cell for row in target for cell in row} - {"-", "#"}:
            raise ValueError("Start and target must contain the same pieces")
        pieces, initial, goal = [], [], []
        for label in names:
            shapes, anchors = [], []
            for board in (start, target):
                cells = [
                    (r, c)
                    for r, row in enumerate(board)
                    for c, cell in enumerate(row)
                    if cell == label
                ]
                top, left = min(r for r, _ in cells), min(c for _, c in cells)
                shapes.append(tuple((r - top, c - left) for r, c in cells))
                anchors.append(top * width + left)
            if shapes[0] != shapes[1]:
                raise ValueError(
                    f"Piece {label!r} changes shape between boards (rotation is unsupported)"
                )
            # A label denotes one connected rigid body, including nonrectangular bodies.
            reached, todo, shape = set(), [shapes[0][0]], set(shapes[0])
            while todo:
                cell = todo.pop()
                if cell in reached:
                    continue
                reached.add(cell)
                todo.extend(
                    (cell[0] + dr, cell[1] + dc)
                    for dr, dc in DIRECTIONS.values()
                    if (cell[0] + dr, cell[1] + dc) in shape - reached
                )
            if reached != shape:
                raise ValueError(f"Piece {label!r} is disconnected")
            pieces.append(Piece(label, shapes[0]))
            initial.append(anchors[0])
            goal.append(anchors[1])
        return cls(name, start, target, tuple(pieces), tuple(initial), tuple(goal))

    @classmethod
    def load(cls, path: str | Path | None = None) -> PuzzleSpec:
        source = Path(path) if path else files("blockstar").joinpath("data/default.json")
        data = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_boards(data["start"], data["target"], data.get("name", "puzzle"))

    @property
    def width(self) -> int:
        return len(self.start[0])

    @property
    def height(self) -> int:
        return len(self.start)

    @property
    def fixed_cells(self) -> tuple[tuple[int, int], ...]:
        """Permanent terrain, never part of the movable-piece state or action space."""
        return tuple(
            (r, c) for r, row in enumerate(self.start) for c, cell in enumerate(row) if cell == "#"
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps([self.start, self.target], separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def lower_bound(self) -> int:
        """Obstacle-free lower bound in slides, not a claim of optimality."""
        return sum(
            (a // self.width != b // self.width) + (a % self.width != b % self.width)
            for a, b in zip(self.initial, self.goal, strict=True)
        )


@dataclass(frozen=True, slots=True)
class Replay:
    board: Board
    move_count: int
    unit_distance: int
    solved: bool


def replay(spec: PuzzleSpec, moves: Iterable[Move], *, require_solved: bool = True) -> Replay:
    """Validate EVERY swept cell, independently of the solver's bitboard implementation.

    One positive, straight slide is one move regardless of distance. No rotations,
    simultaneous moves, overlaps, jumping, or off-board movement are allowed.
    '#' cells are permanent obstacles and must remain unchanged.
    """
    board = [list(row) for row in spec.start]
    count, distance = 0, 0
    labels = {piece.name for piece in spec.pieces}
    for count, move in enumerate(moves, 1):
        if not isinstance(move, Move):
            raise ValueError(f"Move {count}: expected a Move")
        if move.piece not in labels:
            raise ValueError(f"Move {count}: unknown piece {move.piece!r}")
        cells = {
            (r, c)
            for r, row in enumerate(board)
            for c, cell in enumerate(row)
            if cell == move.piece
        }
        dr, dc = DIRECTIONS[move.direction]
        # Check the whole translation before committing any part of this move.
        for step in range(1, move.distance + 1):
            for r, c in cells:
                nr, nc = r + dr * step, c + dc * step
                if not (0 <= nr < spec.height and 0 <= nc < spec.width):
                    raise ValueError(f"Move {count} {move.as_list()}: leaves board")
                if board[nr][nc] not in ("-", move.piece):
                    raise ValueError(
                        f"Move {count} {move.as_list()}: collides with {board[nr][nc]}"
                    )
        for r, c in cells:
            board[r][c] = "-"
        for r, c in cells:
            board[r + dr * move.distance][c + dc * move.distance] = move.piece
        distance += move.distance
    final = tuple(map(tuple, board))
    solved = final == spec.target
    if require_solved and not solved:
        raise ValueError("Legal moves, but the final board does not match the target")
    return Replay(final, count, distance, solved)
