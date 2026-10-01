"""Compact bitboard geometry and shortest paths measured in whole slides."""

from __future__ import annotations

from collections import deque

from .model import Move, PuzzleSpec, State


class Engine:
    def __init__(self, spec: PuzzleSpec):
        self.spec = spec
        self.masks: list[list[int]] = []
        self.rays: list[list[tuple[tuple[int, ...], ...]]] = []
        width, height = spec.width, spec.height
        self.fixed_mask = sum(1 << (r * width + c) for r, c in spec.fixed_cells)
        for piece in spec.pieces:
            masks = [0] * (width * height)
            for r in range(height):
                for c in range(width):
                    if all(r + dr < height and c + dc < width for dr, dc in piece.cells):
                        masks[r * width + c] = sum(
                            1 << ((r + dr) * width + c + dc) for dr, dc in piece.cells
                        )
            self.masks.append(masks)
            rays = []
            for pos in range(width * height):
                r, c = divmod(pos, width)
                rays.append(
                    tuple(
                        tuple(p for p in ray if masks[p])
                        for ray in (
                            range(pos - width, -1, -width),
                            range(pos + width, width * height, width),
                            range(pos - 1, r * width - 1, -1),
                            range(pos + 1, (r + 1) * width),
                        )
                    )
                )
            self.rays.append(rays)

    def occupied(self, state: State) -> int:
        result = self.fixed_mask
        for masks, pos in zip(self.masks, state, strict=True):
            result |= masks[pos]
        return result

    def path(
        self, state: State, piece: int, target: int, occupied: int | None = None
    ) -> tuple[int, ...] | None:
        """Breadth-first search over all legal positive slide distances (unit edge cost)."""
        start = state[piece]
        if start == target:
            return ()
        masks = self.masks[piece]
        blocked = (self.occupied(state) if occupied is None else occupied) ^ masks[start]
        return self.find_path(piece, start, target, blocked)

    def find_path(
        self, piece: int, start: int, target: int, blocked: int
    ) -> tuple[int, ...] | None:
        # Permanent terrain applies even to relaxed searches that ignore movable blocks.
        blocked |= self.fixed_mask
        if start == target:
            return ()
        masks = self.masks[piece]
        if not masks[target] or masks[target] & blocked:
            return None
        parents = {start: -1}
        queue = deque([start])
        rays = self.rays[piece]
        while queue:
            pos = queue.popleft()
            for ray in rays[pos]:
                for nxt in ray:
                    if masks[nxt] & blocked:
                        break
                    if nxt in parents:
                        continue
                    parents[nxt] = pos
                    if nxt == target:
                        path = [nxt]
                        while parents[path[-1]] != start:
                            path.append(parents[path[-1]])
                        return tuple(reversed(path))
                    queue.append(nxt)
        return None

    def reachable(self, piece: int, start: int, blocked: int) -> dict[int, tuple[int, ...]]:
        blocked |= self.fixed_mask
        paths: dict[int, tuple[int, ...]] = {start: ()}
        queue = deque([start])
        masks, rays = self.masks[piece], self.rays[piece]
        while queue:
            pos = queue.popleft()
            for ray in rays[pos]:
                for nxt in ray:
                    if masks[nxt] & blocked:
                        break
                    if nxt not in paths:
                        paths[nxt] = (*paths[pos], nxt)
                        queue.append(nxt)
        return paths

    def corridor(self, piece: int, start: int, path: tuple[int, ...]) -> int:
        result = self.masks[piece][start]
        width = self.spec.width
        for pos in path:
            step = (
                (width if pos > start else -width)
                if start % width == pos % width
                else (1 if pos > start else -1)
            )
            for cell in range(start + step, pos + step, step):
                result |= self.masks[piece][cell]
            start = pos
        return result

    def moves(self, piece: int, start: int, path: tuple[int, ...]) -> tuple[Move, ...]:
        result = []
        width = self.spec.width
        for pos in path:
            r, c = divmod(start, width)
            nr, nc = divmod(pos, width)
            direction = ("down" if nr > r else "up") if nr != r else ("right" if nc > c else "left")
            result.append(Move(self.spec.pieces[piece].name, direction, abs(nr - r) + abs(nc - c)))
            start = pos
        return tuple(result)
