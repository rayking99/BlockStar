"""Exact time-expanded replanning of one piece amid the other pieces' trajectories."""

from __future__ import annotations

import random
from collections import deque

from .engine import Engine
from .model import DIRECTIONS, Move, PuzzleSpec, replay


def replan_piece(
    engine: Engine, moves: tuple[Move, ...], piece: int, *, rng: random.Random | None = None
) -> tuple[Move, ...]:
    """Minimize this piece's slides while preserving the order of all other moves.

    A vertex is (other-move index, piece position). A legal slide costs one;
    allowing the next other-piece move costs zero. 0-1 BFS finds an exact optimum
    for this restricted subproblem, including moving this piece earlier/later.
    """
    spec = engine.spec
    name = spec.pieces[piece].name
    others = tuple(move for move in moves if move.piece != name)
    index = {p.name: i for i, p in enumerate(spec.pieces)}
    state = list(spec.initial)
    blocked, sweeps = [], []
    mask = engine.occupied(tuple(state)) ^ engine.masks[piece][state[piece]]
    blocked.append(mask)
    for move in others:
        i = index[move.piece]
        start = state[i]
        dr, dc = DIRECTIONS[move.direction]
        end = start + (dr * spec.width + dc) * move.distance
        sweeps.append(engine.corridor(i, start, (end,)))
        mask ^= engine.masks[i][start] ^ engine.masks[i][end]
        blocked.append(mask)
        state[i] = end
    original_cost = sum(move.piece == name for move in moves)
    if original_cost == 0:
        return moves
    # With an RNG, explore equally short schedules to escape coordinate minima.
    limit = original_cost + (1 if rng is not None else 0)
    direction_order = list(range(4))
    far_first = False
    if rng is not None:
        rng.shuffle(direction_order)
        far_first = rng.random() < 0.5
    cells = spec.width * spec.height
    start = spec.initial[piece]
    finish = len(others) * cells + spec.goal[piece]
    dist = {start: 0}
    parent: dict[int, int] = {}
    queue = deque([(start, 0)])
    masks, rays = engine.masks[piece], engine.rays[piece]
    while queue:
        node, cost = queue.popleft()
        if dist[node] != cost:
            continue
        if node == finish:
            if cost >= limit:
                return moves
            path = []
            while node != start:
                prev = parent[node]
                t, pos = divmod(node, cells)
                pt, ppos = divmod(prev, cells)
                path.append(others[pt] if t > pt else engine.moves(piece, ppos, (pos,))[0])
                node = prev
            return tuple(reversed(path))
        t, pos = divmod(node, cells)
        if t < len(others) and not masks[pos] & sweeps[t]:
            nxt = node + cells
            if cost < dist.get(nxt, limit):
                dist[nxt] = cost
                parent[nxt] = node
                queue.appendleft((nxt, cost))
        if cost + 1 >= limit:
            continue
        for direction in direction_order:
            free = []
            for next_pos in rays[pos][direction]:
                if masks[next_pos] & blocked[t]:
                    break
                free.append(next_pos)
            if far_first:
                free.reverse()
            for next_pos in free:
                nxt = t * cells + next_pos
                if cost + 1 < dist.get(nxt, limit):
                    dist[nxt] = cost + 1
                    parent[nxt] = node
                    queue.append((nxt, cost + 1))
    return moves


def optimize(
    spec: PuzzleSpec,
    moves: tuple[Move, ...],
    *,
    seed: int = 0,
    passes: int = 3,
    engine: Engine | None = None,
    explore: bool = False,
) -> tuple[Move, ...]:
    """Coordinate descent, accepting replayed runs that never increase slide count.

    ``explore`` admits equal-cost alternatives and allows three plateau passes.
    """
    if passes < 0:
        raise ValueError("passes must be nonnegative")
    replay(spec, moves)
    engine = engine or Engine(spec)
    rng = random.Random(seed)
    pieces = list(range(len(spec.pieces)))
    plateau = 0
    for _ in range(passes):
        before = len(moves)
        rng.shuffle(pieces)
        for i in pieces:
            candidate = replan_piece(engine, moves, i, rng=rng if explore else None)
            if candidate != moves and len(candidate) <= len(moves):
                replay(spec, candidate)
                moves = candidate
        plateau = plateau + 1 if len(moves) == before else 0
        if plateau >= (3 if explore else 1):
            break
    return moves
