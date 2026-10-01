import random
import unittest
from collections import deque

from blockstar import Move, PuzzleSpec, replay
from blockstar.discovery import DiscoveryConfig, discover
from blockstar.engine import Engine
from blockstar.optimize import optimize
from blockstar.search import Planner, Policy


class SearchTests(unittest.TestCase):
    def test_path_minimizes_slides_not_cells(self):
        spec = PuzzleSpec.from_boards(["A----", "--B--", "-----"], ["-----", "--B--", "----A"])
        engine = Engine(spec)
        path = engine.path(spec.initial, 0, spec.goal[0])
        self.assertEqual(len(path), 2)
        replay(spec, engine.moves(0, spec.initial[0], path))

    def test_paths_match_independent_grid_bfs(self):
        rng = random.Random(123)
        for _ in range(40):
            start = rng.randrange(16)
            finish = rng.randrange(16)
            obstacles = set(rng.sample([i for i in range(16) if i not in (start, finish)], 4))
            board = [["-"] * 4 for _ in range(4)]
            for n, pos in enumerate(sorted(obstacles)):
                board[pos // 4][pos % 4] = str(n)
            board[start // 4][start % 4] = "A"
            target = [row.copy() for row in board]
            target[start // 4][start % 4] = "-"
            target[finish // 4][finish % 4] = "A"
            spec = PuzzleSpec.from_boards(board, target)
            queue, distances = deque([start]), {start: 0}
            while queue:
                pos = queue.popleft()
                r, c = divmod(pos, 4)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 4 and 0 <= nc < 4 and nr * 4 + nc not in obstacles:
                        nxt = nr * 4 + nc
                        if nxt not in distances:
                            distances[nxt] = distances[pos] + 1
                            queue.append(nxt)
                        nr, nc = nr + dr, nc + dc
            engine = Engine(spec)
            i = [p.name for p in spec.pieces].index("A")
            path = engine.path(spec.initial, i, spec.goal[i])
            self.assertEqual(None if path is None else len(path), distances.get(finish))
            if path is not None:
                replay(spec, engine.moves(i, start, path))

    def test_optimizer_reschedules_across_other_pieces(self):
        spec = PuzzleSpec.from_boards(["A--B", "----"], ["---A", "B---"])
        moves = tuple(
            Move(*m)
            for m in [
                ("A", "right", 2),
                ("B", "down", 1),
                ("A", "left", 1),
                ("B", "left", 3),
                ("A", "right", 2),
            ]
        )
        result = optimize(spec, moves)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result), spec.lower_bound)
        replay(spec, result)

    def test_discovery_from_scratch_is_reproducible_and_sub_100(self):
        spec = PuzzleSpec.load()
        config = DiscoveryConfig(trials=16, seed=0)
        a, b = discover(spec, config), discover(spec, config)
        self.assertIsNotNone(a.moves)
        self.assertLess(len(a.moves), 100)
        self.assertEqual(a.moves, b.moves)
        self.assertEqual(a.report["records"], b.report["records"])
        replay(spec, a.moves)
        self.assertEqual(a.report["successes"] + a.report["failures"], config.trials)

    def test_no_piece_specific_order(self):
        original = PuzzleSpec.load()
        mapping = {p.name: str(99 - i) for i, p in enumerate(original.pieces)}
        spec = PuzzleSpec.from_boards(
            [[mapping.get(c, c) for c in r] for r in original.start],
            [[mapping.get(c, c) for c in r] for r in original.target],
        )
        result = discover(spec, DiscoveryConfig(trials=16, seed=7))
        self.assertIsNotNone(result.moves)
        replay(spec, result.moves)

    def test_identity_and_failed_search(self):
        spec = PuzzleSpec.from_boards(["AB"], ["AB"])
        self.assertEqual(discover(spec, DiscoveryConfig(trials=1)).moves, ())
        blocked = PuzzleSpec.from_boards(["AB"], ["BA"])
        result = discover(blocked, DiscoveryConfig(trials=4))
        self.assertIsNone(result.moves)
        self.assertEqual(result.report["failures"], 4)

    def test_board_orientation_is_not_hardcoded(self):
        original = PuzzleSpec.load()
        variants = (
            ([row[::-1] for row in original.start], [row[::-1] for row in original.target]),
            (list(zip(*original.start, strict=True)), list(zip(*original.target, strict=True))),
        )
        for start, target in variants:
            spec = PuzzleSpec.from_boards(start, target)
            result = discover(spec, DiscoveryConfig(trials=24, seed=7))
            self.assertIsNotNone(result.moves)
            replay(spec, result.moves)

    def test_equal_cost_exploration_stays_legal_and_never_worsens(self):
        spec = PuzzleSpec.load()
        moves = Planner(spec).rollout(0)
        result = optimize(spec, moves, seed=19, passes=5, explore=True)
        replay(spec, result)
        self.assertLessEqual(len(result), len(moves))

    def test_policy_and_budget_validation(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Policy(temperature=value)
        with self.assertRaises(ValueError):
            DiscoveryConfig(trials=0)

    def test_rollouts_are_independently_replayed(self):
        spec = PuzzleSpec.load()
        planner = Planner(spec)
        for seed in range(10):
            moves = planner.rollout(seed, reverse=seed % 2 == 0)
            if moves is not None:
                replay(spec, moves)
