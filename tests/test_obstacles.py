import unittest

from blockstar import Move, PuzzleSpec, replay
from blockstar.discovery import DiscoveryConfig, discover
from blockstar.engine import Engine
from blockstar.learning import ActionSpace
from blockstar.optimize import optimize


class FixedObstacleTests(unittest.TestCase):
    def setUp(self):
        self.spec = PuzzleSpec.from_boards(["-----", "A-#--", "-----"], ["-----", "--#-A", "-----"])

    def test_fixed_terrain_is_not_a_piece_and_cannot_move(self):
        self.assertEqual([p.name for p in self.spec.pieces], ["A"])
        self.assertEqual(self.spec.fixed_cells, ((1, 2),))
        with self.assertRaisesRegex(ValueError, "cannot move"):
            Move("#", "right", 1)
        with self.assertRaisesRegex(ValueError, "collides with #"):
            replay(self.spec, [Move("A", "right", 4)])
        with self.assertRaisesRegex(ValueError, "identical cells"):
            PuzzleSpec.from_boards(["A-#"], ["#-A"])

    def test_relaxed_paths_and_optimizer_must_detour(self):
        engine = Engine(self.spec)
        # A caller may ignore movable pieces, but cannot turn off permanent terrain.
        path = engine.find_path(0, self.spec.initial[0], self.spec.goal[0], 0)
        self.assertEqual(len(path), 3)
        moves = engine.moves(0, self.spec.initial[0], path)
        for explore in (False, True):
            result = optimize(self.spec, moves, passes=4, explore=explore)
            self.assertEqual(len(result), 3)
            self.assertTrue(replay(self.spec, result).solved)

    def test_wall_cannot_be_jumped_by_rays_or_parked(self):
        spec = PuzzleSpec.from_boards(["--#--", "A-#--", "--#--"], ["--#--", "--#-A", "--#--"])
        engine = Engine(spec)
        self.assertIsNone(engine.find_path(0, spec.initial[0], spec.goal[0], 0))
        reachable = engine.reachable(0, spec.initial[0], 0)
        self.assertTrue(all(position % spec.width < 2 for position in reachable))
        self.assertIsNone(discover(spec, DiscoveryConfig(trials=4)).moves)

    def test_forward_reverse_and_action_masks_respect_fixed_cells(self):
        result = discover(self.spec, DiscoveryConfig(trials=4))
        self.assertIsNotNone(result.moves)
        self.assertEqual(len(result.moves), 3)
        self.assertTrue(replay(self.spec, result.moves).solved)
        actions = ActionSpace(self.spec)
        self.assertFalse(any(m.piece == "#" for m in actions.moves))
        for i in actions.legal_indices(self.spec.initial):
            replay(self.spec, [actions.moves[i]], require_solved=False)

    def test_new_board_is_larger_and_has_two_obstacles(self):
        spec = PuzzleSpec.load("blockstar/data/large-yard.json")
        self.assertEqual((spec.height, spec.width, len(spec.pieces)), (28, 30, 24))
        remaining, components = set(spec.fixed_cells), []
        while remaining:
            todo, component = [remaining.pop()], set()
            while todo:
                r, c = todo.pop()
                component.add((r, c))
                for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        todo.append(neighbor)
            components.append(component)
        self.assertEqual(sorted(map(len, components)), [20, 28])
        i = [p.name for p in spec.pieces].index("I")
        engine = Engine(spec)
        self.assertEqual(len(engine.find_path(i, spec.initial[i], spec.goal[i], 0)), 3)
