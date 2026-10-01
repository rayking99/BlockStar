import unittest

from blockstar import Move, PuzzleSpec, replay


class RefereeTests(unittest.TestCase):
    def test_swept_collision_even_when_destination_is_empty(self):
        spec = PuzzleSpec.from_boards(["A-B--"], ["--B-A"])
        with self.assertRaisesRegex(ValueError, "collides with B"):
            replay(spec, [Move("A", "right", 4)])

    def test_rectangle_can_overlap_its_old_footprint(self):
        spec = PuzzleSpec.from_boards(["AA--", "AA--"], ["--AA", "--AA"])
        result = replay(spec, [Move("A", "right", 2)])
        self.assertTrue(result.solved)
        self.assertEqual((result.move_count, result.unit_distance), (1, 2))

    def test_nonrectangular_shape(self):
        spec = PuzzleSpec.from_boards(["A---", "AA--"], ["--A-", "--AA"])
        self.assertTrue(replay(spec, [Move("A", "right", 2)]).solved)

    def test_rejects_bad_moves_and_boards(self):
        for distance in (0, -1, True, 1.5, "2"):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                Move("A", "right", distance)
        with self.assertRaises(ValueError):
            Move("A", "diagonal", 1)
        for start, target in [
            ([], []),
            (["A-", "---"], ["A--"]),
            (["A-"], ["B-"]),
            (["AA"], ["A-"]),
            (["A-A"], ["A-A"]),
            (["AA", "--"], ["A-", "A-"]),
        ]:
            with self.subTest(start=start, target=target), self.assertRaises(ValueError):
                PuzzleSpec.from_boards(start, target)

    def test_unknown_piece_bounds_and_incomplete_solution(self):
        spec = PuzzleSpec.from_boards(["A--"], ["--A"])
        for move in (Move("X", "right", 1), Move("A", "left", 1), Move("A", "right", 3)):
            with self.assertRaises(ValueError):
                replay(spec, [move])
        with self.assertRaisesRegex(ValueError, "final board"):
            replay(spec, [Move("A", "right", 1)])
        self.assertFalse(replay(spec, [], require_solved=False).solved)

    def test_identity_and_fingerprint(self):
        spec = PuzzleSpec.from_boards(["A--"], ["A--"])
        self.assertEqual(replay(spec, []).move_count, 0)
        self.assertEqual(spec.lower_bound, 0)
        self.assertEqual(
            spec.fingerprint, PuzzleSpec.from_boards(["A--"], ["A--"], "renamed").fingerprint
        )

    def test_default_lower_bound(self):
        spec = PuzzleSpec.load()
        self.assertEqual(
            (spec.height, spec.width, len(spec.pieces), spec.lower_bound), (20, 18, 16, 30)
        )
