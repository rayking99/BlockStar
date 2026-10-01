import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from blockstar import Move, PuzzleSpec
from blockstar.artifacts import audit_dataset, load_solution, solution_document, write_json
from blockstar.cli import main


class ArtifactTests(unittest.TestCase):
    def test_certificate_rechecks_score_and_identity(self):
        spec = PuzzleSpec.from_boards(["A--"], ["--A"])
        document = solution_document(spec, (Move("A", "right", 2),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solution.json"
            write_json(path, document)
            self.assertEqual(len(load_solution(path, spec)), 1)
            for key, value in (("score", 0), ("puzzle_sha256", "wrong"), ("unit_distance", 200)):
                write_json(path, {**document, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_solution(path, spec)

    def test_dataset_audit_preserves_failure_denominator(self):
        spec = PuzzleSpec.from_boards(["A--"], ["--A"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text(
                "[('A', 'right', 2)]\n[('A', 'right', 1)]\n__import__('os').getcwd()\n[('A', 'right', 2)]\n"
            )
            report, episodes = audit_dataset(spec, path)
        self.assertEqual(
            (report["records"], report["valid"], report["invalid"], report["duplicates"]),
            (4, 2, 2, 1),
        )
        self.assertEqual(len(episodes), 2)

    def test_cli_end_to_end_and_failure_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            puzzle = directory / "puzzle.json"
            output, report = directory / "best.json", directory / "report.json"
            write_json(puzzle, {"start": ["A--"], "target": ["--A"]})
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "--puzzle",
                            str(puzzle),
                            "discover",
                            "--trials",
                            "2",
                            "--quiet",
                            "--output",
                            str(output),
                            "--report",
                            str(report),
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["--puzzle", str(puzzle), "replay", str(output)]), 0)
                data = json.loads(output.read_text())
                data["moves"][0][2] = 99
                write_json(output, data)
                self.assertEqual(main(["--puzzle", str(puzzle), "replay", str(output)]), 2)
                write_json(puzzle, {"start": ["AB"], "target": ["BA"]})
                self.assertEqual(
                    main(
                        [
                            "--puzzle",
                            str(puzzle),
                            "discover",
                            "--trials",
                            "2",
                            "--quiet",
                            "--output",
                            str(directory / "unsolved.json"),
                            "--report",
                            str(report),
                        ]
                    ),
                    1,
                )
            self.assertFalse((directory / "unsolved.json").exists())

    def test_imports_do_not_train_run_or_write(self):
        root = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import BlockStar, AgentFormer, blockstar; import sys; assert 'torch' not in sys.modules",
                ],
                cwd=directory,
                env={**os.environ, "PYTHONPATH": root},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_original_replay_facade_is_transactional(self):
        from BlockStar import Puzzle

        puzzle = Puzzle(["A--"], ["--A"], ["A"])
        with self.assertRaises(ValueError):
            puzzle.apply_move("A", "left", 1)
        self.assertEqual(puzzle.move_count, 0)
        puzzle.apply_move("A", "right", 2)
        self.assertTrue(puzzle.solved)
