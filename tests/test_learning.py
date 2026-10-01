import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from blockstar import Move, PuzzleSpec, replay
from blockstar.learning import ActionSpace, ModelConfig, build_model, sample, split_episodes, train


class PolicyDataTests(unittest.TestCase):
    def test_legal_mask_matches_independent_referee(self):
        spec = PuzzleSpec.from_boards(["A-B-", "----", "----"], ["A-B-", "----", "----"])
        actions = ActionSpace(spec)
        expected = set()
        for i, move in enumerate(actions.moves):
            try:
                replay(spec, [move], require_solved=False)
                expected.add(i)
            except ValueError:
                pass
        self.assertEqual(set(actions.legal_indices(spec.initial)), expected)

    def test_episode_split_is_deterministic_disjoint_and_deduplicated(self):
        episodes = [(Move("A", "right", n),) for n in range(1, 10)]
        training, validation = split_episodes(episodes * 3, 7)
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(len(training) + len(validation), 9)
        self.assertEqual((training, validation), split_episodes(episodes, 7))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_attention_backward_and_deterministic_evaluation(self):
        import torch

        spec = PuzzleSpec.from_boards(["A--B", "----"], ["---A", "B---"])
        model = build_model(spec, ModelConfig(width=16, heads=2, layers=1, dropout=0.5))
        inputs = torch.tensor([spec.initial])
        output = model(inputs)
        self.assertEqual(output.shape, (1, len(ActionSpace(spec).moves)))
        output.sum().backward()
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )
        model.eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(inputs), model(inputs)))

    def test_train_checkpoint_and_legally_masked_sampling(self):
        spec = PuzzleSpec.from_boards(["A--B", "----"], ["---A", "B---"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.txt"
            dataset.write_text(
                "[('B','down',1),('B','left',3),('A','right',3)]\n"
                "[('A','right',2),('B','down',1),('A','left',1),('B','left',3),('A','right',2)]\n"
            )
            args = SimpleNamespace(
                steps=2,
                batch_size=2,
                learning_rate=0.001,
                seed=7,
                threads=1,
                device="cpu",
                dataset=dataset,
                width=16,
                heads=2,
                layers=1,
                checkpoint=root / "model.pt",
                max_moves=10,
                temperature=0.5,
                output=root / "rollout.json",
            )
            report = train(spec, args)
            self.assertEqual(report["train_episodes"], 1)
            self.assertEqual(report["validation_episodes"], 1)
            document = sample(spec, args)
            result = replay(spec, [Move(*m) for m in document["moves"]], require_solved=False)
            self.assertEqual(document["solved"], result.solved)
