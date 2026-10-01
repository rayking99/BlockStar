"""Optional state-conditioned Transformer imitation; never required by discovery.

Unlike an open-loop character model, this model sees every piece's current and
target coordinates. Inference masks every illegal slide. Legality does not imply
successful planning: sampled runs are independently replayed and labelled honestly.
PyTorch is loaded only when training or sampling is explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .artifacts import audit_dataset, write_json
from .engine import Engine
from .model import DIRECTIONS, Move, PuzzleSpec, replay


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'Training requires the optional dependency: pip install -e ".[train]"'
        ) from exc
    return torch


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if self.width < 1 or self.heads < 1 or self.layers < 1 or self.width % self.heads:
            raise ValueError(
                "width, heads and layers must be positive; width must be divisible by heads"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class ActionSpace:
    def __init__(self, spec: PuzzleSpec):
        self.engine = Engine(spec)
        self.moves = tuple(
            Move(p.name, direction, distance)
            for p in spec.pieces
            for direction in DIRECTIONS
            for distance in range(1, spec.height if direction in ("up", "down") else spec.width)
        )
        self.index = {move: i for i, move in enumerate(self.moves)}
        self.pieces = {p.name: i for i, p in enumerate(spec.pieces)}
        self.per_piece = 2 * (spec.height + spec.width - 2)

    def advance(self, state: tuple[int, ...], move: Move) -> tuple[int, ...]:
        dr, dc = DIRECTIONS[move.direction]
        result = list(state)
        result[self.pieces[move.piece]] += (dr * self.engine.spec.width + dc) * move.distance
        return tuple(result)

    def legal_indices(self, state: tuple[int, ...]) -> list[int]:
        result = []
        occupied = self.engine.occupied(state)
        for i, start in enumerate(state):
            blocked = occupied ^ self.engine.masks[i][start]
            for ray in self.engine.rays[i][start]:
                for pos in ray:
                    if self.engine.masks[i][pos] & blocked:
                        break
                    move = self.engine.moves(i, start, (pos,))[0]
                    result.append(self.index[move])
        return result


def build_model(spec: PuzzleSpec, config: ModelConfig | None = None):
    config = config or ModelConfig()
    torch = _torch()
    nn, functional = torch.nn, torch.nn.functional
    if not spec.pieces or spec.height + spec.width <= 2:
        raise ValueError("Training needs at least one piece and room for a slide")

    class AttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.norm2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
            self.qkv = nn.Linear(config.width, 3 * config.width)
            self.projection = nn.Linear(config.width, config.width)
            self.feedforward = nn.Sequential(
                nn.Linear(config.width, 4 * config.width),
                nn.GELU(),
                nn.Linear(4 * config.width, config.width),
                nn.Dropout(config.dropout),
            )

        def forward(self, x):
            batch, length, width = x.shape
            qkv = self.qkv(self.norm1(x)).reshape(
                batch, length, 3, config.heads, width // config.heads
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attention = functional.scaled_dot_product_attention(
                q, k, v, dropout_p=config.dropout if self.training else 0.0, is_causal=False
            )
            x = x + self.projection(attention.transpose(1, 2).reshape(batch, length, width))
            return x + self.feedforward(self.norm2(x))

    class StatePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = nn.Embedding(len(spec.pieces), config.width)
            self.row = nn.Embedding(spec.height, config.width)
            self.col = nn.Embedding(spec.width, config.width)
            self.goal_row = nn.Embedding(spec.height, config.width)
            self.goal_col = nn.Embedding(spec.width, config.width)
            self.blocks = nn.Sequential(*(AttentionBlock() for _ in range(config.layers)))
            self.norm = nn.LayerNorm(config.width)
            self.head = nn.Linear(config.width, 2 * (spec.height + spec.width - 2))
            self.register_buffer("goal", torch.tensor(spec.goal, dtype=torch.long))
            self.register_buffer("piece_ids", torch.arange(len(spec.pieces)))

        def forward(self, positions):
            x = (
                self.identity(self.piece_ids)
                + self.row(positions // spec.width)
                + self.col(positions % spec.width)
                + self.goal_row(self.goal // spec.width)
                + self.goal_col(self.goal % spec.width)
            )
            return self.head(self.norm(self.blocks(x))).flatten(1)

    return StatePolicy()


def split_episodes(episodes: list[tuple[Move, ...]], seed: int):
    unique = sorted(
        set(episodes), key=lambda run: tuple((m.piece, m.direction, m.distance) for m in run)
    )
    if len(unique) < 2:
        raise ValueError("Training needs at least two distinct valid episodes for a held-out split")
    random.Random(seed).shuffle(unique)
    split = max(1, min(len(unique) - 1, int(0.9 * len(unique))))
    return unique[:split], unique[split:]


def _examples(spec: PuzzleSpec, episodes, actions: ActionSpace):
    torch = _torch()
    states, labels = [], []
    for moves in episodes:
        state = spec.initial
        for move in moves:
            states.append(state)
            labels.append(actions.index[move])
            state = actions.advance(state, move)
    if not states:
        raise ValueError("Training split has no moves")
    return torch.tensor(states, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def _device(requested: str):
    torch = _torch()
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(requested)


def train(spec: PuzzleSpec, args) -> dict:
    torch = _torch()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("steps, batch size and learning rate must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = _device(args.device)
    audit, episodes = audit_dataset(spec, args.dataset)
    train_episodes, validation_episodes = split_episodes(episodes, args.seed)
    actions = ActionSpace(spec)
    train_x, train_y = _examples(spec, train_episodes, actions)
    val_x, val_y = _examples(spec, validation_episodes, actions)
    config = ModelConfig(width=args.width, heads=args.heads, layers=args.layers)
    model = build_model(spec, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.steps):
        indices = torch.randint(len(train_x), (args.batch_size,), generator=generator)
        logits = model(train_x[indices].to(device))
        loss = torch.nn.functional.cross_entropy(logits, train_y[indices].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Training produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    losses, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for offset in range(0, len(val_x), args.batch_size):
            x, y = (
                val_x[offset : offset + args.batch_size].to(device),
                val_y[offset : offset + args.batch_size].to(device),
            )
            logits = model(x)
            losses += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(-1) == y).sum().item()
            count += len(y)
    report = {
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        "torch": str(torch.__version__),
        "model": asdict(config),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "train_examples": len(train_x),
        "validation_examples": len(val_x),
        "final_batch_train_loss": loss.item(),
        "validation_loss": losses / count,
        "validation_top1": correct / count,
        "dataset_audit": audit,
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1,
        "puzzle_sha256": spec.fingerprint,
        "model_config": asdict(config),
        "state_dict": model.cpu().state_dict(),
        "report": report,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    write_json(args.checkpoint.with_suffix(".json"), report)
    return report


def sample(spec: PuzzleSpec, args) -> dict:
    torch = _torch()
    if args.max_moves < 1 or args.temperature <= 0:
        raise ValueError("max moves and temperature must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("puzzle_sha256") != spec.fingerprint:
        raise ValueError("Checkpoint schema or puzzle does not match")
    model = build_model(spec, ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    actions = ActionSpace(spec)
    state, moves, visited = spec.initial, [], {spec.initial}
    with torch.inference_mode():
        for _ in range(args.max_moves):
            if state == spec.goal:
                break
            legal = [
                i
                for i in actions.legal_indices(state)
                if actions.advance(state, actions.moves[i]) not in visited
            ]
            if not legal:
                break
            logits = model(torch.tensor([state], dtype=torch.long, device=device))[0]
            probabilities = (logits[legal] / args.temperature).softmax(-1)
            index = legal[torch.multinomial(probabilities, 1).item()]
            move = actions.moves[index]
            moves.append(move)
            state = actions.advance(state, move)
            visited.add(state)
    result = replay(spec, moves, require_solved=False)
    document = {
        "schema_version": 1,
        "puzzle_sha256": spec.fingerprint,
        "solved": result.solved,
        "score": result.move_count,
        "unit_distance": result.unit_distance,
        "moves": [m.as_list() for m in moves],
        "seed": args.seed,
        "note": "Legal policy rollout; a solution only if solved is true.",
    }
    write_json(args.output, document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--dataset", type=Path, default=Path("results.txt"))
    fit.add_argument("--steps", type=int, default=1000)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--width", type=int, default=128)
    fit.add_argument("--heads", type=int, default=4)
    fit.add_argument("--layers", type=int, default=2)
    fit.add_argument("--learning-rate", type=float, default=3e-4)
    generate = sub.add_parser("sample")
    generate.add_argument("--max-moves", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=0.5)
    generate.add_argument("--output", type=Path, default=Path("runs/policy-rollout.json"))
    for command in (fit, generate):
        command.add_argument("--checkpoint", type=Path, default=Path("runs/agentformer.pt"))
        command.add_argument("--seed", type=int, default=1337)
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        spec = PuzzleSpec.load(args.puzzle)
        result = train(spec, args) if args.command == "train" else sample(spec, args)
        print(json.dumps(result, indent=2))
        return 0 if args.command == "train" or result["solved"] else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
