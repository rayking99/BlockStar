"""Render a verified solution as a terminal-style GIF (optional dependency: Pillow).

Usage: python -m blockstar.render_gif benchmarks/champion.json --output docs/champion.gif
Install Pillow in the rendering environment; the planner itself needs no dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import load_solution
from .model import DIRECTIONS, PuzzleSpec, replay


def render_gif(spec: PuzzleSpec, solution: Path, output: Path, font_path: Path | None = None):
    from PIL import Image, ImageDraw, ImageFont

    moves = load_solution(solution, spec)
    verified = replay(spec, moves)
    candidates = (
        [font_path]
        if font_path
        else [
            Path("/System/Library/Fonts/Menlo.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            Path("C:/Windows/Fonts/consola.ttf"),
        ]
    )
    chosen = next((path for path in candidates if path.is_file()), None)

    def font(size):
        return (
            ImageFont.truetype(str(chosen), size) if chosen else ImageFont.load_default(size=size)
        )

    compact = max(spec.width, spec.height) > 24
    title_font, board_font, status_font, small_font = (
        font(30),
        font(22 if compact else 26),
        font(18),
        font(14),
    )
    cell_width, cell_height = (20, 25) if compact else (24, 30)
    width = max(640, spec.width * cell_width + 96)
    height = spec.height * cell_height + 240
    left = (width - spec.width * cell_width) // 2
    top = 146
    colors = {
        "background": (12, 14, 17),
        "text": (231, 233, 237),
        "muted": (139, 146, 157),
        "empty": (68, 74, 84),
        "active": (119, 220, 193),
        "obstacle": (237, 181, 105),
        "line": (40, 45, 53),
    }
    board = [list(row) for row in spec.start]
    frames, durations = [], []

    def frame(number, action, active=None, finished=False):
        image = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(image)
        draw.text((40, 25), "BLOCKSTAR", font=title_font, fill=colors["text"])
        draw.text(
            (40, 67),
            f"VERIFIED SOLUTION  /  {len(moves)} MOVES",
            font=small_font,
            fill=colors["muted"],
        )
        draw.line((40, 98, width - 40, 98), fill=colors["line"])
        draw.text(
            (40, 111),
            f"MOVE {number:02d} / {len(moves):02d}",
            font=status_font,
            fill=colors["active"],
        )
        draw.text((width - 40, 111), action, anchor="ra", font=status_font, fill=colors["text"])
        for row, cells in enumerate(board):
            for col, value in enumerate(cells):
                color = (
                    colors["empty"]
                    if value == "-"
                    else colors["obstacle"]
                    if value == "#"
                    else colors["active"]
                    if value == active
                    else colors["text"]
                )
                draw.text(
                    (left + col * cell_width + cell_width // 2, top + row * cell_height),
                    value,
                    anchor="ma",
                    font=board_font,
                    fill=color,
                )
        draw.line((40, height - 65, width - 40, height - 65), fill=colors["line"])
        footer = "TARGET MATCHED" if finished else "One straight slide = one move"
        draw.text(
            (40, height - 46),
            footer,
            font=small_font,
            fill=colors["active"] if finished else colors["muted"],
        )
        if spec.fixed_cells:
            draw.text(
                (width - 40, height - 46),
                "# = IMMOVABLE",
                anchor="ra",
                font=small_font,
                fill=colors["obstacle"],
            )
        return image

    # Quantize immediately, so larger boards don't retain hundreds of full RGB frames.
    palette = frame(
        0, "Starting arrangement", active=moves[0].piece if moves else None, finished=True
    ).quantize(colors=128)

    def indexed(image):
        return image.quantize(palette=palette, dither=Image.Dither.NONE)

    frames.append(indexed(frame(0, "Starting arrangement")))
    durations.append(900)
    for number, move in enumerate(moves, 1):
        dr, dc = DIRECTIONS[move.direction]
        for step in range(1, move.distance + 1):
            cells = {
                (r, c)
                for r, row in enumerate(board)
                for c, value in enumerate(row)
                if value == move.piece
            }
            translated = {(r + dr, c + dc) for r, c in cells}
            for r, c in translated:
                if not (0 <= r < spec.height and 0 <= c < spec.width):
                    raise ValueError("Animated frame leaves board")
                if board[r][c] not in ("-", move.piece):
                    raise ValueError("Animated frame has a collision")
            for r, c in cells:
                board[r][c] = "-"
            for r, c in translated:
                board[r][c] = move.piece
            finished = number == len(moves) and step == move.distance
            frames.append(
                indexed(
                    frame(
                        number,
                        f"{move.piece} {move.direction} {move.distance}",
                        active=move.piece,
                        finished=finished,
                    )
                )
            )
            durations.append(2200 if finished else 200 if step == move.distance else 50)
    if tuple(map(tuple, board)) != verified.board:
        raise ValueError("Animated final board differs from independently verified target")
    if not moves:
        frames[0] = indexed(frame(0, "Already solved", finished=True))
        durations[0] = 2200

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    with Image.open(output) as gif:
        if gif.n_frames != len(frames):
            raise ValueError("Encoded GIF frame count differs from rendered frames")
        gif.seek(gif.n_frames - 1)
        gif.load()
    return {
        "path": str(output),
        "moves": len(moves),
        "frames": len(frames),
        "size": [width, height],
        "duration_seconds": sum(durations) / 1000,
        "unit_distance": verified.unit_distance,
        "bytes": output.stat().st_size,
    }


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/champion.gif"))
    parser.add_argument("--puzzle", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.solution.resolve():
        parser.error("Output must not overwrite the solution document")
    print(
        json.dumps(
            render_gif(PuzzleSpec.load(args.puzzle), args.solution, args.output, args.font),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
