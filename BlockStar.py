"""Import-safe compatibility facade. Run with the same CLI as `python -m blockstar`."""

from blockstar import Move, PuzzleSpec, replay
from blockstar.cli import main

_DEFAULT = PuzzleSpec.load()
start_state = [list(row) for row in _DEFAULT.start]
target_state = [list(row) for row in _DEFAULT.target]
PIECES = [piece.name for piece in _DEFAULT.pieces]


class Puzzle:
    """Compatibility for the original README's move-replay API.

    Invalid moves now raise ValueError, instead of silently doing nothing.
    Experimental heuristic internals have been replaced by blockstar.search.
    """

    def __init__(self, state, target_state, pieces=None):
        self.spec = PuzzleSpec.from_boards(state, target_state)
        labels = {piece.name for piece in self.spec.pieces}
        if pieces is not None and (set(pieces) != labels or len(pieces) != len(labels)):
            raise ValueError("pieces must name each board piece exactly once")
        self.state = [list(row) for row in self.spec.start]
        self.target_state = [list(row) for row in self.spec.target]
        self.pieces = [piece.name for piece in self.spec.pieces]
        self.move_register = []
        self.move_count = 0

    def apply_move(self, piece, direction, distance):
        move = Move(piece, direction, distance)
        current = PuzzleSpec.from_boards(self.state, self.target_state)
        result = replay(current, [move], require_solved=False)
        self.state = [list(row) for row in result.board]
        self.move_register.append((piece, direction, distance))
        self.move_count += 1
        return self.state

    @property
    def solved(self):
        return self.state == self.target_state


if __name__ == "__main__":
    raise SystemExit(main())
