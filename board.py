"""Board graph for knight moves.

This module builds an adjacency list for a rows x cols board where each
vertex is a square and edges represent legal knight moves.

Squares are represented internally as 0-based indices: index = r * cols + c.
Helpers are provided to convert between (r, c) and index.
"""
from typing import Dict, List, Tuple


class BoardGraph:
    """Graph of a rectangular board with knight moves.

    Attributes:
        rows (int): number of rows
        cols (int): number of columns
        size (int): total number of squares
        adj (Dict[int, List[int]]): adjacency list mapping index -> neighbor indices
    """

    KNIGHT_DELTAS: List[Tuple[int, int]] = [
        (2, 1),
        (1, 2),
        (-1, 2),
        (-2, 1),
        (-2, -1),
        (-1, -2),
        (1, -2),
        (2, -1),
    ]

    def __init__(self, rows: int = 8, cols: int = 8) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")
        self.rows = int(rows)
        self.cols = int(cols)
        self.size = self.rows * self.cols
        self.adj: Dict[int, List[int]] = self._build_adj()

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def rc_to_index(self, r: int, c: int) -> int:
        """Convert (row, col) to index."""
        if not self._in_bounds(r, c):
            raise IndexError(f"square out of bounds: ({r}, {c})")
        return r * self.cols + c

    def index_to_rc(self, idx: int) -> Tuple[int, int]:
        """Convert index to (row, col)."""
        if idx < 0 or idx >= self.size:
            raise IndexError(f"index out of bounds: {idx}")
        return divmod(idx, self.cols)

    def _build_adj(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                idx = self.rc_to_index(r, c)
                neighbors: List[int] = []
                for dr, dc in self.KNIGHT_DELTAS:
                    nr, nc = r + dr, c + dc
                    if self._in_bounds(nr, nc):
                        neighbors.append(self.rc_to_index(nr, nc))
                adj[idx] = neighbors
        return adj

    def neighbors(self, pos) -> List[int]:
        """Return neighbor indices for a given position.

        Accepts either an index (int) or an (r, c) tuple.
        """
        if isinstance(pos, tuple):
            r, c = pos
            idx = self.rc_to_index(r, c)
        else:
            idx = int(pos)
            if idx < 0 or idx >= self.size:
                raise IndexError(f"index out of bounds: {idx}")
        return list(self.adj[idx])

    def neighbors_rc(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return neighbors as list of (r, c) tuples for given coordinates."""
        idx_neighbors = self.neighbors((r, c))
        return [self.index_to_rc(i) for i in idx_neighbors]

    def __repr__(self) -> str:
        return f"BoardGraph({self.rows}x{self.cols})"
