"""2D/3D lattice with configurable boundary conditions."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


class Lattice:
    """2D lattice with configurable boundary conditions.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    boundary : str
        Boundary condition: 'periodic', 'fixed', or 'open'.
    """

    def __init__(self, rows: int, cols: int, boundary: str = "periodic"):
        if rows < 1 or cols < 1:
            raise ValueError("Lattice dimensions must be positive")
        if boundary not in ("periodic", "fixed", "open"):
            raise ValueError(f"Unknown boundary condition: {boundary}")
        self._rows = rows
        self._cols = cols
        self._boundary = boundary

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols)."""
        return (self._rows, self._cols)

    @property
    def boundary(self) -> str:
        return self._boundary

    @property
    def size(self) -> int:
        return self._rows * self._cols

    def neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Return list of neighbor coordinates for site (i, j).

        For a 2D square lattice this returns up to 4 neighbors
        (up, down, left, right).
        """
        if not (0 <= i < self._rows and 0 <= j < self._cols):
            raise IndexError(f"Site ({i}, {j}) out of bounds for lattice {self.shape}")

        nbrs = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if self._boundary == "periodic":
                ni %= self._rows
                nj %= self._cols
                nbrs.append((ni, nj))
            elif self._boundary == "fixed":
                if 0 <= ni < self._rows and 0 <= nj < self._cols:
                    nbrs.append((ni, nj))
            else:  # open
                if 0 <= ni < self._rows and 0 <= nj < self._cols:
                    nbrs.append((ni, nj))
        return nbrs

    def neighbor_pairs(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return all unique nearest-neighbor pairs (for energy computation)."""
        seen = set()
        pairs = []
        for i in range(self._rows):
            for j in range(self._cols):
                for ni, nj in self.neighbors(i, j):
                    pair = tuple(sorted(((i, j), (ni, nj))))
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)
        return pairs
