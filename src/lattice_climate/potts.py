"""Potts model for multi-state climate zones."""

from __future__ import annotations

import numpy as np
from typing import List


class PottsModel:
    """q-state Potts model.

    H = -J Σ_{<i,j>} δ(s_i, s_j)

    Parameters
    ----------
    lattice : Lattice
        The lattice.
    q : int
        Number of states (≥ 2).
    J : float
        Coupling constant.
    T : float
        Temperature (k_B = 1).
    """

    def __init__(self, lattice, q: int = 4, J: float = 1.0, T: float = 1.0):
        if q < 2:
            raise ValueError("q must be ≥ 2")
        self.lattice = lattice
        self.q = q
        self.J = J
        self.T = T

    def energy(self, states: np.ndarray) -> float:
        """Compute total Potts energy.

        Parameters
        ----------
        states : np.ndarray
            Integer array with values in {0, 1, ..., q-1}.

        Returns
        -------
        float
            Total energy.
        """
        rows, cols = self.lattice.shape
        E = 0.0
        for i in range(rows):
            for j in range(cols):
                s = states[i, j]
                for di, dj in [(0, 1), (1, 0)]:
                    ni = (i + di) % rows if self.lattice.boundary == "periodic" else i + di
                    nj = (j + dj) % cols if self.lattice.boundary == "periodic" else j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if s == states[ni, nj]:
                            E -= self.J
        return E

    def order_parameter(self, states: np.ndarray) -> float:
        """Compute Potts order parameter.

        m = (q * max_count - N) / ((q - 1) * N)

        Returns 1 when all sites share the same state, ~0 for disordered.
        """
        N = self.lattice.size
        counts = np.bincount(states.ravel(), minlength=self.q)
        max_count = int(np.max(counts))
        return (self.q * max_count - N) / ((self.q - 1) * N)

    def delta_energy(self, states: np.ndarray, i: int, j: int, new_s: int) -> float:
        """Energy change from changing state at (i, j) to new_s."""
        old_s = states[i, j]
        if old_s == new_s:
            return 0.0
        dE = 0.0
        for ni, nj in self.lattice.neighbors(i, j):
            ns = states[ni, nj]
            if old_s == ns:
                dE += self.J
            if new_s == ns:
                dE -= self.J
        return dE
