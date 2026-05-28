"""Ising model for binary climate states."""

from __future__ import annotations

import numpy as np
from typing import List


class IsingModel:
    """Ising model on a lattice.

    H = -J Σ_{<i,j>} s_i s_j  - h Σ_i s_i

    Parameters
    ----------
    lattice : Lattice
        The lattice to use.
    J : float
        Coupling constant (>0 ferromagnetic).
    T : float
        Temperature (in units where k_B = 1).
    h : float
        External field.
    """

    def __init__(self, lattice, J: float = 1.0, T: float = 2.269, h: float = 0.0):
        self.lattice = lattice
        self.J = J
        self.T = T
        self.h = h

    def energy(self, spins: np.ndarray) -> float:
        """Compute total energy of a spin configuration.

        Parameters
        ----------
        spins : np.ndarray
            2D array of ±1 values with shape matching lattice.shape.

        Returns
        -------
        float
            Total energy.
        """
        rows, cols = self.lattice.shape
        E = 0.0
        for i in range(rows):
            for j in range(cols):
                s = spins[i, j]
                # Only check right and down neighbors to avoid double-counting
                for di, dj in [(0, 1), (1, 0)]:
                    ni = (i + di) % rows if self.lattice.boundary == "periodic" else i + di
                    nj = (j + dj) % cols if self.lattice.boundary == "periodic" else j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        E -= self.J * s * spins[ni, nj]
                E -= self.h * s
        return E

    def magnetization(self, spins: np.ndarray) -> float:
        """Compute average magnetization per site.

        Returns
        -------
        float
            ⟨s⟩ = (1/N) Σ s_i
        """
        return float(np.mean(spins))

    def specific_heat(self, energies: List[float]) -> float:
        """Compute specific heat from a list of energy samples.

        C = (⟨E²⟩ - ⟨E⟩²) / (N T²)
        """
        E = np.array(energies, dtype=float)
        N = self.lattice.size
        return float(np.var(E) / (N * self.T ** 2))

    def susceptibility(self, mags: List[float]) -> float:
        """Compute magnetic susceptibility from magnetization samples.

        χ = N (⟨m²⟩ - ⟨m⟩²) / T
        """
        M = np.array(mags, dtype=float)
        N = self.lattice.size
        return float(N * np.var(M) / self.T)

    def delta_energy(self, spins: np.ndarray, i: int, j: int) -> float:
        """Energy change from flipping spin at (i, j)."""
        s = spins[i, j]
        neighbor_sum = 0
        for ni, nj in self.lattice.neighbors(i, j):
            neighbor_sum += spins[ni, nj]
        dE = 2.0 * self.J * s * neighbor_sum + 2.0 * self.h * s
        return dE
