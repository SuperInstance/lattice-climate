"""Monte Carlo samplers: Metropolis-Hastings, Wolff cluster, Swendsen-Wang."""

from __future__ import annotations

import numpy as np
from typing import Tuple


class MonteCarlo:
    """Collection of Monte Carlo sampling algorithms."""

    @staticmethod
    def metropolis(model, spins: np.ndarray, T: float, n_steps: int) -> Tuple[np.ndarray, list, list]:
        """Metropolis-Hastings single-spin-flip Monte Carlo.

        Parameters
        ----------
        model : IsingModel or PottsModel
            The model to sample.
        spins : np.ndarray
            Initial spin/state configuration.
        T : float
            Temperature.
        n_steps : int
            Number of MC steps (each step = N attempted flips).

        Returns
        -------
        final_spins : np.ndarray
        energies : list of float
        magnetizations : list of float (or order params for Potts)
        """
        from .ising import IsingModel
        from .potts import PottsModel

        spins = spins.copy()
        rows, cols = spins.shape
        N = rows * cols
        is_ising = isinstance(model, IsingModel)
        energies = []
        mags = []
        beta = 1.0 / max(T, 1e-10)

        for step in range(n_steps):
            for _ in range(N):
                i = np.random.randint(0, rows)
                j = np.random.randint(0, cols)

                if is_ising:
                    dE = model.delta_energy(spins, i, j)
                    if dE <= 0 or np.random.random() < np.exp(-beta * dE):
                        spins[i, j] *= -1
                else:
                    new_s = np.random.randint(0, model.q)
                    if new_s != spins[i, j]:
                        dE = model.delta_energy(spins, i, j, new_s)
                        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
                            spins[i, j] = new_s

            energies.append(model.energy(spins))
            if is_ising:
                mags.append(model.magnetization(spins))
            else:
                mags.append(model.order_parameter(spins))

        return spins, energies, mags

    @staticmethod
    def wolff_cluster(model, spins: np.ndarray, T: float, n_steps: int) -> Tuple[np.ndarray, list, list]:
        """Wolff single-cluster algorithm (Ising only).

        Grows a cluster via bond activation probability p = 1 - exp(-2J/T).
        """
        from .ising import IsingModel

        if not isinstance(model, IsingModel):
            raise NotImplementedError("Wolff cluster only implemented for Ising model")

        spins = spins.copy()
        rows, cols = spins.shape
        beta = 1.0 / max(T, 1e-10)
        p_add = 1.0 - np.exp(-2.0 * model.J / max(T, 1e-10))
        energies = []
        mags = []

        for step in range(n_steps):
            # Pick random seed
            si, sj = np.random.randint(0, rows), np.random.randint(0, cols)
            seed_spin = spins[si, sj]
            cluster = {(si, sj)}
            stack = [(si, sj)]

            while stack:
                ci, cj = stack.pop()
                for ni, nj in model.lattice.neighbors(ci, cj):
                    if (ni, nj) not in cluster and spins[ni, nj] == seed_spin:
                        if np.random.random() < p_add:
                            cluster.add((ni, nj))
                            stack.append((ni, nj))

            # Flip entire cluster
            for ci, cj in cluster:
                spins[ci, cj] *= -1

            energies.append(model.energy(spins))
            mags.append(model.magnetization(spins))

        return spins, energies, mags

    @staticmethod
    def swendsen_wang(model, spins: np.ndarray, T: float, n_steps: int) -> Tuple[np.ndarray, list, list]:
        """Swendsen-Wang multi-cluster algorithm (Ising only).

        Activates bonds between like spins with probability p = 1 - exp(-2J/T),
        then flips each cluster independently with probability 1/2.
        """
        from .ising import IsingModel

        if not isinstance(model, IsingModel):
            raise NotImplementedError("Swendsen-Wang only implemented for Ising model")

        spins = spins.copy()
        rows, cols = spins.shape
        p_add = 1.0 - np.exp(-2.0 * model.J / max(T, 1e-10))
        energies = []
        mags = []

        for step in range(n_steps):
            # Build bond activation
            visited = set()
            cluster_id = np.full((rows, cols), -1, dtype=int)
            cluster_labels = {}
            current_label = 0

            for i in range(rows):
                for j in range(cols):
                    if cluster_id[i, j] == -1:
                        # BFS to label connected component through active bonds
                        stack = [(i, j)]
                        cluster_id[i, j] = current_label
                        members = [(i, j)]
                        while stack:
                            ci, cj = stack.pop()
                            for ni, nj in model.lattice.neighbors(ci, cj):
                                if cluster_id[ni, nj] == -1 and spins[ni, nj] == spins[ci, cj]:
                                    if np.random.random() < p_add:
                                        cluster_id[ni, nj] = current_label
                                        stack.append((ni, nj))
                                        members.append((ni, nj))
                        cluster_labels[current_label] = members
                        current_label += 1

            # Flip each cluster with prob 1/2
            flip = np.random.randint(0, 2, size=current_label).astype(bool)
            for label, members in cluster_labels.items():
                if flip[label]:
                    for ci, cj in members:
                        spins[ci, cj] *= -1

            energies.append(model.energy(spins))
            mags.append(model.magnetization(spins))

        return spins, energies, mags
