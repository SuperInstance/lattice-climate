"""Detect phase transitions: critical temperature, Binder cumulant, finite-size scaling."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict


class PhaseDetector:
    """Detect phase transitions in lattice models."""

    @staticmethod
    def critical_temperature(model, T_range: np.ndarray, n_samples: int = 1000) -> float:
        """Find Tc by locating the peak in specific heat.

        Parameters
        ----------
        model : IsingModel or PottsModel
            The model to analyze.
        T_range : np.ndarray
            Temperature values to sweep.
        n_samples : int
            Number of MC steps per temperature.

        Returns
        -------
        float
            Estimated critical temperature.
        """
        from .ising import IsingModel
        from .monte_carlo import MonteCarlo

        lattice = model.lattice
        cv_values = []

        for T in T_range:
            # Start from all-up for better equilibration
            if isinstance(model, IsingModel):
                spins = np.ones(lattice.shape, dtype=int)
                model_T = IsingModel(lattice, J=model.J, T=T, h=model.h)
            else:
                spins = np.zeros(lattice.shape, dtype=int)
                model_T = model.__class__(lattice, q=model.q, J=model.J, T=T)

            _, energies, _ = MonteCarlo.metropolis(model_T, spins, T, n_steps=n_samples)

            # Use second half for thermal average
            half = len(energies) // 2
            E = np.array(energies[half:], dtype=float)
            N = lattice.size
            cv = float(np.var(E) / (N * T ** 2))
            cv_values.append(cv)

        cv_values = np.array(cv_values)
        peak_idx = int(np.argmax(cv_values))
        return float(T_range[peak_idx])

    @staticmethod
    def binder_cumulant(magnetizations: list) -> float:
        """Compute Binder cumulant U_L = 1 - ⟨m⁴⟩ / (3 ⟨m²⟩²).

        Parameters
        ----------
        magnetizations : list of float
            Magnetization samples.

        Returns
        -------
        float
            Binder cumulant.
        """
        m = np.array(magnetizations, dtype=float)
        m2 = np.mean(m ** 2)
        m4 = np.mean(m ** 4)
        if m2 == 0:
            return 0.0
        return float(1.0 - m4 / (3.0 * m2 ** 2))

    @staticmethod
    def finite_size_scaling(lattice_sizes: list, T_range: np.ndarray, n_samples: int = 500) -> dict:
        """Extract critical exponents from finite-size scaling.

        Runs Ising model on different lattice sizes and computes
        Binder cumulant crossings.

        Parameters
        ----------
        lattice_sizes : list of int
            Lattice side lengths to use.
        T_range : np.ndarray
            Temperature range to sweep.
        n_samples : int
            MC steps per temperature.

        Returns
        -------
        dict with keys 'Tc_estimates', 'binder_curves', 'sizes'
        """
        from .lattice import Lattice
        from .ising import IsingModel
        from .monte_carlo import MonteCarlo

        binder_curves = {}
        Tc_estimates = {}

        for L in lattice_sizes:
            lat = Lattice(L, L, boundary="periodic")
            binders = []
            for T in T_range:
                model = IsingModel(lat, J=1.0, T=T)
                spins = np.random.choice([-1, 1], size=(L, L))
                _, _, mags = MonteCarlo.metropolis(model, spins, T, n_steps=n_samples)
                half = len(mags) // 2
                binders.append(PhaseDetector.binder_cumulant(mags[half:]))
            binder_curves[L] = binders
            # Estimate Tc from minimum of Binder cumulant derivative (approximate)
            binders_arr = np.array(binders)
            # Use peak of |dU/dT| as proxy
            dU = np.abs(np.gradient(binders_arr, T_range))
            Tc_estimates[L] = float(T_range[np.argmax(dU)])

        return {
            "sizes": lattice_sizes,
            "Tc_estimates": Tc_estimates,
            "binder_curves": binder_curves,
        }
