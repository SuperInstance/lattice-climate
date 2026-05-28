"""Transfer matrix exact solutions for 1D/2D small systems."""

from __future__ import annotations

import numpy as np
from typing import Dict


class TransferMatrix:
    """Exact solutions via transfer matrix method."""

    @staticmethod
    def ising_1d(L: int, T: float, J: float = 1.0, h: float = 0.0) -> Dict[str, float]:
        """Exact free energy, magnetization for 1D Ising chain.

        Parameters
        ----------
        L : int
            Chain length.
        T : float
            Temperature.
        J : float
            Coupling constant.
        h : float
            External field.

        Returns
        -------
        dict with 'free_energy', 'magnetization', 'partition_function'
        """
        beta = 1.0 / max(T, 1e-10)

        # Transfer matrix
        T_mat = np.array([
            [np.exp(beta * (J + h)), np.exp(-beta * J)],
            [np.exp(-beta * J), np.exp(beta * (J - h))]
        ])

        eigenvalues = np.linalg.eigvalsh(T_mat)
        lambda_max = max(eigenvalues)
        lambda_min = min(eigenvalues)

        Z = lambda_max ** L + lambda_min ** L
        free_energy = -T * np.log(Z) / L

        if h == 0:
            magnetization = 0.0
        else:
            sh = np.sinh(beta * h)
            magnetization = sh / np.sqrt(sh ** 2 + np.exp(-4 * beta * J))

        return {
            "free_energy": float(free_energy),
            "magnetization": float(magnetization),
            "partition_function": float(Z),
        }

    @staticmethod
    def ising_2d_exact(L: int, T: float, J: float = 1.0) -> float:
        """Onsager exact specific heat for 2D Ising (thermodynamic limit).

        Uses the standard Onsager result:
        f = -kT ln(2 cosh(2βJ)) - kT/(2π) ∫₀^π ln{[1+√(1-κ²sin²θ)]/2} dθ
        where κ = 2 sinh(2βJ) / cosh²(2βJ)

        Returns specific heat per site computed via numerical differentiation.

        Parameters
        ----------
        L : int
            Lattice side length (result is thermodynamic limit).
        T : float
            Temperature.
        J : float
            Coupling constant.

        Returns
        -------
        float
            Specific heat per site.
        """
        f = TransferMatrix._onsager_free_energy(T, J)
        dT = T * 1e-4
        f_plus = TransferMatrix._onsager_free_energy(T + dT, J)
        f_minus = TransferMatrix._onsager_free_energy(T - dT, J)
        d2f = (f_plus - 2 * f + f_minus) / dT ** 2
        return float(-T * d2f)

    @staticmethod
    def _onsager_free_energy(T: float, J: float = 1.0) -> float:
        """Compute Onsager free energy per site."""
        beta = 1.0 / max(T, 1e-10)
        K = beta * J
        sinh2K = np.sinh(2 * K)
        cosh2K = np.cosh(2 * K)

        # κ = 2 sinh(2K) / cosh²(2K)
        kappa = 2 * sinh2K / cosh2K ** 2

        n_pts = 500
        theta = np.linspace(0, np.pi, n_pts + 1)[:-1]
        dtheta = np.pi / n_pts

        integral = 0.0
        for t in theta:
            val = 1 - kappa ** 2 * np.sin(t) ** 2
            if val < 0:
                val = 0
            integral += np.log((1 + np.sqrt(val)) / 2)
        integral *= dtheta / np.pi

        return -T * np.log(2 * cosh2K) - T * integral
