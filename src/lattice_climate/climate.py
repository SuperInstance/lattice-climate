"""Map real climate data to lattice models."""

from __future__ import annotations

import numpy as np


class ClimateMapper:
    """Map real climate data to lattice models."""

    @staticmethod
    def temperature_to_spins(temps: np.ndarray, threshold: float) -> np.ndarray:
        """Binary spin mapping: above threshold → +1, below → -1.

        Parameters
        ----------
        temps : np.ndarray
            Temperature values.
        threshold : float
            Cutoff temperature.

        Returns
        -------
        np.ndarray
            Array of ±1.
        """
        return np.where(temps >= threshold, 1, -1).astype(int)

    @staticmethod
    def precipitation_to_potts(precip: np.ndarray, quantiles: list = None) -> np.ndarray:
        """Map precipitation values to q-state Potts model.

        Parameters
        ----------
        precip : np.ndarray
            Precipitation values.
        quantiles : list of float, optional
            Quantile boundaries (len q-1). Defaults to quartiles [0.25, 0.5, 0.75].

        Returns
        -------
        np.ndarray
            Integer array with values in {0, 1, ..., q-1}.
        """
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]

        boundaries = np.quantile(precip.ravel(), quantiles)
        result = np.zeros_like(precip, dtype=int)
        for k, b in enumerate(boundaries):
            result[precip >= b] = k + 1
        return result

    @staticmethod
    def spatial_correlation(spins: np.ndarray) -> float:
        """Compute Moran's I for spatial autocorrelation.

        I = (N / W) * Σ_{i≠j} w_{ij} (x_i - x̄)(x_j - x̄) / Σ_i (x_i - x̄)²

        Uses queen contiguity (8 neighbors) with uniform weights.

        Parameters
        ----------
        spins : np.ndarray
            2D array of values.

        Returns
        -------
        float
            Moran's I statistic.
        """
        rows, cols = spins.shape
        x = spins.ravel().astype(float)
        N = len(x)
        x_mean = np.mean(x)
        dx = x - x_mean

        # Build weight matrix for queen contiguity on 2D grid
        W = 0.0
        numerator = 0.0
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            nidx = ni * cols + nj
                            W += 1.0
                            numerator += dx[idx] * dx[nidx]

        denominator = np.sum(dx ** 2)
        if denominator == 0 or W == 0:
            return 0.0
        return (N / W) * (numerator / denominator)
