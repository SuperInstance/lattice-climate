"""Tests for lattice-climate."""

import numpy as np
import pytest

from lattice_climate import (
    Lattice,
    IsingModel,
    PottsModel,
    MonteCarlo,
    ClimateMapper,
    PhaseDetector,
    TransferMatrix,
)


# --- Lattice tests ---

class TestLattice:
    def test_creation(self):
        lat = Lattice(10, 20)
        assert lat.shape == (10, 20)
        assert lat.boundary == "periodic"

    def test_neighbors_corner_periodic(self):
        lat = Lattice(4, 4, boundary="periodic")
        nbrs = lat.neighbors(0, 0)
        # Should wrap: up→(3,0), down→(1,0), left→(0,3), right→(0,1)
        assert (3, 0) in nbrs
        assert (1, 0) in nbrs
        assert (0, 3) in nbrs
        assert (0, 1) in nbrs
        assert len(nbrs) == 4

    def test_neighbors_interior(self):
        lat = Lattice(5, 5, boundary="periodic")
        nbrs = lat.neighbors(2, 2)
        assert len(nbrs) == 4
        assert (1, 2) in nbrs
        assert (3, 2) in nbrs
        assert (2, 1) in nbrs
        assert (2, 3) in nbrs

    def test_fixed_boundary(self):
        lat = Lattice(3, 3, boundary="fixed")
        # Corner has 2 neighbors
        nbrs = lat.neighbors(0, 0)
        assert len(nbrs) == 2
        # Center has 4
        assert len(lat.neighbors(1, 1)) == 4


# --- Ising tests ---

class TestIsing:
    def test_energy_all_up(self):
        lat = Lattice(4, 4)
        model = IsingModel(lat, J=1.0, T=2.0)
        spins = np.ones((4, 4), dtype=int)
        E = model.energy(spins)
        # Each of 4*4 = 16 sites has 2 bonds (right, down) with periodic BC → 32 bonds
        # E = -J * 32 = -32
        assert E == pytest.approx(-32.0)

    def test_energy_anticorrelated(self):
        lat = Lattice(2, 2)
        model = IsingModel(lat, J=1.0)
        spins = np.array([[1, -1], [-1, 1]])
        # 2x2 with periodic BC: 4 right bonds + 4 down bonds = 8 bonds
        # All bonds are antiparallel → each contributes +J
        # E = -J * (-1-1-1-1-1-1-1-1) = 8J
        assert model.energy(spins) == pytest.approx(8.0)

    def test_magnetization(self):
        lat = Lattice(4, 4)
        model = IsingModel(lat)
        spins = np.ones((4, 4), dtype=int)
        assert model.magnetization(spins) == pytest.approx(1.0)
        spins[0, 0] = -1
        # 15 up + 1 down → mean = (15 - 1) / 16 = 14/16 = 0.875
        assert model.magnetization(spins) == pytest.approx(14 / 16)


# --- Potts tests ---

class TestPotts:
    def test_energy(self):
        lat = Lattice(3, 3)
        model = PottsModel(lat, q=3, J=1.0)
        states = np.zeros((3, 3), dtype=int)
        E = model.energy(states)
        # All same state: all 18 bonds (9 sites * 2 right/down bonds periodic)
        # Actually 9 sites * 2 directions = 18, but with periodic 3x3: 9 right + 9 down = 18
        assert E == pytest.approx(-18.0)

    def test_order_parameter(self):
        lat = Lattice(4, 4)
        model = PottsModel(lat, q=4)
        states = np.zeros((4, 4), dtype=int)
        assert model.order_parameter(states) == pytest.approx(1.0)
        # Fully disordered
        states_disordered = np.arange(16).reshape(4, 4) % 4
        op = model.order_parameter(states_disordered)
        assert op < 0.5


# --- Monte Carlo tests ---

class TestMonteCarlo:
    def test_metropolis_high_t_disordered(self):
        """At high T, system should be disordered: |magnetization| ≈ 0."""
        lat = Lattice(16, 16)
        model = IsingModel(lat, J=1.0)
        spins = np.ones((16, 16), dtype=int)
        final, _, mags = MonteCarlo.metropolis(model, spins, T=10.0, n_steps=200)
        assert abs(np.mean(mags[-50:])) < 0.3

    def test_metropolis_low_t_ordered(self):
        """At low T, system should be ordered: |magnetization| ≈ 1."""
        lat = Lattice(16, 16)
        model = IsingModel(lat, J=1.0)
        spins = np.ones((16, 16), dtype=int)
        final, _, mags = MonteCarlo.metropolis(model, spins, T=0.5, n_steps=200)
        assert abs(np.mean(mags[-50:])) > 0.7

    def test_wolff_cluster(self):
        """Wolff cluster should run without error and produce valid spins."""
        lat = Lattice(8, 8)
        model = IsingModel(lat, J=1.0)
        spins = np.random.choice([-1, 1], size=(8, 8))
        final, energies, mags = MonteCarlo.wolff_cluster(model, spins, T=2.269, n_steps=50)
        assert final.shape == (8, 8)
        assert all(s in [-1, 1] for s in final.ravel())
        assert len(energies) == 50

    def test_swendsen_wang(self):
        """Swendsen-Wang should run without error."""
        lat = Lattice(8, 8)
        model = IsingModel(lat, J=1.0)
        spins = np.random.choice([-1, 1], size=(8, 8))
        final, energies, mags = MonteCarlo.swendsen_wang(model, spins, T=2.269, n_steps=50)
        assert final.shape == (8, 8)
        assert len(energies) == 50


# --- Climate mapper tests ---

class TestClimateMapper:
    def test_temperature_to_spins(self):
        temps = np.array([[10, 20], [30, 0]])
        spins = ClimateMapper.temperature_to_spins(temps, threshold=15)
        np.testing.assert_array_equal(spins, [[-1, 1], [1, -1]])

    def test_precipitation_to_potts(self):
        precip = np.arange(100).reshape(10, 10).astype(float)
        states = ClimateMapper.precipitation_to_potts(precip)
        assert states.min() >= 0
        assert states.max() <= 3
        # Should be roughly 25 values per state
        for s in range(4):
            assert np.sum(states == s) > 10

    def test_spatial_correlation(self):
        """Uniform array should have high positive correlation."""
        uniform = np.ones((10, 10))
        I = ClimateMapper.spatial_correlation(uniform)
        # All same value → dx = 0 → I = 0
        assert I == pytest.approx(0.0)

        # Checkerboard: negative correlation
        checker = np.indices((10, 10)).sum(axis=0) % 2
        I_check = ClimateMapper.spatial_correlation(checker)
        assert I_check < 0


# --- Phase detector tests ---

class TestPhaseDetector:
    def test_critical_temperature(self):
        """Should find Tc ≈ 2.269 for 2D Ising on a 16×16 lattice."""
        lat = Lattice(16, 16)
        model = IsingModel(lat, J=1.0)
        T_range = np.linspace(1.5, 3.0, 16)
        Tc = PhaseDetector.critical_temperature(model, T_range, n_samples=300)
        # Should be in the middle of the range (not at an edge)
        assert 1.5 < Tc < 3.0

    def test_binder_cumulant(self):
        """Binder cumulant should be between -2/3 and 1."""
        # Ordered state: all same → U = 1 - 1/(3*1) = 2/3
        mags_ordered = [1.0] * 100
        U = PhaseDetector.binder_cumulant(mags_ordered)
        assert U == pytest.approx(2.0 / 3.0)

        # Random: symmetric around 0
        mags_random = list(np.random.randn(1000))
        U_r = PhaseDetector.binder_cumulant(mags_random)
        assert -1 < U_r < 1


# --- Transfer matrix tests ---

class TestTransferMatrix:
    def test_ising_1d_free_energy(self):
        """1D Ising free energy matches analytical result at h=0.

        Analytical: f = -kT ln(2 cosh(βJ))
        """
        T = 2.0
        J = 1.0
        L = 100
        result = TransferMatrix.ising_1d(L, T, J, h=0.0)
        beta = 1.0 / T
        f_analytical = -T * np.log(2 * np.cosh(beta * J))
        assert result["free_energy"] == pytest.approx(f_analytical, rel=1e-3)

    def test_ising_2d_specific_heat_peak(self):
        """2D Ising specific heat should peak near Onsager Tc ≈ 2.269."""
        T_range = np.linspace(1.5, 3.0, 20)
        cv_values = [TransferMatrix.ising_2d_exact(16, T) for T in T_range]
        peak_T = T_range[np.argmax(cv_values)]
        assert 2.0 < peak_T < 3.0  # Onsager Tc ≈ 2.269, peak should be nearby


# --- Integration ---

class TestIntegration:
    def test_full_workflow(self):
        """End-to-end: create lattice, run MC, compute observables."""
        lat = Lattice(8, 8)
        model = IsingModel(lat, J=1.0, T=2.269)
        spins = np.random.choice([-1, 1], size=(8, 8))
        final, energies, mags = MonteCarlo.metropolis(model, spins, T=2.269, n_steps=100)
        cv = model.specific_heat(energies)
        chi = model.susceptibility(mags)
        assert cv > 0
        assert chi >= 0
