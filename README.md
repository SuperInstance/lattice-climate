# lattice-climate

**Lattice Hamiltonian models for climate simulation — Ising and Potts models on geographical grids with Monte Carlo sampling and phase transition detection.**

Maps real climate data to statistical mechanics lattice models. Temperature → Ising spins, precipitation → Potts states. Run Metropolis-Hastings, Wolff cluster, and Swendsen-Wang Monte Carlo. Detect phase transitions via specific heat peaks and Binder cumulant crossing.

## What This Gives You

- **Ising model** — binary climate states (warm/cold) with ferromagnetic coupling
- **Potts model** — multi-state climate zones (dry/wet/frozen/etc.)
- **Monte Carlo samplers** — Metropolis, Wolff cluster, Swendsen-Wang
- **Climate mapping** — temperature → spins, precipitation → Potts states
- **Phase detection** — critical temperature via specific heat peaks
- **Transfer matrix** — exact solutions for 1D Ising chains
- **Configurable lattices** — periodic, fixed, or open boundary conditions

## Quick Start

```python
from lattice_climate import Lattice, IsingModel, MonteCarlo, ClimateMapper

# Create a 20×20 lattice with periodic boundaries
lattice = Lattice(20, 20, boundary="periodic")

# Ising model at critical temperature
model = IsingModel(lattice, J=1.0, T=2.269, h=0.0)

# Run Monte Carlo
spins = np.ones(lattice.shape, dtype=int)
final, energies, mags = MonteCarlo.metropolis(model, spins, T=2.269, n_steps=1000)

# Map real temperature data to spins
mapper = ClimateMapper()
spins = mapper.temperature_to_spins(temps, threshold=15.0)
```

## Installation

```bash
pip install -e .
```

Requires: `numpy>=1.21`

## API Reference

| Module | Key Classes |
|--------|------------|
| `lattice` | `Lattice` — 2D grid with boundary conditions |
| `ising` | `IsingModel` — H = -JΣsᵢsⱼ - hΣsᵢ |
| `potts` | `PottsModel` — q-state Potts, H = -JΣδ(sᵢ,sⱼ) |
| `monte_carlo` | `MonteCarlo` — Metropolis, Wolff, Swendsen-Wang |
| `climate` | `ClimateMapper` — data → lattice mapping |
| `phase` | `PhaseDetector` — Tc via specific heat |
| `transfer` | `TransferMatrix` — exact 1D solutions |

## Testing

```bash
pytest tests/
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[regime-detection](https://github.com/SuperInstance/regime-detection)** — Detect climate regime changes via conservation drops
- **lattice-climate** — Statistical mechanics models for climate (this repo)

## License

MIT
