# lattice-climate

Lattice Hamiltonian models for climate simulation. Implements Ising and Potts models on geographical grids with temperature-driven phase transitions, connecting statistical mechanics to real climate data.

## Features

- **2D/3D lattices** with configurable boundary conditions (periodic, fixed, open)
- **Ising model** for binary climate states (e.g., above/below freezing)
- **Potts model** for multi-state climate zones
- **Monte Carlo samplers**: Metropolis-Hastings, Wolff cluster, Swendsen-Wang
- **Climate mapping**: temperature → spins, precipitation → Potts states, spatial autocorrelation
- **Phase detection**: critical temperature via specific heat peaks, Binder cumulant, finite-size scaling
- **Transfer matrix**: exact solutions for 1D Ising and Onsager's exact 2D solution

## Quick Start

```python
import numpy as np
from lattice_climate import Lattice, IsingModel, MonteCarlo

# Create a 32×32 lattice
lattice = Lattice(32, 32, boundary="periodic")

# Initialize random spins
spins = np.random.choice([-1, 1], size=lattice.shape)

# Run Ising model at various temperatures
model = IsingModel(lattice, J=1.0, T=2.269)
final, energies, mags = MonteCarlo.metropolis(model, spins, T=1.5, n_steps=5000)
```

## Installation

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
