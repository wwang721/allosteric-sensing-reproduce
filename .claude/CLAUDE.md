# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproducible code for the paper *"Perfect adaptation in eukaryotic gradient sensing using cooperative allosteric binding"* (Physical Review E, 2026). The codebase runs biophysics simulations and generates all figures in the paper.

## Running Scripts

There is no build system. Each script must be run from within its own directory:

```bash
cd compare_models && python FI_comparing.py
cd adapt_dynamics && python response.py
cd adapt_dynamics && python time_series.py
cd check_perturbation && python plot.py
cd C0_alpha && python plot.py
cd vary_KM && python plot.py
cd adaptation_time && python plot.py
```

> **Important:** Always `cd` into the directory first. Running scripts from the repo root (e.g. `python check_perturbation/plot.py`) will fail due to relative path dependencies.

For HPC long-running simulations (adaptation_time):
```bash
cd adaptation_time && sbatch run.slurm
```

## Dependencies

Python 3.11 with numpy, matplotlib, scipy. Install via:

```bash
uv venv
uv pip install numpy==2.1.3 matplotlib==3.10.0 scipy==1.15.3
```

For headless (non-GUI) environments, set `MPLBACKEND=Agg`.

## CI

GitHub Actions (`.github/workflows/auto-plot.yml`) runs all scripts on push/PR, then auto-commits any regenerated `.png` figures back to `main` via a GitHub App token. No linting or unit tests exist — CI success means all scripts run to completion.

## Architecture

Each top-level directory corresponds to specific paper figures and is self-contained:

| Directory | Figures | Purpose |
|-----------|---------|---------|
| `compare_models/` | 2, 6 | Fisher Information comparison across receptor models; heatmaps vs. diffusion/concentration |
| `adapt_dynamics/` | 4 | Response curves and time-series for different Michaelis-Menten constants |
| `check_perturbation/` | 5 | High vs. low diffusion regime comparison to validate perturbation analysis |
| `C0_alpha/` | 7 | Fisher Information sensitivity to initial G-protein concentration and allosteric coupling |
| `vary_KM/` | 8 | Fisher Information dependence on KM |
| `adaptation_time/` | 9 | Adaptation time analysis (3 scripts for HPC parallelization) |

Each directory typically has:
- `simulation.py` — runs the computation and saves `.npy`/`.npz` data
- `plot.py` — loads saved data and generates `.png` figures
- Pre-computed data and figures already checked in

## Code Patterns

All scripts are procedural (no classes). The typical flow is:
1. Define physical parameters at the top of the file
2. Initialize spatial grids (circular cell geometry, periodic BCs)
3. Time-step coupled ODEs/PDEs (reaction + diffusion)
4. Compute Fisher Information as the performance metric
5. Save results as numpy arrays; plot with matplotlib

Key numerical constraint: stability requires `D * dt / dx² < 0.5`.
