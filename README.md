# QIF-Micro

A Quantitative Information Flow (QIF) library for measuring inference risks in linkage attacks against datasets produced from a privacy-preserving pipeline.

## Installation & Setup

### Step 1: Install the package manager `uv`

If you use Nix flakes, simply run:

```bash
nix develop
```

This sets up the complete development environment with the correct Python version and all dependencies.

If you don't use Nix, install the `uv` package manager. Visit the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for instructions specific to your platform.

### Step 2: Install the library

Once you have `uv` installed, synchronise dependencies:

```bash
uv sync # Installs the library with Polars' default runtime
uv sync --extra rtcompat # Installs the library with Polars' legacy runtime (for older CPUs)
```

## Quick start

Check out the interactive tutorial in `tutorial.ipynb` to walk through the library's core functionality.

## Documentation

For detailed API documentation, examples and benchmarking information, see:

- `src/qif_micro/`: Core library modules
- `tests/`: Test suite with examples
- `experiments/benchmark/`: Performance benchmarking tools
