# Experiments

This folder contains comprehensive benchmarking and analysis experiments for the QIF-Micro library.

## Heuristic adversary analysis

The `heuristic.ipynb` Jupyter notebook walks through a comparative analysis between:

- **Optimal Bayesian adversary**: The theoretically optimal trategy modeled by the library
- **Heuristic adversary**: A practical, simplified attack strategy

## Benchmarking Tools

The `benchmark/` subdirectory contains performance benchmarking tools that measure runtime and memory usage of the library across different privacy mechanisms and dataset configurations.

### Benchmark Categories

**Generic Benchmark** (`generic/`)
- Benchmarks the generic privacy model with sanitization options
- Covers combinations of sanitizing categorical and numerical attributes
- Tests the geometric mechanism and random response mechanisms

**Count-Sum Benchmark** (`count_sum/`)
- Benchmarks the count-sum aggregation query model
- Tests privacy mechanisms for aggregate database queries
- Simplified model without attribute-level sanitization options

### Running Benchmarks

The benchmark tools use shell scripts for easy execution:

```bash
# Run generic mechanism benchmarks
sh benchmark/run_generic.sh

# Run count-sum mechanism benchmarks
sh benchmark/run_count_sum.sh

# Run all benchmarks (generic + count-sum)
sh benchmark/run_all.sh
```

Each script:
1. Runs benchmarks with multiple dataset configurations (sizes, domains, iterations)
2. Collects timing and peak memory measurements
3. Generates visualization plots (SVG format)

### Benchmark Output

Results are saved in `benchmark/results/` organized by scenario:
- `{scenario_name}/time.parquet`: Execution time measurements
- `{scenario_name}/peak.parquet`: Peak memory usage
- `benchmark/plots/`: Generated visualization plots

### Customizing Benchmarks

For more control over benchmarking parameters, run the Python modules directly:

```bash
# Generic benchmarks with custom parameters
uv run python -m benchmark.generic.run \
  --load-from benchmark/generic/scenarios \
  --scenarios 100x2_5e3.yaml 200x4_5e3.yaml \
  --iterations 3 \
  --experiments 5

# Plot results separately
uv run python -m benchmark.generic.plot \
  --load-from benchmark/results \
  --scenarios 100x2_5e3 200x4_5e3

# Count-sum benchmarks
uv run python -m benchmark.count_sum.run \
  --load-from benchmark/count_sum/scenarios \
  --scenarios 100x2_5e3.yaml 200x4_5e3.yaml \
  --iterations 3 \
  --experiments 5

# Count-sum plots
uv run python -m benchmark.count_sum.plot \
  --load-from benchmark/results \
  --scenarios 100x2_5e3 200x4_5e3
```

### Benchmark Configuration

Benchmark scenarios are defined in YAML files under `generic/scenarios/` and `count_sum/scenarios/`:

- `{n_num}x{n_cat}_{n_entries}.yaml`: Dataset with `n_num` numerical domain size; `n_cat` categorical domain size; and `n_entries` dataset entries
- Example: `100x2_5e3.yaml` means 100-element numerical domain, 2-element categorical domain, 5000 entries

Each scenario can specify multiple attribute sanitization configurations (generic) or a single configuration (count-sum).
