"""Entry point for count-sum benchmarks

Runs count-sum benchmarks with configurable parameters. Supports:
- YAML scenarios: --load-from <dir> --scenarios file1.yaml file2.yaml
- All scenarios from directory: --load-from <dir>
- CLI arguments: --n-entries 1000 --iterations 5
"""
import argparse
from pathlib import Path

import polars as pl

from benchmark.count_sum import benchmark
from benchmark.count_sum.config import ExperimentConfig, load_multiple_scenarios


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run count-sum benchmarks")

    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="Directory containing scenario files (default: None)"
    )

    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[],
        help=("Scenario file names to load from --load-from. "
              "If not specified, load all .yaml files from --load-from.")
    )

    parser.add_argument(
        "--n-entries",
        type=int,
        help="Number of entries in dataset"
    )

    parser.add_argument(
        "--n-cat",
        type=int,
        help="Domain size for cat attribute"
    )

    parser.add_argument(
        "--n-num",
        type=int,
        help="Domain size for num attribute"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of iterations (default: 3)"
    )

    parser.add_argument(
        "--experiments",
        type=int,
        default=5,
        help="Number of distinct experiments (default: 5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/benchmark/results",
        help="Directory to save results (default: experiments/benchmark/results)"
    )

    return parser.parse_args()


def main():
    """
    Run count-sum benchmarks.

    Supports multiple invocation modes:
    1. YAML scenarios: --load-from <dir> --scenarios file1.yaml file2.yaml
    2. All scenarios from directory: --load-from <dir>
    3. CLI arguments: partial args allowed, missing use defaults
    4. Defaults: runs with built-in defaults

    The --iterations flag is separate and applies to all configs (default: 3).

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = _parse_args()

    configs = []

    # Load from YAML scenarios if --load-from is provided
    if args.load_from:
        load_from_path = Path(args.load_from)
        scenario_paths = [load_from_path / f for f in args.scenarios]

        # If no scenario was selected, load all .yaml files from directory
        if len(scenario_paths) == 0:
            scenario_paths = [load_from_path]
        
        configs = load_multiple_scenarios(scenario_paths)  # ty:ignore[invalid-argument-type]
        print(f"Loaded {len(configs)} scenarios from {args.load_from}")

    # Create config from CLI arguments
    # Pre-condition: these arguments must not have a default value
    # defined via argparse, otherwise they will always be set.
    cli_cfg_dict = {}
    if args.n_entries: cli_cfg_dict["n_entries"] = args.n_entries
    if args.n_cat: cli_cfg_dict["n_cat"] = args.n_cat
    if args.n_num: cli_cfg_dict["n_num"] = args.n_num
    if args.iterations: cli_cfg_dict["iterations"] = args.iterations

    # If any CLI arg was provided, fill missing with defaults
    if len(cli_cfg_dict.keys()) > 0:
        cli_cfg = ExperimentConfig(**cli_cfg_dict)
        configs.append(("cli_config", cli_cfg))

    # Use defaults if no configs loaded
    if not configs:
        configs = [("default", ExperimentConfig())]
        print("Running with default configuration")

    # Run benchmarks
    print(f"Running {len(configs)} benchmark(s)...\n")

    for name, cfg in configs:
        print(f"[{name}] Starting experiment...")
        print(f"  n_entries={cfg.n_entries}, n_cat={cfg.n_cat}, "
              f"n_num={cfg.n_num}")
        print(f"  iterations={args.iterations}")

        result_time, result_peak = benchmark.run_many(cfg, args.experiments)
        output_dir = Path(args.output_dir) / name

        cfg_df = pl.DataFrame(cfg.to_dict())

        cfg_df.write_parquet(output_dir / "cfg.parquet", mkdir=True)
        result_time.write_parquet(output_dir / "time.parquet", mkdir=True)
        result_peak.write_parquet(output_dir / "peak.parquet", mkdir=True)

        print(f"  ✓ Completed. Results saved to {output_dir}\n")

    print("All benchmarks completed successfully")


if __name__ == "__main__":
    main()
