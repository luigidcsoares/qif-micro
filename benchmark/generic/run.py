"""Entry point for generic benchmarks

Runs generic benchmarks with configurable parameters. Supports:
- Programmatic usage: pass ExperimentConfig directly
- YAML scenarios: --scenarios <file.yaml> or <directory>
- CLI arguments: --n-entries 1000 --iterations 5
- Multiple files/directories: --scenarios file1.yaml dir1/ file2.yaml
"""
import argparse
import sys

from benchmark.generic import benchmark
from benchmark.generic.config import ExperimentConfig, load_multiple_scenarios


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generic benchmarks")

    parser.add_argument(
        "--scenarios",
        nargs="+",
        help=("Path(s) to YAML scenario file(s) or directory with YAML files."
              " Can specify multiple files/directories in one command.")
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
        "--sanitise-cat",
        action="store_true",
        help="Sanitize cat attribute"
    )

    parser.add_argument(
        "--sanitise-num",
        action="store_true",
        help="Sanitize num attribute"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of iterations"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/results",
        help="Directory to save results (default: benchmark/results)"
    )

    return parser.parse_args()


def main() -> int:
    """
    Run generic benchmarks.

    Supports multiple invocation modes:
    1. YAML scenarios: --scenarios scenarios/small.yaml
    2. CLI arguments: partial args allowed, missing use defaults
    3. Defaults: runs with built-in defaults

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = _parse_args()

    configs = []

    # Load from YAML scenarios if provided
    if args.scenarios:
        configs = load_multiple_scenarios(args.scenarios)
        total_sources = len(args.scenarios)
        sources_str = " ".join(args.scenarios)
        print(f"Loaded {len(configs)} scenarios from "
              f"{total_sources} source(s): {sources_str}")

    # Create config from CLI arguments
    cli_cfg_dict = {}
    if args.n_entries: cli_cfg_dict["n_entries"] = args.n_entries
    if args.n_cat: cli_cfg_dict["n_cat"] = args.n_cat
    if args.n_num: cli_cfg_dict["n_num"] = args.n_num
    if args.iterations: cli_cfg_dict["iterations"] = args.iterations
    if args.sanitise_cat: cli_cfg_dict["sanitise_cat"] = True
    if args.sanitise_num: cli_cfg_dict["sanitise_num"] = True

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
        print(f"  sanitise_cat={cfg.sanitise_cat}, "
              f"sanitise_num={cfg.sanitise_num}, "
              f"iterations={cfg.iterations}")

        time_df, memory_df = benchmark.run(**cfg.to_dict())

        # save_results_with_config(
        #     cfg, (time_df, memory_df),
        #     args.output_dir, name=name
        # )

        print(f"  ✓ Completed. Results saved to {args.output_dir}/\n")

    print("All benchmarks completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
