"""Entry point for generic benchmarks

Runs generic benchmarks with configurable parameters. Supports:
- Programmatic usage: pass ExperimentConfig directly
- YAML scenarios: --scenarios <file.yaml> or <directory>
- CLI arguments: --n-entries 1000 --iterations 5
- Multiple files/directories: --scenarios file1.yaml dir1/ file2.yaml
"""
import argparse
from pathlib import Path

import polars as pl

from benchmark.generic import benchmark
from benchmark.generic.config import ExperimentConfig, load_multiple_scenarios
from benchmark.utils import plotting


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
        default="benchmark/results",
        help="Directory to save results (default: benchmark/results)"
    )

    return parser.parse_args()


def main():
    """
    Run generic benchmarks.

    Supports multiple invocation modes:
    1. YAML scenarios: --scenarios scenarios/small.yaml
    2. CLI arguments: partial args allowed, missing use defaults
    3. Defaults: runs with built-in defaults

    The --iterations flag is separate and applies to all configs (default: 3).

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
        print(f"Loaded {len(configs)} scenarios from")

    # Create config from CLI arguments
    # Pre-condition: these arguments must not have a default value
    # defined via argparse, otherwise they will always be set.
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

    plotdata_time = {
        "cat": pl.DataFrame(),
        "num": pl.DataFrame(),
        "both": pl.DataFrame(),
        "none": pl.DataFrame()
    }

    plotdata_peak = {
        "cat": pl.DataFrame(),
        "num": pl.DataFrame(),
        "both": pl.DataFrame(),
        "none": pl.DataFrame()
    }

    for name, cfg in configs:
        print(f"[{name}] Starting experiment...")
        print(f"  n_entries={cfg.n_entries}, n_cat={cfg.n_cat}, "
              f"n_num={cfg.n_num}")
        print(f"  sanitise_cat={cfg.sanitise_cat}, "
              f"sanitise_num={cfg.sanitise_num}, "
              f"iterations={cfg.iterations}")

        result_time, result_peak = benchmark.run_many(cfg, args.experiments)
        output_dir = Path(args.output_dir) / name

        cfg_df = pl.DataFrame(cfg.to_dict())

        cfg_df.write_parquet(output_dir / "cfg.parquet", mkdir=True)
        result_time.write_parquet(output_dir / "time.parquet", mkdir=True)
        result_peak.write_parquet(output_dir / "peak.parquet", mkdir=True)

        if args.sanitise_cat and args.sanitise_num: kind = "both"
        elif args.sanitise_cat: kind = "cat"
        elif args.sanitise_num: kind = "num"
        else: kind = "both"

        plotdata_time[kind] = pl.concat([plotdata_time[kind], result_time])
        plotdata_peak[kind] = pl.concat([plotdata_peak[kind], result_peak])

        print(f"  ✓ Completed. Results saved to {output_dir}\n")

    print("All benchmarks completed successfully")

    for kind, data in plotdata_time.items():
        if data.height == 0: continue
    
        data = data.explode("time")
        xvalues = data["length"].unique().sort().to_list()
        ymax = data["time"].max()

        for step in ["mechanism", "model", "risk", "all"]:
            plot_data = data.filter(pl.col("step") == step)

            chart = plotting.running_time(
                plot_data,
                xvalues,
                ymax,  # ty:ignore[invalid-argument-type]
                log_scale=True
            )  

            path = Path(args.output_dir) / f"time_{kind}_{step}.svg"
            chart.save(path)
            print(f"Plot on execution time saved to {path}")

    for kind, plot_data in plotdata_peak.items():
        if plot_data.height == 0: continue
    
        xvalues = plot_data["length"].unique().sort().to_list()
        ymax = plot_data["peak"].max()

        chart = plotting.peak_memory(
            plot_data,
            xvalues,
            ymax,  # ty:ignore[invalid-argument-type]
            log_scale=True
        )  

        path = Path(args.output_dir) / f"peak_{kind}.svg"
        chart.save(path)
        print(f"Plot on memory usage saved to {path}")

            
if __name__ == "__main__":
    main()
