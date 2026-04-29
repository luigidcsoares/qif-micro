"""Plotting tool for count_sum benchmark results

Generates plots from .parquet result files produced by run.py.
"""
import argparse
from pathlib import Path

import polars as pl

from benchmark.utils import plotting


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot count_sum benchmark results"
    )

    parser.add_argument(
        "--load-from",
        type=str,
        required=True,
        help="Directory containing benchmark result subdirectories"
    )

    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[],
        help=("Scenario names to plot (subdirectory names in --load-from). "
              "If not specified, plot all results in --load-from.")
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/plots",
        help="Directory to save plots (default: benchmark/plots)"
    )

    return parser.parse_args()


def main():
    """
    Generate plots from benchmark results.

    Reads .parquet files from result directories and creates plots
    for execution time and memory usage.
    """
    args = _parse_args()

    load_from_path = Path(args.load_from)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_dirs = [
        load_from_path / f"count-sum_{d}"
        for d in args.scenarios
    ]

    # If no scenario was selected, load all subdirectiores
    if len(scenario_dirs) == 0: scenario_dirs = [
        d for d in load_from_path.iterdir() if d.is_dir()
    ]

    plotdata_time = pl.DataFrame()
    plotdata_peak = pl.DataFrame()

    # Load data from each scenario
    for scenario_dir in sorted(scenario_dirs):
        time_file = scenario_dir / "time.parquet"
        peak_file = scenario_dir / "peak.parquet"

        if not time_file.exists() or not peak_file.exists():
            print(f"Skipping {scenario_dir.name}: missing .parquet files")
            continue

        result_time = pl.read_parquet(time_file)
        result_peak = pl.read_parquet(peak_file)

        plotdata_time = pl.concat([plotdata_time, result_time])
        plotdata_peak = pl.concat([plotdata_peak, result_peak])

    step = "count-sum"
    # Generate execution time plots
    if plotdata_time.height > 0:
        plot_data = plotdata_time.explode("time")
        xvalues = plot_data["length"].unique().sort().to_list()
        ymax = plot_data["time"].max()

        chart = plotting.running_time(
            plot_data,
            xvalues,
            ymax,  # ty:ignore[invalid-argument-type]
            log_scale=False
        )

        path = output_dir / f"time_{step}.svg"
        chart.save(path)
        print(f"Plot on execution time saved to {path}")

    # Generate memory usage plots
    if plotdata_peak.height > 0:
        xvalues = plotdata_peak["length"].unique().sort().to_list()
        ymax = plotdata_peak["peak"].max()

        chart = plotting.peak_memory(
            plotdata_peak,
            xvalues,
            ymax,  # ty:ignore[invalid-argument-type]
            log_scale=False
        )

        path = output_dir / f"peak_{step}.svg"
        chart.save(path)
        print(f"Plot on memory usage saved to {path}")

    print("All plots completed successfully")


if __name__ == "__main__":
    main()
