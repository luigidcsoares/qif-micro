"""Plotting tool for generic benchmark results

Generates plots from .parquet result files produced by run.py.
"""
import argparse
from pathlib import Path

import polars as pl

from benchmark.utils import plotting


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot generic benchmark results"
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

    scenario_dirs = []
    for d in args.scenarios:
        scenario_dirs.append(load_from_path / f"cat_{d}")
        scenario_dirs.append(load_from_path / f"num_{d}")
        scenario_dirs.append(load_from_path / f"both_{d}")

    # If no scenario was selected, load all subdirectiores
    if len(scenario_dirs) == 0: scenario_dirs = [
        d for d in load_from_path.iterdir() if d.is_dir()
    ]

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

    # Load and categorize data from each scenario
    for scenario_dir in sorted(scenario_dirs):
        time_file = scenario_dir / "time.parquet"
        peak_file = scenario_dir / "peak.parquet"
        cfg_file = scenario_dir / "cfg.parquet"

        if not time_file.exists() or not peak_file.exists():
            print(f"Skipping {scenario_dir.name}: missing .parquet files")
            continue

        result_time = pl.read_parquet(time_file)
        result_peak = pl.read_parquet(peak_file)

        # Determine sanitisation kind
        if cfg_file.exists():
            cfg_df = pl.read_parquet(cfg_file)
            sanitise_cat = cfg_df["sanitise_cat"].item()
            sanitise_num = cfg_df["sanitise_num"].item()
        else:
            # Fallback: assume from name pattern
            name = scenario_dir.name
            sanitise_cat = "cat" in name
            sanitise_num = "num" in name

        if sanitise_cat and sanitise_num: kind = "both"
        elif sanitise_cat: kind = "cat"
        elif sanitise_num: kind = "num"
        else: kind = "none"

        plotdata_time[kind] = pl.concat([plotdata_time[kind], result_time])
        plotdata_peak[kind] = pl.concat([plotdata_peak[kind], result_peak])

    # Generate execution time plots
    for kind, data in plotdata_time.items():
        if data.height == 0: continue

        data = data.explode("time")
        xvalues = data["length"].unique().sort().to_list()
        ymax = data["time"].max()

        for step in ["mechanism", "model", "risk", "all"]:
            plot_data = data.filter(pl.col("step") == step)
            if plot_data.height == 0: continue

            chart = plotting.running_time(
                plot_data,
                xvalues,
                ymax,  # ty:ignore[invalid-argument-type]
                log_scale=True
            )

            path = output_dir / f"time_{kind}_{step}.svg"
            chart.save(path)
            print(f"Plot on execution time saved to {path}")

    # Generate memory usage plots
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

        path = output_dir / f"peak_{kind}.svg"
        chart.save(path)
        print(f"Plot on memory usage saved to {path}")

    print("All plots completed successfully")


if __name__ == "__main__":
    main()
