"""Plotting utilities for benchmarking"""
from collections.abc import Sequence

import altair as alt
import polars as pl


def running_time(
    results: pl.DataFrame,
    xvalues: Sequence[int],
    ymax: float | None = None,
    log_scale: bool = False
) -> alt.Chart:
    """
    Create Altair line plot for timing results.

    Parameters
    ----------
    results : pl.DataFrame
        DataFrame with timing data.

    ymax : int, optional (default: None)
        Control the domain of the y-axis
    """
    domain_len_expr = (pl.col("n_num") * pl.col("n_cat")).alias("domain_len")
    domain_str_expr = (
        pl.concat_str("n_num", "n_cat", separator=" x ").alias("domain")
    )

    results = results.with_columns(domain_len_expr, domain_str_expr)

    sort_domain = (
        results
        .select("domain_len", "domain").unique()
        .sort("domain_len")["domain"]
        .to_list()
    )

    color_range = ["#332288", "#44AA99", "#999933", "#CC6677", "#AA4499"]
    scale = alt.Scale(range=color_range, domain=sort_domain)
    color = alt.Color("domain", title="Domain", scale=scale)

    shape_range = ["circle", "square", "triangle", "diamond", "cross"]
    scale = alt.Scale(range=shape_range, domain=sort_domain)
    shape = alt.Shape("domain", title="Domain", scale=scale)

    chart = alt.Chart(results)

    xaxis = alt.Axis(values=xvalues, format="e")
    xaxis_options = {"axis": xaxis, "title": "Dataset length"}
    if log_scale:
        xaxis_options |= {"scale": alt.Scale(type="log")}  # ty:ignore[unsupported-operator]
    
    yaxis_options = {"title": "Running time (second)"}
    if ymax is not None:
        yaxis_options |= {"scale": alt.Scale(domain=[0, ymax])}  # ty:ignore[unsupported-operator]

    x = alt.X("length", **xaxis_options)  # ty:ignore[invalid-argument-type]
    y = alt.Y("mean(time)", **yaxis_options)  # ty:ignore[invalid-argument-type]

    line = chart.mark_line(tooltip=True).encode(x=x, y=y, color=color)

    y = alt.Y("time", **yaxis_options)  # ty:ignore[invalid-argument-type]
    point = chart.mark_point(tooltip=True, size=70, filled=True).encode(
        x=x, y=y, color=color, shape=shape
    )
    
    return line + point


def peak_memory(
    results: pl.DataFrame,
    xvalues: Sequence[int],
    ymax: float | None = None,
    log_scale: bool = False
) -> alt.Chart:
    """
    Create Altair line plot for timing results.

    Parameters
    ----------
    results : pl.DataFrame
        DataFrame with timing data.

    ymax : int, optional (default: None)
        Control the domain of the y-axis
    """
    domain_len_expr = (pl.col("n_num") * pl.col("n_cat")).alias("domain_len")
    domain_str_expr = (
        pl.concat_str("n_num", "n_cat", separator=" x ").alias("domain")
    )

    results = results.with_columns(domain_len_expr, domain_str_expr)

    sort_domain = (
        results
        .select("domain_len", "domain").unique()
        .sort("domain_len")["domain"]
        .to_list()
    )

    color_range = ["#332288", "#44AA99", "#999933", "#CC6677", "#AA4499"]
    scale = alt.Scale(range=color_range, domain=sort_domain)
    color = alt.Color("domain", title="Domain", scale=scale)

    shape_range = ["circle", "square", "triangle", "diamond", "cross"]
    scale = alt.Scale(range=shape_range, domain=sort_domain)
    shape = alt.Shape("domain", title="Domain", scale=scale)

    chart = alt.Chart(results)

    xaxis = alt.Axis(values=xvalues, format="e")
    xaxis_options = {"axis": xaxis, "title": "Dataset length"}
    if log_scale:
        xaxis_options |= {"scale": alt.Scale(type="log")}  # ty:ignore[unsupported-operator]
    
    yaxis_options = {"title": "Memory usage (GiB)"}
    if ymax is not None:
        yaxis_options |= {"scale": alt.Scale(domain=[0, ymax])}  # ty:ignore[unsupported-operator]

    x = alt.X("length", **xaxis_options)  # ty:ignore[invalid-argument-type]
    y = alt.Y("mean(peak)", **yaxis_options)  # ty:ignore[invalid-argument-type]

    line = chart.mark_line(tooltip=True).encode(x=x, y=y, color=color)

    y = alt.Y("peak", **yaxis_options)  # ty:ignore[invalid-argument-type]
    point = chart.mark_point(tooltip=True, size=70, filled=True).encode(
        x=x, y=y, color=color, shape=shape
    )
    
    return line + point
