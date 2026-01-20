from collections.abc import Iterable

import polars as pl

from qif_micro.datatypes import channel, LazyChannel

def build(
    input_domain: Iterable[int],
    output_domain: Iterable[int],
    alpha: float
) -> LazyChannel:
    """
    This function constructs a truncated geometric noise
    mapping integers to integers.

    We assume the input domain is a subset (possibly eq) of the output.
    The truncated geometric noise is then defined as

    let mask = output: [min(output_domain) < output < max(output_domain)]
    Pr(out | in) = (1 - alpha)^mask(out) * alpha^(d(in, out)) / (1 + alpha)

    where [P] = 1 if P else 0 and d(a, b) is a distance measure

    TODO: allow custom distance metrics?

    ## Example
    >>> import polars as pl
    >>> from qif_micro import mechanism
    >>> from qif_micro.datatypes import channel
    >>> input_domain = [0, 1, 2]
    >>> output_domain = [0, 1, 2]
    >>> ch = mechanism.geometric(input_domain, output_domain, 1/3)
    >>> channel.collect(ch)
    shape: (9, 3)
    ┌───────┬────────┬──────────┐
    │ input ┆ output ┆ p        │
    │ ---   ┆ ---    ┆ ---      │
    │ i64   ┆ i64    ┆ f64      │
    ╞═══════╪════════╪══════════╡
    │ 0     ┆ 0      ┆ 0.75     │
    │ 0     ┆ 1      ┆ 0.166667 │
    │ 0     ┆ 2      ┆ 0.083333 │
    │ 1     ┆ 0      ┆ 0.25     │
    │ 1     ┆ 1      ┆ 0.5      │
    │ 1     ┆ 2      ┆ 0.25     │
    │ 2     ┆ 0      ┆ 0.083333 │
    │ 2     ┆ 1      ┆ 0.166667 │
    │ 2     ┆ 2      ┆ 0.75     │
    └───────┴────────┴──────────┘
    """
    # ==================================================
    # Pre-conditions: validate
    #   - 0 <= alpha <= 1
    #   - input/output domains
    # ==================================================
     
    if (alpha < 0) or (alpha > 1):
        raise ValueError("Alpha param has to be in the range [0, 1]")

    input_domain = set(input_domain)
    output_domain = set(output_domain)

    if len(input_domain - output_domain) > 0:
        raise ValueError("Input domain must be a subset of output")

    # ==================================================
    # Finished pre-conditions
    # ==================================================
    
    input_domain = pl.LazyFrame({"input": list(input_domain)})
    output_domain = pl.LazyFrame({"output": list(output_domain)})

    _ = input_domain.join(output_domain, how="cross")
    _ = _.with_columns(
        (pl.col("input") - pl.col("output")).abs().alias("d"),
        (pl.col("output") == pl.col("output").min()).alias("is_min"),
        (pl.col("output") == pl.col("output").max()).alias("is_max"),
    )

    mask_expr = (~pl.col("is_max") & ~pl.col("is_min")).cast(int)
    cond_expr = (1 - alpha)**mask_expr
    fixed_expr = alpha**pl.col("d") / (1 + alpha)
    p_expr = (cond_expr * fixed_expr).alias("p")

    ch_dist = _.select("input", "output", p_expr)
    return channel.make_lazy(ch_dist, ["input"], ["output"])
