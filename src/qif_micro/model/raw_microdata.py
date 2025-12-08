from collections.abc import Iterable

import polars as pl

from qif_micro._internal.dataset import (
    _extract_from_record,
    _prepare_dataset,
)
from qif_micro.datatypes import Channel, ProbabDist

def build(
    dataset: pl.DataFrame | pl.LazyFrame,
    owner_col: str,
    hint_cols: Iterable[str],
    other_cols: Iterable[str]
) -> tuple[ProbabDist, Channel, pl.LazyFrame, pl.LazyFrame]:
    """
    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> dataset = pl.DataFrame({
    ...     "uid": [0, 0, 1, 1, 2, 2, 2],
    ...     "amount": [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
    ... })
    >>> owner_col = "uid"
    >>> hint_cols = ["amount"]
    >>> prior, ch = model.raw_microdata.build(
    ...     dataset,
    ...     owner_col,
    ...     hint_cols,
    ...     []
    ... )
    >>> prior
    shape: (3, 2)
    ┌───────────────────────┬──────────┐
    │ record                ┆ p        │
    │ ---                   ┆ ---      │
    │ list[struct[1]]       ┆ f64      │
    ╞═══════════════════════╪══════════╡
    │ [{0.0}, {1.0}, {1.0}] ┆ 0.333333 │
    │ [{0.0}, {2.0}]        ┆ 0.333333 │
    │ [{1.0}, {1.0}]        ┆ 0.333333 │
    └───────────────────────┴──────────┘
    >>> ch
    shape: (5, 3)
    ┌───────────────────────┬───────────┬──────────┐
    │ record                ┆ hint      ┆ p        │
    │ ---                   ┆ ---       ┆ ---      │
    │ list[struct[1]]       ┆ struct[1] ┆ f64      │
    ╞═══════════════════════╪═══════════╪══════════╡
    │ [{0.0}, {1.0}, {1.0}] ┆ {0.0}     ┆ 0.333333 │
    │ [{0.0}, {1.0}, {1.0}] ┆ {1.0}     ┆ 0.666667 │
    │ [{0.0}, {2.0}]        ┆ {0.0}     ┆ 0.5      │
    │ [{0.0}, {2.0}]        ┆ {2.0}     ┆ 0.5      │
    │ [{1.0}, {1.0}]        ┆ {1.0}     ┆ 1.0      │
    └───────────────────────┴───────────┴──────────┘
    """
    record_cols = [*hint_cols, *other_cols]

    dataset = _prepare_dataset(dataset.lazy(), owner_col, record_cols)

    expr_p = (pl.col("len") / pl.col("len").sum()).alias("p")

    record_freq = dataset.group_by("record").agg(pl.len())
    prior_dist = record_freq.select("record", expr_p)

    prior = ProbabDist.from_polars(prior_dist, ["record"])

    expr_hint = pl.struct(hint_cols).alias("hint")
    expr_record_len = pl.col("record").list.len().alias("record_len")
    
    records_with_hints = ( 
        dataset.drop("uid").unique().pipe(_extract_from_record, hint_cols)
        .with_columns(expr_record_len)
        .explode(hint_cols)
        .select("record", "record_len", expr_hint)
    )

    expr_p = (pl.col("len") / pl.col("record_len")).alias("p")

    ch_dist = (
        records_with_hints 
        .group_by("record", "hint")
        .agg(pl.len(), pl.col("record_len").first())
        .select("record", "hint", expr_p)
    )

    ch = Channel.from_polars(ch_dist, ["record"], ["hint"])
    return prior, ch
