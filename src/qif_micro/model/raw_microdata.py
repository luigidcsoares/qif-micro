from collections.abc import Iterable

import polars as pl

from qif_micro import qif
from qif_micro._internal.dataset import (
    _extract_from_record,
    _prepare_dataset,
)
from qif_micro.datatypes import Channel, Joint, ProbabDist

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
    >>> prior, ch, map_records, map_hints = model.raw_microdata.build(
    ...     dataset,
    ...     owner_col,
    ...     hint_cols,
    ...     []
    ... )
    >>> prior
    shape: (3, 2)
    ┌───────────┬──────────┐
    │ record_id ┆ p        │
    │ ---       ┆ ---      │
    │ u32       ┆ f64      │
    ╞═══════════╪══════════╡
    │ 1         ┆ 0.333333 │
    │ 2         ┆ 0.333333 │
    │ 3         ┆ 0.333333 │
    └───────────┴──────────┘
    >>> ch
    shape: (5, 3)
    ┌───────────┬─────────┬──────────┐
    │ record_id ┆ hint_id ┆ p        │
    │ ---       ┆ ---     ┆ ---      │
    │ u32       ┆ u32     ┆ f64      │
    ╞═══════════╪═════════╪══════════╡
    │ 1         ┆ 1       ┆ 0.333333 │
    │ 1         ┆ 2       ┆ 0.666667 │
    │ 2         ┆ 1       ┆ 0.5      │
    │ 2         ┆ 3       ┆ 0.5      │
    │ 3         ┆ 2       ┆ 1.0      │
    └───────────┴─────────┴──────────┘
    """
    record_cols = [*hint_cols, *other_cols]

    expr_rec_id = pl.col("record").rank("dense").alias("record_id")
    dataset = (
        _prepare_dataset(dataset.lazy(), owner_col, record_cols)
        .select("record", expr_rec_id)
    )

    expr_p = (pl.col("len") / pl.col("len").sum()).alias("p")
    prior_dist = (
        dataset
        .group_by("record_id")
        .agg(pl.len())
        .select("record_id", expr_p)
    )

    prior = ProbabDist.from_polars(prior_dist, ["record_id"])

    expr_hint = pl.struct(hint_cols).alias("hint")
    expr_hint_id = pl.col("hint").rank("dense").alias("hint_id")
    expr_record_len = pl.col("record").list.len().alias("record_len")
    
    dataset_with_meta = ( 
        dataset.unique().pipe(_extract_from_record, hint_cols)
        .select(pl.exclude("record"), expr_record_len)
        .explode(hint_cols)
        .select("record_id", "record_len", expr_hint)
        .with_columns(expr_hint_id)
    )

    expr_p = (pl.col("len") / pl.col("record_len")).alias("p")
    ch_dist = (
        dataset_with_meta
        .group_by("record_id", "hint_id")
        .agg(pl.len(), pl.col("record_len").first())
        .select("record_id", "hint_id", expr_p)
    )

    ch = Channel.from_polars(ch_dist, ["record_id"], ["hint_id"])

    map_records = dataset.unique()
    map_hints = dataset_with_meta.select("hint", "hint_id").unique()

    return prior, ch, map_records, map_hints
