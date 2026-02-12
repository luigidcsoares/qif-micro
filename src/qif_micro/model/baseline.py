from collections.abc import Iterable

import polars as pl

from scipy.sparse import coo_array

from qif_micro.qif.datatypes import ProbabDist, Channel
from qif_micro._internal import _valid_columns

def build(
    dataset: pl.DataFrame | pl.LazyFrame,
    hint_attrs: Iterable[str],
    *,
    owner_col: str = "owner_id",
    record_col: str = "record"
) -> tuple[ProbabDist, Channel]:
    """
    ## Example
    >>> import polars as pl
    >>> from qif_micro import model

    >>> dataset = pl.DataFrame({
    ...     "owner_id": [0, 1, 2, 2, 3, 3],
    ...     "hint": [0, 0, 0, 1, 0, 1],
    ...     "sensitive": [0, 0, 0, 0, 1, 1]
    ... })
    
    >>> pi, ch = model.baseline(dataset, ["hint"])

    >>> pi
    ProbabDist(dist=array([0.5 , 0.25, 0.25]))

    >>> ch.dist.toarray()
    array([[1. , 0. ],
           [0.5, 0.5],
           [0.5, 0.5]])
    """
    # =============================================================
    # Pre-conditions: The dataset must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    # =============================================================
    dataset = dataset.lazy()
    
    _, ok_owner = _valid_columns(dataset, [owner_col])
    if not ok_owner:
       raise ValueError(f"Dataset must have a column ``owner_id``")

    schema = dataset.collect_schema()
    missing_attrs = set(hint_attrs) - set(schema.keys())
    if len(missing_attrs) > 0:
        raise ValueError(f"Missing the following attrs: {missing_attrs}")

    # =============================================================
    # End pre-conditions
    # =============================================================

    # We begin by constructing a map from records to hints,
    # so that each record is identified as a row (of the prior and channel),
    # and each hint is identified as a column (of the channel).
    # 
    # We also add the record length as a metadata.
    len_expr = pl.len().alias("len")
    hint_expr = (pl.struct(hint_attrs).rank("dense") - 1).alias("hint")
    record_entry_expr = pl.struct(pl.exclude(owner_col)).alias("record_entry")
    record_expr = (pl.col("record_entry").rank("dense") - 1).alias("record")

    record_attrs = [c for c in schema.keys() if c != owner_col]
    map_records_to_hints = (
        dataset
        .select(owner_col, record_entry_expr, hint_expr)
        .group_by(owner_col)
        .agg("record_entry", "hint", len_expr)
        .select(record_expr, "hint", "len")
    )

    n_records_expr = pl.len().alias("n_records")
    p_expr = (pl.len() / pl.col("n_records").first()).alias("p")

    prior_dist = (
        map_records_to_hints
        .with_columns(n_records_expr)
        .group_by("record")
        .agg(p_expr)
        .sort("record")
        .select("p")
        .collect()
        .to_numpy()
        .ravel()
    )

    p_expr = (pl.len() / pl.col("len").first()).alias("p")
    ch_dist_df = (
        map_records_to_hints
        # Drop possible duplicate records from the dataset,
        # as in the case of the channel we count things within records
        .unique()
        .explode("hint")
        .group_by("record", "hint")
        .agg(p_expr)
        .sort("record", "hint")
        .collect()
    )

    n_rows = ch_dist_df["record"].max() + 1
    n_cols = ch_dist_df["hint"].max() + 1

    data = ch_dist_df["p"].to_numpy()
    rows = ch_dist_df["record"].to_numpy()
    cols = ch_dist_df["hint"].to_numpy()

    ch_dist = coo_array((data, (rows, cols)), shape=(n_rows, n_cols))

    pi = ProbabDist(prior_dist)
    ch = Channel(ch_dist.tocsr())
    
    return pi, ch
