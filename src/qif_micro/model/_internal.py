from collections.abc import Iterable

import polars as pl

from qif_micro.typing import DataFrame


def _mk_long_dataset(
    records: Iterable[DataFrame],
    owner_col: str = "owner_id",
) -> DataFrame:
    # First we construct the new the longitudinal records
    def record_idx_expr(i): return (
        pl.struct("record", pl.lit(i).alias("i")).alias("record")
    )

    records_with_idx = (
        df.select(owner_col, record_idx_expr(i))
        for i, df in enumerate(records)
    )

    record_expr = pl.col("record").rank("dense") - 1
    return (
        pl.concat(records_with_idx)
        .group_by(owner_col)
        # The longitudinal record will be a sequence of record ids.
        .agg("record")
        # We then transform the seq of ids into in a single id (row)
        .with_columns(record_expr)
    )


def _mk_records(
    df: DataFrame,
    owner_col: str = "owner_id",
    entry_col: str = "entry_id"
) -> DataFrame:
    id_cols = [owner_col, entry_col]
    attrs = sorted(c for c in df.collect_schema().names() if c not in id_cols)
    record_entry_expr = pl.struct(entry_col, *attrs).alias("record")
    return df.sort(*id_cols).group_by(owner_col).agg(record_entry_expr)
