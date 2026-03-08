from collections.abc import Iterable

import polars as pl

from qif_micro.model.datatypes import Dataset, MapOwners

def _mk_long_dataset(
    records_it: Iterable[MapOwners],
    owner_col: str = "owner_id"
) -> Dataset:
    # First we construct the new the longitudinal records
    record_idx_expr = lambda i: (
        pl.struct("record", pl.lit(i).alias("i"))
        .alias("record")
    )

    records_with_idx_it = (
        m.select(owner_col, record_idx_expr(i))
        for i, m in enumerate(records_it)
    )

    record_expr = pl.col("record").rank("dense") - 1
    return (
        pl.concat(records_with_idx_it)
        # The longitudinal record will be a sequence of record ids.
        # Ensure order, so that rank is deterministic.
        .group_by(owner_col, maintain_order=True).agg("record")
        # We then transform the seq of ids into in a single id (row)
        .with_columns(record_expr)
    )


def _mk_records(dataset: Dataset, owner_col: str = "owner_id") -> Dataset:
    record_entry_expr = pl.struct(pl.exclude(owner_col)).alias("record")
    return dataset.group_by(owner_col).agg(record_entry_expr)
