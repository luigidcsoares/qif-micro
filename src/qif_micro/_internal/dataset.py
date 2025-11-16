from collections.abc import Iterable

import polars as pl

def _extract_from_record(
    dataset: pl.LazyFrame,
    cols: Iterable[str]
) -> pl.LazyFrame:
    _, ok = _valid_columns(dataset, ["record"])
    assert ok

    expr_field = lambda col: pl.element().struct.field(col)
    extract_list = [
        pl.col("record").list.eval(expr_field(col)).alias(col)
        for col in cols
    ]

    return dataset.with_columns(extract_list)


def _prepare_dataset(
    dataset: pl.LazyFrame,
    owner_col: str,
    record_cols: Iterable[str]
) -> pl.LazyFrame:
    expected_cols = [owner_col, *record_cols]
    diff, ok = _valid_columns(dataset, expected_cols)
    if not ok: raise ValueError(f"Missing columns {diff}")

    return (
        dataset
        .sort(record_cols)
        .with_columns(pl.struct(record_cols).alias("record"))
        .group_by(owner_col, maintain_order=True)
        .agg("record")
    )


def _single_record_per_owner(
    dataset: pl.LazyFrame,
    owner_col: str
) -> bool:
    n_records = (
        dataset
        .select(pl.col(owner_col).n_unique())
        .collect()
        .item()
    )

    return (
        dataset
        .select(n_records == pl.len())
        .collect()
        .item()
    )


def _valid_columns(
    lf: pl.LazyFrame,
    required: Iterable[str]
) -> tuple[set[str], bool]:
    missing = set(required) - set(lf.collect_schema().names())
    return missing, len(missing) == 0
