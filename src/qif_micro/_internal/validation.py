from collections.abc import Iterable

import polars as pl

def _valid_columns(
    lf: pl.LazyFrame,
    required: Iterable[str]
) -> tuple[set[str], bool]:
    missing = set(required) - set(lf.collect_schema().names())
    return missing, len(missing) == 0
