from collections.abc import Iterable
from typing import Any

import polars as pl

def _filter_optional(xs: Iterable[Any]) -> Iterable[Any]:
    return (x for x in xs if x is not None)

   
def _valid_columns(
    lf: pl.LazyFrame,
    required: Iterable[str]
) -> tuple[bool, set[str]]:
    missing = set(required) - set(lf.collect_schema().names())
    return len(missing) == 0, missing
