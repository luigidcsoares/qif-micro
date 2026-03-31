from collections.abc import Iterable
from typing import Any

from qif_micro.typing import DataFrame

def _filter_optional(xs: Iterable[Any]) -> list[Any]:
    return [x for x in xs if x is not None]


def _standard_cols(df: DataFrame) -> DataFrame:
    cols = sorted(df.collect_schema().names())
    return df.select(*cols)

   
def _valid_columns(
    df: DataFrame,
    required: Iterable[str]
) -> tuple[bool, set[str]]:
    missing = set(required) - set(df.collect_schema().names())
    return len(missing) == 0, missing
