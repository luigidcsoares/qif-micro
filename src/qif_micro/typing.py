from typing import Any, TypeIs

import polars as pl

from qif_micro.qif.datatypes import Joint, Strategy

type DataFrame = pl.DataFrame | pl.LazyFrame

# Keeping labels in memory is expensive, so use the lazy API:
type MapLabels = pl.LazyFrame
type MapOwners = pl.LazyFrame

type BaselineModel = (
    Joint
    | tuple[Joint, MapOwners | MapLabels]
    | tuple[Joint, MapOwners, MapLabels]
)

type Model = tuple[Joint, Strategy]

type RecordEntry = dict[str, Any]
type Record = list[RecordEntry]


def is_dataframe(x: Any) -> TypeIs[DataFrame]:
    return isinstance(x, DataFrame.__value__.__args__)
