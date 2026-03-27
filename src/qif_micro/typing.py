from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import polars as pl

from qif_micro.qif.datatypes import Channel, Joint, Strategy

class AttrMechanism(Protocol):
    def __call__(
        self,
        return_labels: bool = False,
        **kwargs: Any,
    ) -> Channel | tuple[Channel, Sequence[Any]]: ...
        
type RecordEntry = dict[str, Any]
type Record = list[RecodEntry]

# Allow only DataFrame, to ensure determinism!
type Dataset = pl.DataFrame

# But keeping labels in memory is expensive, so use the lazy API:
type MapLabels = pl.LazyFrame
type MapOwners = pl.LazyFrame

type BaselineModel = (
    Joint
    | tuple[Joint, MapOwners | MapLabels]
    | tuple[Joint, MapOwners, MapLabels]
)

type Model = tuple[Joint, Strategy]
