from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import polars as pl

from qif_micro.qif.datatypes import Channel, Joint, Strategy

class AttrMechanism(Protocol):
    def __call__(
        self,
        input_domain: Iterable[Any],
        return_labels: bool = False
    ) -> Channel | tuple[Channel, Sequence[Any]]: ...
        
type RecordEntry = dict[str, Any]
type Record = list[RecodEntry]

# Allow only DataFrame, to ensure determinism!
type Dataset = pl.DataFrame
type MapLabels = pl.DataFrame
type MapOwners = pl.DataFrame

type BaselineModel = (
    Joint
    | tuple[Joint, MapOwners | MapLabels]
    | tuple[Joint, MapOwners, MapLabels]
)

type Model = (
    tuple[Joint, Strategy]
    | tuple[Joint, Strategy, MapOwners | MapLabels]
    | tuple[Joint, Strategy, MapOwners, MapLabels]
)
