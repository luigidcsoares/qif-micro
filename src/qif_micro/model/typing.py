import polars as pl

from qif_micro.qif.datatypes import Joint, Strategy

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
