import polars as pl

from qif_micro.qif.datatypes import Joint, Strategy

type Dataset = pl.DataFrame | pl.LazyFrame
type MapLabels = pl.DataFrame | pl.LazyFrame
type MapOwners = pl.DataFrame | pl.LazyFrame

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
