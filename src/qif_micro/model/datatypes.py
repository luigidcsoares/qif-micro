import polars as pl

type Dataset = pl.DataFrame | pl.LazyFrame
type MapLabels = pl.DataFrame | pl.LazyFrame
type MapOwners = pl.DataFrame | pl.LazyFrame
