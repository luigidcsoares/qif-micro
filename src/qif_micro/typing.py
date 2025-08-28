from typing import Any

import polars

type ProbabDist = polars.LazyDataFrame

type Channel = polars.LazyDataFrame
type Hyper = tuple[ProbabDist, polars.LazyDataFrame]
type Strategies = polars.LazyDataFrame

type FieldName = str 
type FieldValue = Any
