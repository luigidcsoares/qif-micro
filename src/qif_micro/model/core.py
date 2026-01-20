import polars as pl

from qif_micro.datatypes import channel, LazyChannel
from qif_micro._internal.dataset import _valid_columns

def build_hint_ch(
    domain_records: pl.DataFrame | pl.LazyFrame,
    domain_hints: pl.DataFrame | pl.LazyFrame,
) -> LazyChannel:
    """
    This function expects the following shapes for each domain:

    Domain of records:
    - A single column named `record`, with type List
    - The inner type of `record` must be a struct

    Domain of hints:
    - A single column named `hint`, with type Struct
    - All fields of `hint` must be fields of the inner type of `record`

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> from qif_micro.datatypes import channel

    Consider the following example with attributes `q0`, `q1`, `s0`:

    >>> domain_q0 = pl.LazyFrame({ "q0": [0, 1, 2] })
    >>> domain_q1 = pl.LazyFrame({ "q1": [0, 1, 2] })
    >>> domain_s0 = pl.LazyFrame({ "s0": [0, 1, 2] })

    Let us assume that records may have one or two entries:
    
    >>> _ = domain_q0.join(domain_q1, how="cross")
    >>> _ = _.join(domain_s0, how="cross")
    >>> domain_hints = _.select(pl.struct("q0", "q1").alias("hint").unique())
    >>> domain_records1 = _.select(pl.concat_list(pl.struct("q0", "q1", "s0").alias("record").unique()))
    >>> domain_records2 = domain_records1.join(domain_records1, how="cross").select(pl.concat_list(pl.all()))
    >>> domain_records = pl.concat([domain_records1, domain_records2])

    Then the hint channel is
    
    >>> ch = model.hint_ch(domain_records, domain_hints)
    >>> channel.collect(ch)
    shape: (1_404, 3)
    ┌────────────────────┬───────────┬─────┐
    │ record             ┆ hint      ┆ p   │
    │ ---                ┆ ---       ┆ --- │
    │ list[struct[3]]    ┆ struct[2] ┆ f64 │
    ╞════════════════════╪═══════════╪═════╡
    │ [{0,0,0}]          ┆ {0,0}     ┆ 1.0 │
    │ [{0,0,0}, {0,0,0}] ┆ {0,0}     ┆ 1.0 │
    │ [{0,0,0}, {0,0,1}] ┆ {0,0}     ┆ 1.0 │
    │ [{0,0,0}, {0,0,2}] ┆ {0,0}     ┆ 1.0 │
    │ [{0,0,0}, {0,1,0}] ┆ {0,0}     ┆ 0.5 │
    │ …                  ┆ …         ┆ …   │
    │ [{2,1,2}, {2,2,2}] ┆ {2,1}     ┆ 0.5 │
    │ [{2,1,2}, {2,2,2}] ┆ {2,2}     ┆ 0.5 │
    │ [{2,2,0}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
    │ [{2,2,1}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
    │ [{2,2,2}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
    └────────────────────┴───────────┴─────┘
    """
    # ==================================================
    # Pre-conditions: valid both domains first
    # ==================================================

    expected_cols = ["record"]
    diff, ok = _valid_columns(domain_records, expected_cols)
    if not ok: raise ValueError("Record domain missing `record`")

    record_schema = domain_records.collect_schema()["record"]
    if record_schema != pl.List:
        raise ValueError("`record` dtype must be list")

    entry_schema = record_schema.inner
    if entry_schema != pl.Struct:
        raise ValueError("`record` inner dtype must be struct")

    expected_cols = ["hint"]
    diff, ok = _valid_columns(domain_hints, expected_cols)
    if not ok: raise ValueError("Record domain missing `hint`")

    hint_schema = domain_hints.collect_schema()["hint"]
    if hint_schema != pl.Struct:
        raise ValueError("`hint` dtype must be a struct")

    record_attrs = list(entry_schema.to_schema().keys())
    hint_attrs = list(hint_schema.to_schema().keys())

    diff = set(hint_attrs) - set(record_attrs)
    if len(diff) != 0:
        raise ValueError("Mismatch between hint and record attributes")

    # ==================================================
    # Finished pre-conditions
    # ==================================================

    extract_single_expr = pl.struct(pl.element().struct.field(hint_attrs))
    extract_all_expr = pl.col("record").list.eval(extract_single_expr)

    _ = domain_records.with_columns(extract_all_expr.alias("hint"))
    _ = _.explode("hint").join(domain_hints, on="hint")
    _ = _.group_by("record", "hint").agg(pl.len().alias("p"))

    ch_dist = _.with_columns(pl.col("p") / pl.col("record").list.len())
    return channel.make_lazy(ch_dist, ["record"], ["hint"])
