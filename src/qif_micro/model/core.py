# import polars as pl

# from qif_micro.datatypes import channel, LazyChannel
# from qif_micro.datatypes import probab_dist, LazyProbabDist
# from qif_micro._internal import (
#      _prepare_hints,
#      _prepare_records,
#      _valid_columns   
# )

# def hint_ch(
#     domain_records: pl.DataFrame | pl.LazyFrame,
#     domain_hints: pl.DataFrame | pl.LazyFrame,
# ) -> LazyChannel:
#     """
#     ## Example

#     >>> import polars as pl
#     >>> from qif_micro import model
#     >>> from qif_micro.datatypes import channel

#     Consider the following example with attributes `q0`, `q1`, `s0`:

#     >>> domain_q0 = pl.LazyFrame({ "q0": [0, 1, 2] })
#     >>> domain_q1 = pl.LazyFrame({ "q1": [0, 1, 2] })
#     >>> domain_s0 = pl.LazyFrame({ "s0": [0, 1, 2] })

#     Let us assume that records may have one or two entries:
    
#     >>> _ = domain_q0.join(domain_q1, how="cross")
#     >>> _ = _.join(domain_s0, how="cross")
#     >>> domain_hints = _.select(pl.struct("q0", "q1").alias("hint").unique())
#     >>> domain_records1 = _.select(pl.concat_list(pl.struct("q0", "q1", "s0").alias("record").unique()))
#     >>> domain_records2 = domain_records1.join(domain_records1, how="cross").select(pl.concat_list(pl.all()))
#     >>> domain_records = pl.concat([domain_records1, domain_records2])

#     Then the hint channel is
    
#     >>> ch = model.hint_ch(domain_records, domain_hints)
#     >>> channel.collect(ch)
#     shape: (1_404, 3)
#     ┌────────────────────┬───────────┬─────┐
#     │ record             ┆ hint      ┆ p   │
#     │ ---                ┆ ---       ┆ --- │
#     │ list[struct[3]]    ┆ struct[2] ┆ f64 │
#     ╞════════════════════╪═══════════╪═════╡
#     │ [{0,0,0}]          ┆ {0,0}     ┆ 1.0 │
#     │ [{0,0,0}, {0,0,0}] ┆ {0,0}     ┆ 1.0 │
#     │ [{0,0,0}, {0,0,1}] ┆ {0,0}     ┆ 1.0 │
#     │ [{0,0,0}, {0,0,2}] ┆ {0,0}     ┆ 1.0 │
#     │ [{0,0,0}, {0,1,0}] ┆ {0,0}     ┆ 0.5 │
#     │ …                  ┆ …         ┆ …   │
#     │ [{2,1,2}, {2,2,2}] ┆ {2,1}     ┆ 0.5 │
#     │ [{2,1,2}, {2,2,2}] ┆ {2,2}     ┆ 0.5 │
#     │ [{2,2,0}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
#     │ [{2,2,1}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
#     │ [{2,2,2}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
#     └────────────────────┴───────────┴─────┘
#     """
#     # ==================================================
#     # Pre-conditions: valid both domains first
#     # ==================================================
#     domain_records = _prepare_records(domain_records)
#     domain_hints = _prepare_hints(domain_hints)

#     entry_schema = domain_records.collect_schema()["record"].inner
#     hint_schema = domain_hints.collect_schema()["hint"]

#     record_attrs = list(entry_schema.to_schema().keys())
#     hint_attrs = list(hint_schema.to_schema().keys())

#     diff = set(hint_attrs) - set(record_attrs)
#     if len(diff) != 0:
#         raise ValueError("Mismatch between hint and record attributes")

#     # ==================================================
#     # Finished pre-conditions
#     # ==================================================

#     extract_single_expr = pl.struct(pl.element().struct.field(hint_attrs))
#     extract_all_expr = pl.col("record").list.eval(extract_single_expr)

#     _ = domain_records.with_columns(extract_all_expr.alias("hint"))
#     _ = _.explode("hint").join(domain_hints, on="hint")
#     _ = _.group_by("record", "hint").agg(pl.len().alias("p"))

#     ch_dist = _.with_columns(pl.col("p") / pl.col("record").list.len())
#     return channel.make_lazy(ch_dist, ["record"], ["hint"])


# def baseline(dataset: pl.DataFrame | pl.LazyFrame) -> LazyProbabDist:
#     """
#     This functions takes the real (de-identified) dataset
#     and constructs the adversary's intermediate knowledge on records
#     upon observing the real dataset. This is computed following a
#     frequestist approach, meaning that for each record we count
#     how many times that record occurs in the dataset.

#     ## Example
    
#     # >>> import polars as pl
#     # >>> from qif_micro import model
#     # >>> dataset = pl.DataFrame({
#     # ...     "record_id": [0, 0, 1, 1, 2, 2, 2],
#     # ...     "amount": [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
#     # ... })
#     # >>> mid_knowledge = model.baseline(dataset)
#     # >>> probab_dist.collect(mid_knowledge)
#     # shape: (3, 2)
#     # ┌───────────────────────┬──────────┐
#     # │ record                ┆ p        │
#     # │ ---                   ┆ ---      │
#     # │ list[struct[1]]       ┆ f64      │
#     # ╞═══════════════════════╪══════════╡
#     # │ [{0.0}, {1.0}, {1.0}] ┆ 0.333333 │
#     # │ [{0.0}, {2.0}]        ┆ 0.333333 │
#     # │ [{1.0}, {1.0}]        ┆ 0.333333 │
#     # └───────────────────────┴──────────┘
#     """
#     # ==================================================
#     # Pre-conditions: dataset must either be in "long"
#     #   format with rows tagged with a record_id,
#     #   or must have a single record column.
#     # ==================================================
#     dataset = _prepare_records(dataset)
    
#     p_expr = (pl.len() / pl.col("n_rows").first()).alias("p")
#     _ = dataset.with_columns(pl.len().alias("n_rows"))
#     dist = _.group_by("record").agg(p_expr)

#     return probab_dist.make_lazy(dist, ["record"])
