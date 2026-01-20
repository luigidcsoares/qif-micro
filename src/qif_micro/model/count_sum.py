# from enum import Enum, auto

# import polars as pl

# from scipy.special import gammaln

# from qif_micro import qif
# from qif_micro._internal.dataset import (
#     _extract_from_record,
#     _prepare_dataset,
#     _valid_columns
# )
# from qif_micro.datatypes import Channel, Joint, ProbabDist
# from qif_micro.model import raw_microdata

# class ProcessMethod(Enum):
#     AS_INT = auto()
#     CEIL = auto()
#     ROUND = auto()


# def _process_expr(method: ProcessMethod, col: str | pl.Expr) -> pl.Expr:
#     col = pl.col(col) if isinstance(col, str) else col
#     match method:
#         case ProcessMethod.AS_INT: return (col * 10).cast(int)
#         case ProcessMethod.CEIL: return col.ceil()
#         case ProcessMethod.ROUND: return col.round()


# def _build_ch(
#     dataset_with_hints: pl.LazyFrame,
#     count_col: str,
#     sum_col: str,
#     agg_col: str
# ) -> pl.LazyFrame:
#     n = pl.col(sum_col)
#     k = pl.col(count_col)
#     h = pl.col(agg_col)

#     log_p = (
#         (k - 1).log() - (n + 1).log() 
#         + gammaln(n + k - h - 1) - gammaln(n - h + 1)
#         + gammaln(n + 2) - gammaln(n + k)
#     )

#     ch_dist = dataset_with_hints.with_columns(
#         pl
#         .when(pl.col(count_col) == 1)
#         .then((pl.col(sum_col) == pl.col(agg_col)).cast(float))
#         .otherwise(log_p.exp())
#         .over(count_col, sum_col, agg_col)
#         .alias("p")
#     )

#     return ch_dist.filter(pl.col("p") > 0)



# def _fill_invalid(ch_dist: pl.LazyFrame) -> pl.LazyFrame:
#     _, ok = _valid_columns(ch_dist, ["p", "hint"])
#     assert ok

#     schema = ch_dist.collect_schema()
#     expr_remaining_p = (1 - pl.col("p").sum()).alias("p")
#     expr_null_hint = pl.lit(None, schema["hint"]).alias("hint")

#     ch_dist_invalid = (
#         ch_dist
#         .group_by(pl.exclude("p", "hint"))
#         .agg(expr_null_hint, expr_remaining_p)
#         .select(schema.keys())
#     )

#     ch_dist = pl.concat([ch_dist, ch_dist_invalid])
#     return ch_dist.filter(pl.col("p") > 0)


# def build(
#     dataset: pl.DataFrame | pl.LazyFrame,
#     owner_col: str,
#     agg_col: str,
#     split_col: str | None = None,
#     count_col: str = "count",
#     sum_col: str = "sum",
#     valid_hints: pl.DataFrame | pl.LazyFrame | None = None,
#     process_method: ProcessMethod = ProcessMethod.CEIL
# ) -> tuple[ProbabDist, Channel]:
#     """
#     The input to this function is a dataset that is result of
#     aggregating some original data by count and sum.

#     We pre-process the original aggregated sum so that it is an integer.
#     By default, we take the ceil of the sum. This can be controlled 
#     by the `post_process` parameter. 

#     We assume the adversary knows/will learn (from external sources) 
#     one of the values that have been aggregated into the sum. These
#     values are implicitly post-processed so that they are integers.

#     This function then returns the adversary's intermediate knowledge
#     (i.e., after the aggregated data has been observed) and the
#     channel that models the relation between each record and agg_col.

#     That is, this function models an attribute-inference attack, in
#     which the adversary's goal is to guess a target's (count, sum).

#     A record is usually considered to be 1-dimensional, but this can be
#     controlled via `split_col`, which must be one of the columns
#     that "glue" together subrecords into a n-dimensional record 
#     (e.g., transaction categories). In this case we consider that the
#     output of the channel (the adversary's extra knowledge) is a pair.

#     If the real data is known, this function accepts the valid hints
#     in the form of a dataframe listing hint values (agg_col and maybe split_col).
#     In this case we create a Null hint that represents all invalid hints.

#     ## Example

#     >>> import polars as pl
#     >>> from qif_micro import model
#     >>> dataset = pl.DataFrame({
#     ...     "uid": [0, 1, 2],
#     ...     "count": [2, 2, 3],
#     ...     "sum": [2, 2, 2]
#     ... })
#     >>> owner_col = "uid"
#     >>> agg_col = "agg_value"
#     >>> prior, ch = model.count_sum.build(
#     ...     dataset,
#     ...     owner_col,
#     ...     agg_col
#     ... )
#     >>> prior
#     shape: (2, 2)
#     ┌─────────────────┬──────────┐
#     │ record          ┆ p        │
#     │ ---             ┆ ---      │
#     │ list[struct[2]] ┆ f64      │
#     ╞═════════════════╪══════════╡
#     │ [{2,2}]         ┆ 0.666667 │
#     │ [{3,2}]         ┆ 0.333333 │
#     └─────────────────┴──────────┘
#     >>> ch
#     shape: (6, 3)
#     ┌─────────────────┬───────────┬──────────┐
#     │ record          ┆ hint      ┆ p        │
#     │ ---             ┆ ---       ┆ ---      │
#     │ list[struct[2]] ┆ struct[1] ┆ f64      │
#     ╞═════════════════╪═══════════╪══════════╡
#     │ [{2,2}]         ┆ {0}       ┆ 0.333333 │
#     │ [{2,2}]         ┆ {1}       ┆ 0.333333 │
#     │ [{2,2}]         ┆ {2}       ┆ 0.333333 │
#     │ [{3,2}]         ┆ {0}       ┆ 0.5      │
#     │ [{3,2}]         ┆ {1}       ┆ 0.333333 │
#     │ [{3,2}]         ┆ {2}       ┆ 0.166667 │
#     └─────────────────┴───────────┴──────────┘
#     """
#     hint_cols = [agg_col, split_col]
#     hint_cols = [col for col in hint_cols if col is not None]

#     if valid_hints is not None:
#         diff, ok = _valid_columns(valid_hints.lazy(), hint_cols)
#         if not ok: raise ValueError(f"Missing columns {diff} in hints")

#     record_cols = [split_col, count_col, sum_col]
#     record_cols = [col for col in record_cols if col is not None]

#     expr_sum_col = _process_expr(process_method, sum_col).alias(sum_col)

#     dataset = (
#         dataset.lazy()
#         .with_columns(expr_sum_col)
#         .pipe(_prepare_dataset, owner_col, record_cols)
#     )

#     expr_p = (pl.col("len") / pl.col("len").sum()).alias("p")

#     record_freq = dataset.group_by("record").agg(pl.len())
#     prior_dist = record_freq.select("record", expr_p)

#     prior = ProbabDist.from_polars(prior_dist, ["record"])

#     wide_records = (
#         dataset
#         .drop(owner_col).unique()
#         .pipe(_extract_from_record, record_cols)
#         .explode(record_cols)
#     )

#     expr_agg_col = pl.int_ranges(0, pl.col(sum_col) + 1).alias(agg_col)

#     hints = valid_hints.lazy() if valid_hints is not None else (
#         wide_records
#         .with_columns(expr_agg_col)
#         .select(hint_cols)
#         .unique()
#         .explode(agg_col)
#     )

#     predicate_agg = pl.col(sum_col) >= pl.col(agg_col)
#     predicate_split = pl.lit(True) if split_col is None else (
#         pl.col(split_col) == pl.col(f"{split_col}_right")
#     )

#     predicates = [predicate_agg, predicate_split]
#     records_with_hints = wide_records.join_where(hints, predicates)

#     expr_coalesce_split = pl.coalesce(f"^{split_col}$", pl.lit(0))
#     split_vals = (
#         records_with_hints
#         .select(expr_coalesce_split)
#         .unique()
#         .collect()
#         .to_series()
#     )

#     ch_dist_unnorm = pl.concat(
#         records_with_hints
#         .filter(expr_coalesce_split == v)
#         .pipe(_build_ch, count_col, sum_col, agg_col)

#         for v in split_vals
#     )

#     expr_record_len = (
#         pl.col("record")
#         .list.eval(pl.element().struct.field(count_col).sum())
#         .list.first()
#         .alias("record_len")
#     )

#     record_lens = (
#         dataset
#         .select("record")
#         .unique()
#         .with_columns(expr_record_len)
#     )

#     expr_hint = pl.struct(hint_cols).alias("hint")
#     expr_weight = pl.col(count_col) / pl.col("record_len")
#     expr_p = (pl.col("p") * expr_weight).alias("p")

#     maybe_fill_invalid = (
#         (lambda lf: lf) if valid_hints is None
#          else _fill_invalid
#      )

#     ch_dist = (
#         ch_dist_unnorm
#         .join(record_lens, on="record")
#         .select("record", expr_hint, expr_p)
#         .pipe(maybe_fill_invalid)
#     )

#     ch = Channel.from_polars(ch_dist, ["record"], ["hint"])
#     return prior, ch 


# def baseline(
#     dataset: pl.DataFrame | pl.LazyFrame,
#     owner_col: str,
#     agg_col: str,
#     split_col: str | None = None,
#     count_col: str = "count",
#     sum_col: str = "sum",
#     process_method: ProcessMethod = ProcessMethod.CEIL,
# ) -> tuple[ProbabDist, Channel, pl.LazyFrame, pl.LazyFrame]:
#     """
#     The input to this function is a dataset in the form of microdata.

#     This function then aggregates the original data to generate
#     two statistics: count and sum for some QID attributes (hints).

#     For performance reasons, we process the `agg_col` so that it is
#     an integer. This way, both the hint observed by the adversary
#     and the sum over `agg_col` are integers. This might underestimate
#     privacy risk a bit, but it is much cheaper to implement.
#     By default, we take the ceil of `agg_col`. This can be controlled by
#     the `process_method` parameter.

#     This returns the adversary's intermediate knowledge with respect
#     to the aggregated record (after the dataset has been observed)
#     and the channel that models the relation between each aggregated
#     record and the QID attributes. In other words, we assume that the
#     QID that the adversary learns is a piece of the aggregated value
#     that the adversary wants to infer.

#     A record is usually considered to be 1-dimensional, but this can be
#     controlled via `split_col`, which must be one of the columns
#     that "glue" together subrecords into a n-dimensional record 
#     (for example, categories of transactions). In this case we consider 
#     that the output of the channel (the adversary's extra knowledge) 
#     is a pair of values formed by `agg_col` and `split_col`.

#     ## Example

#     >>> import polars as pl
#     >>> from qif_micro import model
#     >>> dataset = pl.DataFrame({
#     ...     "uid": [0, 0, 1, 1, 2, 2, 2],
#     ...     "amount": [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
#     ...     "category": ["a", "a", "a", "a", "b", "b", "a"]
#     ... })
#     >>> owner_col = "uid"
#     >>> agg_col = "amount"
#     >>> split_col = "category"
#     >>> prior, ch = model.count_sum.baseline(
#     ...     dataset,
#     ...     owner_col,
#     ...     agg_col,
#     ...     split_col
#     ... )
#     >>> prior
#     shape: (2, 2)
#     ┌────────────────────────┬──────────┐
#     │ record                 ┆ p        │
#     │ ---                    ┆ ---      │
#     │ list[struct[3]]        ┆ f64      │
#     ╞════════════════════════╪══════════╡
#     │ [{"a",1,1}, {"b",2,1}] ┆ 0.333333 │
#     │ [{"a",2,2}]            ┆ 0.666667 │
#     └────────────────────────┴──────────┘
#     >>> ch
#     shape: (6, 3)
#     ┌────────────────────────┬───────────┬──────────┐
#     │ record                 ┆ hint      ┆ p        │
#     │ ---                    ┆ ---       ┆ ---      │
#     │ list[struct[3]]        ┆ struct[2] ┆ f64      │
#     ╞════════════════════════╪═══════════╪══════════╡
#     │ [{"a",1,1}, {"b",2,1}] ┆ {0,"b"}   ┆ 0.333333 │
#     │ [{"a",1,1}, {"b",2,1}] ┆ {1,"a"}   ┆ 0.333333 │
#     │ [{"a",1,1}, {"b",2,1}] ┆ {1,"b"}   ┆ 0.333333 │
#     │ [{"a",2,2}]            ┆ {0,"a"}   ┆ 0.25     │
#     │ [{"a",2,2}]            ┆ {1,"a"}   ┆ 0.5      │
#     │ [{"a",2,2}]            ┆ {2,"a"}   ┆ 0.25     │
#     └────────────────────────┴───────────┴──────────┘
#     """
#     hint_cols = [col for col in [agg_col, split_col] if col is not None]

#     expr_agg_col = _process_expr(process_method, agg_col).alias(agg_col)
#     prior, ch = raw_microdata.build(
#         dataset.lazy().with_columns(expr_agg_col.cast(int)),
#         owner_col,
#         hint_cols,
#         []
#     )

#     # TODO: Add remap to the qif api
#     agg_cols = [split_col, count_col, sum_col]
#     agg_cols = [col for col in agg_cols if col is not None]

#     group_cols = ["record", split_col]
#     group_cols = [col for col in group_cols if col is not None]

#     expr_count = pl.len().alias(count_col)
#     expr_sum = pl.col(agg_col).sum().alias(sum_col)

#     records = prior.dist.select("record").unique()
#     map_agg_records = (
#         records
#         # We first explode our records which are lists of structs,
#         # to group each inner struct by the corresponding split value.
#         .with_columns(pl.col("record").alias("agg_record"))
#         .explode("agg_record")
#         .select("record", pl.col("agg_record").struct.unnest())
#         # Now we group by index and possibly split_col,
#         # and construct the aggregated entries of each record
#         .group_by(group_cols).agg(expr_count, expr_sum)
#         # We sort by agg_cols to make the str repr deterministic.
#         .sort(agg_cols)
#         # Finally, we recreate the structs, now with aggregate info,
#         # and regroup them back as a record.
#         .select("record", pl.struct(agg_cols).alias("agg_record"))
#         .group_by("record").agg("agg_record")
#     )

#     joint = qif.push(prior, ch)
#     joint_dist_agg = (
#         joint.dist
#         .join(map_agg_records, on="record")
#         .group_by("agg_record", "hint")
#         .agg(pl.col("p").sum())
#         .select(pl.col("agg_record").alias("record"), "hint", "p")
#     )

#     joint_agg = Joint.from_polars(
#         joint_dist_agg,
#         joint.input,
#         joint.output
#     )

#     prior_agg, ch_agg = qif.push_back(joint_agg)
#     return prior_agg, ch_agg
