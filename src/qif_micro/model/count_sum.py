from enum import Enum, auto

import polars as pl

from scipy.special import gammaln

from qif_micro import qif
from qif_micro._internal.dataset import (
    _extract_from_record,
    _prepare_dataset,
    _valid_columns
)
from qif_micro.datatypes import Channel, Joint, ProbabDist
from qif_micro.model import raw_microdata

class ProcessMethod(Enum):
    AS_INT = auto()
    CEIL = auto()
    ROUND = auto()


def _process_expr(method: ProcessMethod, col: str | pl.Expr) -> pl.Expr:
    col = pl.col(col) if isinstance(col, str) else col
    match method:
        case ProcessMethod.AS_INT: return (col * 10).cast(int)
        case ProcessMethod.CEIL: return col.ceil()
        case ProcessMethod.ROUND: return col.round()


def _build_ch(
    dataset_with_hints: pl.LazyFrame,
    count_col: str,
    sum_col: str,
    agg_col: str
) -> pl.LazyFrame:
    n = pl.col(sum_col)
    k = pl.col(count_col)
    h = pl.col(agg_col)

    log_p = (
        (k - 1).log() - (n + 1).log() 
        + gammaln(n + k - h - 1) - gammaln(n - h + 1)
        + gammaln(n + 2) - gammaln(n + k)
    )

    ch_dist = dataset_with_hints.with_columns(
        pl
        .when(pl.col(count_col) == 1)
        .then((pl.col(sum_col) == pl.col(agg_col)).cast(float))
        .otherwise(log_p.exp())
        .over(count_col, sum_col, agg_col)
        .alias("p")
    )

    return ch_dist.filter(pl.col("p") > 0)


def _fill_invalid(
    ch_dist: pl.LazyFrame,
    with_valid_hints: bool
) -> pl.LazyFrame:
    if not with_valid_hints: return ch_dist
    # If valid hints are available, then _build_ch only considers valid
    # hints as per the baseline. In this case we add a special hint to
    # keep the channel valid. This is equivalent to a post-processing
    # that merges all invalid hints.
    _, ok = _valid_columns(ch_dist, ["p", "hint"])
    assert ok

    schema = ch_dist.collect_schema()
    expr_remaining_p = (1 - pl.col("p").sum()).alias("p")
    expr_null_hint = pl.lit(None, schema["hint"]).alias("hint")

    ch_dist_invalid = (
        ch_dist
        .group_by(pl.exclude("p", "hint"))
        .agg(expr_null_hint, expr_remaining_p)
        .select(schema.keys())
    )

    return pl.concat([ch_dist, ch_dist_invalid])


def build(
    dataset: pl.DataFrame | pl.LazyFrame,
    owner_col: str,
    agg_col: str,
    split_col: str | None = None,
    count_col: str = "count",
    sum_col: str = "sum",
    map_hints: pl.DataFrame | pl.LazyFrame | None = None,
    process_method: ProcessMethod = ProcessMethod.CEIL
) -> tuple[ProbabDist, Channel, pl.LazyFrame, pl.LazyFrame]:
    """
    The input to this function is a dataset that is result of
    aggregating some original data by count and sum.

    We pre-process the original aggregated sum so that it is an integer.
    By default, we take the ceil of the sum. This can be controlled 
    by the `post_process` parameter. 

    We assume the adversary knows/will learn (from external sources) 
    one of values that have been aggregated into the sum. These values
    are implicitly post-processed so that they are integers.

    This function then returns the adversary's intermediate knowledge
    (i.e., after the aggregated data has been observed) and the
    channel that models the relation between each record and agg_col.

    That is, this function models an attribute-inference attack, in
    which the adversary's goal is to guess a target's (count, sum).

    A record is usually considered to be 1-dimensional, but this can be
    controlled via `split_col`, which must be one of the columns
    that "glue" together subrecords into a n-dimensional record 
    (for example, transaction categories). In this case we consider that
    the output of the channel (the adversary's extra knowledge) is a
    pair of values.

    If the real data is known, this function accepts the valid hints
    in the form of a dataframe listing (pairs of) hint values
    (same format as the `map_hints` returned from models). In this case
    we create a Null hint that represents all invalid hint values.

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> dataset = pl.DataFrame({
    ...     "uid": [0, 1, 2],
    ...     "count": [2, 2, 3],
    ...     "sum": [2, 2, 2]
    ... })
    >>> owner_col = "uid"
    >>> agg_col = "agg_value"
    >>> prior, ch, map_records, map_hints = model.count_sum.build(
    ...     dataset,
    ...     owner_col,
    ...     agg_col
    ... )
    >>> prior
    shape: (2, 2)
    ┌───────────┬──────────┐
    │ record_id ┆ p        │
    │ ---       ┆ ---      │
    │ u32       ┆ f64      │
    ╞═══════════╪══════════╡
    │ 1         ┆ 0.666667 │
    │ 2         ┆ 0.333333 │
    └───────────┴──────────┘
    >>> ch
    shape: (6, 3)
    ┌───────────┬─────────┬──────────┐
    │ record_id ┆ hint_id ┆ p        │
    │ ---       ┆ ---     ┆ ---      │
    │ u32       ┆ u32     ┆ f64      │
    ╞═══════════╪═════════╪══════════╡
    │ 1         ┆ 1       ┆ 0.333333 │
    │ 1         ┆ 2       ┆ 0.333333 │
    │ 1         ┆ 3       ┆ 0.333333 │
    │ 2         ┆ 1       ┆ 0.5      │
    │ 2         ┆ 2       ┆ 0.333333 │
    │ 2         ┆ 3       ┆ 0.166667 │
    └───────────┴─────────┴──────────┘
    >>> map_records.sort(["record_id", "record"]).collect()
    shape: (2, 2)
    ┌───────────┬─────────────────┐
    │ record_id ┆ record          │
    │ ---       ┆ ---             │
    │ u32       ┆ list[struct[2]] │
    ╞═══════════╪═════════════════╡
    │ 1         ┆ [{2,2}]         │
    │ 2         ┆ [{3,2}]         │
    └───────────┴─────────────────┘
    >>> map_hints.sort(["hint_id", "hint"]).collect()
    shape: (3, 2)
    ┌─────────┬───────────┐
    │ hint_id ┆ hint      │
    │ ---     ┆ ---       │
    │ u32     ┆ struct[1] │
    ╞═════════╪═══════════╡
    │ 1       ┆ {0}       │
    │ 2       ┆ {1}       │
    │ 3       ┆ {2}       │
    └─────────┴───────────┘
    """
    dataset = dataset.lazy()

    hint_cols = [agg_col, split_col]
    hint_cols = [col for col in hint_cols if col is not None]

    has_hints = map_hints is not None
    if has_hints:
        expected_cols = ["hint_id", "hint"]
        diff, ok = _valid_columns(map_hints.lazy(), expected_cols)
        if not ok: raise ValueError(f"Missing columns {diff} in hints")

    record_cols = [split_col, count_col, sum_col]
    record_cols = [col for col in record_cols if col is not None]
    expr_sum_col = _process_expr(process_method, sum_col).alias(sum_col)
    expr_rec_id = pl.col("record").rank("dense").alias("record_id")

    dataset = (
        dataset.with_columns(expr_sum_col)
        .pipe(_prepare_dataset, owner_col, record_cols)
        .select(owner_col, "record", expr_rec_id)
    )

    expr_p = (pl.col("len") / pl.col("len").sum()).alias("p")
    prior_dist = (
        dataset
        .group_by("record_id")
        .agg(pl.len())
        .select("record_id", expr_p)
    )

    prior = ProbabDist.from_polars(prior_dist, ["record_id"])

    wide_dataset = (
        _extract_from_record(dataset, record_cols)
        .select("record_id", *record_cols)
        .unique()
        .explode(record_cols)
    )

    expr_agg_col = pl.int_ranges(0, pl.col(sum_col) + 1).alias(agg_col)
    map_hints = map_hints.lazy().unique() if has_hints else (
        wide_dataset
        .with_columns(expr_agg_col)
        .select(hint_cols).unique().explode(hint_cols)
        .select(pl.struct(hint_cols).alias("hint"))
        .select(pl.col("hint").rank("dense").alias("hint_id"), "hint")
    )

    predicate_agg = pl.col(sum_col) >= pl.col(agg_col)
    predicate_split = pl.lit(True) if split_col is None else (
        pl.col(split_col) == pl.col(f"{split_col}_right")
    )

    unnested_hints = map_hints.select(pl.col("hint").struct.unnest())
    dataset_with_hints = wide_dataset.join_where(
        unnested_hints,
        [predicate_agg, predicate_split]
    )

    expr_coalesce_split = pl.coalesce(f"^{split_col}$", pl.lit(0))
    split_vals = (
        dataset_with_hints.select(expr_coalesce_split)
        .unique().collect().to_series()
    )

    ch_dist_unnorm = pl.concat(
        dataset_with_hints.filter(expr_coalesce_split == v)
        .pipe(_build_ch, count_col, sum_col, agg_col)

        for v in split_vals
    )

    expr_record_len = (
        pl.col("record")
        .list.eval(pl.element().struct.field(count_col).sum())
        .list.first()
        .alias("record_len")
    )

    map_records = dataset.select("record_id", "record").unique()
    record_lens = map_records.select("record_id", expr_record_len)

    expr_hint = pl.struct(hint_cols).alias("hint")
    expr_weight = pl.col(count_col) / pl.col("record_len")
    expr_p = (pl.col("p") * expr_weight).alias("p")

    ch_dist = (
        ch_dist_unnorm
        .join(record_lens, on="record_id")
        .select("record_id", expr_hint, expr_p)
        .pipe(_fill_invalid, has_hints)
        .join(map_hints, on="hint", how="left")
        # .with_columns(pl.col("hint_id").fill_null(-1))
    )

    map_hints = ch_dist.select("hint_id", "hint").unique()
    ch_dist = ch_dist.select("record_id", "hint_id", "p")
    ch = Channel.from_polars(ch_dist, ["record_id"], ["hint_id"])
    return prior, ch, map_records, map_hints


def baseline(
    dataset: pl.DataFrame | pl.LazyFrame,
    owner_col: str,
    agg_col: str,
    split_col: str | None = None,
    count_col: str = "count",
    sum_col: str = "sum",
    process_method: ProcessMethod = ProcessMethod.CEIL,
) -> tuple[ProbabDist, Channel, pl.LazyFrame, pl.LazyFrame]:
    """
    The input to this function is a dataset in the form of microdata.

    This function then aggregates the original data to generate
    two statistics: count and sum for some QID attributes (hints).

    For performance reasons, we process the `agg_col` so that it is
    an integer. This way, both the hint observed by the adversary
    and the sum over `agg_col` are integers. This might underestimate
    privacy risk a bit, but it is much cheaper to implement.
    By default, we take the ceil of `agg_col`. This can be controlled by
    the `process_method` parameter.

    This returns the adversary's intermediate knowledge with respect
    to the aggregated record (after the dataset has been observed)
    and the channel that models the relation between each aggregated
    record and the QID attributes. In other words, we assume that the
    QID that the adversary learns is a piece of the aggregated value
    that the adversary wants to infer.

    A record is usually considered to be 1-dimensional, but this can be
    controlled via `split_col`, which must be one of the columns
    that "glue" together subrecords into a n-dimensional record 
    (for example, categories of transactions). In this case we consider 
    that the output of the channel (the adversary's extra knowledge) 
    is a pair of values formed by `agg_col` and `split_col`.

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> dataset = pl.DataFrame({
    ...     "uid": [0, 0, 1, 1, 2, 2, 2],
    ...     "amount": [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
    ...     "category": ["a", "a", "a", "a", "b", "b", "a"]
    ... })
    >>> owner_col = "uid"
    >>> agg_col = "amount"
    >>> split_col = "category"
    >>> prior, ch, map_records, map_hints, = model.count_sum.baseline(
    ...     dataset,
    ...     owner_col,
    ...     agg_col,
    ...     split_col
    ... )
    >>> prior
    shape: (2, 2)
    ┌───────────┬──────────┐
    │ record_id ┆ p        │
    │ ---       ┆ ---      │
    │ u32       ┆ f64      │
    ╞═══════════╪══════════╡
    │ 1         ┆ 0.333333 │
    │ 2         ┆ 0.666667 │
    └───────────┴──────────┘
    >>> ch
    shape: (6, 3)
    ┌───────────┬─────────┬──────────┐
    │ record_id ┆ hint_id ┆ p        │
    │ ---       ┆ ---     ┆ ---      │
    │ u32       ┆ u32     ┆ f64      │
    ╞═══════════╪═════════╪══════════╡
    │ 1         ┆ 2       ┆ 0.333333 │
    │ 1         ┆ 3       ┆ 0.333333 │
    │ 1         ┆ 4       ┆ 0.333333 │
    │ 2         ┆ 1       ┆ 0.25     │
    │ 2         ┆ 3       ┆ 0.5      │
    │ 2         ┆ 5       ┆ 0.25     │
    └───────────┴─────────┴──────────┘
    >>> map_records.sort(["record_id", "record"]).collect()
    shape: (2, 2)
    ┌───────────┬────────────────────────┐
    │ record_id ┆ record                 │
    │ ---       ┆ ---                    │
    │ u32       ┆ list[struct[3]]        │
    ╞═══════════╪════════════════════════╡
    │ 1         ┆ [{"a",1,1}, {"b",2,1}] │
    │ 2         ┆ [{"a",2,2}]            │
    └───────────┴────────────────────────┘
    >>> map_hints.sort(["hint_id", "hint"]).collect()
    shape: (5, 2)
    ┌─────────┬───────────┐
    │ hint_id ┆ hint      │
    │ ---     ┆ ---       │
    │ u32     ┆ struct[2] │
    ╞═════════╪═══════════╡
    │ 1       ┆ {0,"a"}   │
    │ 2       ┆ {0,"b"}   │
    │ 3       ┆ {1,"a"}   │
    │ 4       ┆ {1,"b"}   │
    │ 5       ┆ {2,"a"}   │
    └─────────┴───────────┘
    """
    hint_cols = [col for col in [agg_col, split_col] if col is not None]

    expr_agg_col = _process_expr(process_method, agg_col).alias(agg_col)
    prior, ch, map_records, map_hints = raw_microdata.build(
        dataset.lazy().with_columns(expr_agg_col.cast(int)),
        owner_col,
        hint_cols,
        []
    )

    agg_cols = [split_col, count_col, sum_col]
    agg_cols = [col for col in agg_cols if col is not None]

    group_cols = ["record_id", split_col]
    group_cols = [col for col in group_cols if col is not None]

    expr_count = pl.len().alias(count_col)
    expr_sum = pl.col(agg_col).sum().alias(sum_col)

    map_agg_records = (
        map_records
        .explode("record")
        .select(pl.col("record").struct.unnest(), "record_id")
        .group_by(group_cols)
        .agg(expr_count, expr_sum)
        .sort(agg_cols)
        .select("record_id", pl.struct(agg_cols).alias("record"))
        .group_by("record_id")
        .agg("record")
        .with_columns(pl.col("record").rank("dense").alias("agg_record_id"))
    )

    joint = qif.push(prior, ch)
    joint_dist_agg = (
        joint.dist
        .join(map_agg_records.drop("record").unique(), on="record_id")
        .group_by("agg_record_id", "hint_id")
        .agg(pl.col("p").sum())
        .rename({"agg_record_id": "record_id"})
    )

    prior_agg, ch_agg = qif.push_back(
        Joint.from_polars(joint_dist_agg, joint.input, joint.output)
    )

    map_agg_records = map_agg_records.select(
        pl.col("agg_record_id").alias("record_id"),
        "record"
    ).unique()

    return prior_agg, ch_agg, map_agg_records, map_hints
