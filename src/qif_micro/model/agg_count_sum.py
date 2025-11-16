from enum import Enum, auto

import polars as pl
from scipy.special import gammaln

from qif_micro import qif
from qif_micro._internal.validation import _valid_columns
from qif_micro.datatypes import Channel, Joint, ProbabDist

class ProcessMethod(Enum):
    AS_INT = auto()
    CEIL = auto()
    ROUND = auto()


def _process_expr(method: ProcessMethod, col: pl.Expr) -> pl.Expr:
    match method:
        case ProcessMethod.AS_INT: return (col * 10).cast(int)
        case ProcessMethod.CEIL: return col.ceil()
        case ProcessMethod.ROUND: return col.round()


def _build_ch(
    dataset_with_hints: pl.LazyFrame,
    count_col: str,
    sum_col: str,
) -> pl.LazyFrame:
    n = pl.col(sum_col)
    k = pl.col(count_col)
    h = pl.col("hint")

    log_p = (
        (k - 1).log() - (n + 1).log() 
        + gammaln(n + k - h - 1) - gammaln(n - h + 1)
        + gammaln(n + 2) - gammaln(n + k)
    )

    ch_dist = dataset_with_hints.with_columns(
        pl
        .when(pl.col(count_col) == 1)
        .then((pl.col(sum_col) == pl.col("hint")).cast(float))
        .otherwise(log_p.exp())
        .over(count_col, sum_col, "hint")
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
    schema = ch_dist.collect_schema()
    assert "p" in schema.keys()
    assert "hint" in schema.keys()

    remaining_p = 1 - pl.col("p").sum()
    null_hint = pl.lit(None, schema["hint"])

    ch_dist_invalid = (
        ch_dist
        .group_by(pl.exclude("p", "hint"))
        .agg(null_hint.alias("hint"), remaining_p.alias("p"))
        .select(schema.keys())
    )

    return pl.concat([ch_dist, ch_dist_invalid])


def build(
    dataset: pl.DataFrame | pl.LazyFrame,
    owner_col: str,
    count_col: str,
    sum_col: str,
    split_col: str | None = None,
    valid_hints: pl.DataFrame | pl.LazyFrame | None = None,
    pre_process: ProcessMethod = ProcessMethod.CEIL
) -> tuple[ProbabDist, Channel]:
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

    That is, this function models an colibute-inference attack, in
    which the adversary's goal is to guess a target's (count, sum).

    A record is usually considered to be 1-dimensional, but this can be
    controlled via `split_col`, which must be one of the columns
    that "glue" together subrecords into a n-dimensional record 
    (for example, transaction categories). In this case we consider that
    the output of the channel (the adversary's extra knowledge) is a
    pair of values.

    If the real data is known, this function accepts the valid hints 
    in the form of a dataframe mapping `owner_col` to each hint value
    (including `split_col`, if defined). In this case we create a Null
    hint that represents all invalid hint values. This is equivalent 
    to a post-processing that maps a subset of hints to Null.

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> dataset = pl.DataFrame({
    ...     "uid": [0, 1, 2],
    ...     "count": [2, 2, 3],
    ...     "sum": [2, 2, 2]
    ... })
    >>> owner_col = "uid"
    >>> count_col = "count"
    >>> sum_col = "sum"
    >>> prior, ch = model.agg_count_sum.build(
    ...     dataset,
    ...     owner_col,
    ...     count_col,
    ...     sum_col
    ... )
    >>> prior
    shape: (2, 3)
    ┌───────────┬───────────┬──────────┐
    │ count     ┆ sum       ┆ p        │
    │ ---       ┆ ---       ┆ ---      │
    │ list[i64] ┆ list[i64] ┆ f64      │
    ╞═══════════╪═══════════╪══════════╡
    │ [2]       ┆ [2]       ┆ 0.666667 │
    │ [3]       ┆ [2]       ┆ 0.333333 │
    └───────────┴───────────┴──────────┘
    >>> ch
    shape: (6, 4)
    ┌───────────┬───────────┬───────────┬──────────┐
    │ count     ┆ sum       ┆ hint      ┆ p        │
    │ ---       ┆ ---       ┆ ---       ┆ ---      │
    │ list[i64] ┆ list[i64] ┆ struct[1] ┆ f64      │
    ╞═══════════╪═══════════╪═══════════╪══════════╡
    │ [2]       ┆ [2]       ┆ {0.0}     ┆ 0.333333 │
    │ [2]       ┆ [2]       ┆ {1.0}     ┆ 0.333333 │
    │ [2]       ┆ [2]       ┆ {2.0}     ┆ 0.333333 │
    │ [3]       ┆ [2]       ┆ {0.0}     ┆ 0.5      │
    │ [3]       ┆ [2]       ┆ {1.0}     ┆ 0.333333 │
    │ [3]       ┆ [2]       ┆ {2.0}     ┆ 0.166667 │
    └───────────┴───────────┴───────────┴──────────┘
    """
    # If there's no column to split, we create a temporary one,
    # just so we can reuse the same logic...
    split_col = "tmp" if split_col is None else split_col

    # `split_col` must come first, for sorting!
    ext_input_cols = [split_col, count_col, sum_col]

    dataset = dataset.lazy().with_columns(
        pl.coalesce(f"^{split_col}$", pl.lit(""))
        .alias(split_col)
    )

    expected_cols = [owner_col, *ext_input_cols]
    diff, ok = _valid_columns(dataset, expected_cols)
    if not ok: raise ValueError(f"Missing columns {diff}")

    dataset = dataset.with_columns(
        _process_expr(pre_process, pl.col(sum_col))
        .alias(sum_col)
    )

    hints = valid_hints.lazy() if valid_hints is not None else (
        dataset
        .unique(ext_input_cols)
        .with_columns(pl.int_ranges(0, pl.col(sum_col)+1).alias("hint"))
        .explode("hint")
        .with_columns(pl.col("hint").cast(float))
    )

    # TODO: allow another name for the hint column?
    expected_cols = [owner_col, split_col, "hint"]
    expected_cols = [col for col in expected_cols if col != "tmp"]
    diff, ok = _valid_columns(hints, expected_cols)
    if not ok: raise ValueError(f"Missing columns {diff} in hints")

    # Iterate over each of the possible values for `split_col`,
    # construct individual channels, and then combine them, weighting by
    # the proportion of individual values for each `split_col` val.
    records = (
        dataset
        .sort(ext_input_cols)
        .select(owner_col, pl.struct(ext_input_cols).alias("record"))
        .group_by(owner_col, maintain_order=True)
        .agg("record")
    )

    el_field = lambda field: pl.element().struct.field(field)
    subrecord_filter = lambda val: el_field(split_col) == val
    extract_field_at = lambda field, at: (
        pl.col("record")
        .list.filter(subrecord_filter(at))
        .list.first()
        .struct.field(field)
    )

    dataset_with_hints = records.join(hints, on=owner_col)
    prepare_dataset = lambda v: (
        dataset_with_hints
        .filter(pl.col(split_col) == v)
        .with_columns(
            extract_field_at(count_col, v).alias(count_col),
            extract_field_at(sum_col, v).alias(sum_col),
        )
    )

    split_vals = dataset.select(pl.col(split_col).unique())
    extract_field = lambda field: (
        pl.col("record")
        .list.eval(el_field(field))
    )

    # `split_col` is both input, as part of the record, and output,
    # as part of the hint. So, we first deal with the hint side,
    # and then extract the values from the record col
    input_cols = [col for col in ext_input_cols if col != "tmp"]
    output_cols = [col for col in ["hint", split_col] if col != "tmp"]

    ch_dist = pl.concat(
        _build_ch(prepare_dataset(v), count_col, sum_col)
        for v in split_vals.collect().to_series()
    ).with_columns(
        pl.col("record")
        .list.eval(el_field(count_col))
        .list.sum()
        .alias("total_count"),

        (pl.col("p") * pl.col(count_col)).alias("p_scaled")
    ).select(
        *[extract_field(col).alias(col) for col in input_cols],

        pl.struct(output_cols)
        .struct.rename_fields(["hint_value", "hint_split"])
        .alias("hint"),

        (pl.col("p_scaled") / pl.col("total_count")).alias("p")
    )

    ch_dist = _fill_invalid(ch_dist, valid_hints is not None)

    prior_freq = records.group_by("record").agg(pl.len())
    prior_dist = prior_freq.select(
        *[extract_field(col).alias(col) for col in input_cols],
        (pl.col("len") / pl.col("len").sum()).alias("p")
    )

    prior = ProbabDist.from_polars(prior_dist, input_cols)
    ch = Channel.from_polars(ch_dist, input_cols, ["hint"])

    return prior, ch


def baseline(
    dataset: pl.DataFrame | pl.LazyFrame,
    owner_col: str,
    count_col: str,
    sum_col: str,
    agg_col: str,
    split_col: str | None = None,
    pre_process: ProcessMethod = ProcessMethod.CEIL,
    post_process: ProcessMethod = ProcessMethod.CEIL
) -> tuple[ProbabDist, Channel]:
    """
    The input to this function is a dataset in the form of microdata.

    This function then aggregates the original data to generate
    two statistics: count and sum for some QID colibute.

    We pre-process the original aggregated sum so that it is an integer,
    and we post-process the `agg_col` so that it is also an integer.
    By default, we take the ceil of these numbers. This can be
    controlled by the `pre_process` and `post_process` parameters.

    This returns the adversary's intermediate knowledge with respect
    to the aggregated record (after the dataset has been observed)
    and the channel that models the relation between each aggregated
    record and the QID colibute.

    We assume that the prior on raw records (group of rows per user)
    is uniform, as beforehand the adversary would have not reason
    to chosen one record over another. The prior on aggregated records
    is then the number of users whose raw record maps to the same
    aggregated records, normalised by the number of records.

    In other words, we assume that the QID that the adversary learns
    is a piece of the aggregated value that the adversary wants to infer.

    A record is usually considered to be 1-dimensional, but this can be
    controlled via `split_col`, which must be one of the columns
    that "glue" together subrecords into a n-dimensional record 
    (for example, transaction categories). In this case we consider that
    the output of the channel (the adversary's extra knowledge) is a
    pair of values.

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> dataset = pl.DataFrame({
    ...     "uid": [0, 0, 1, 1, 2, 2, 2],
    ...     "transaction_cost": [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0]
    ... })
    >>> owner_col = "uid"
    >>> count_col = "count"
    >>> sum_col = "sum"
    >>> agg_col = "transaction_cost"
    >>> prior, ch = model.agg_count_sum.baseline(
    ...     dataset,
    ...     owner_col,
    ...     count_col,
    ...     sum_col,
    ...     agg_col
    ... )
    >>> prior
    shape: (2, 3)
    ┌───────────┬───────────┬──────────┐
    │ count     ┆ sum       ┆ p        │
    │ ---       ┆ ---       ┆ ---      │
    │ list[u32] ┆ list[f64] ┆ f64      │
    ╞═══════════╪═══════════╪══════════╡
    │ [2]       ┆ [2.0]     ┆ 0.666667 │
    │ [3]       ┆ [2.0]     ┆ 0.333333 │
    └───────────┴───────────┴──────────┘
    >>> ch
    shape: (5, 4)
    ┌───────────┬───────────┬───────────┬──────────┐
    │ count     ┆ sum       ┆ hint      ┆ p        │
    │ ---       ┆ ---       ┆ ---       ┆ ---      │
    │ list[u32] ┆ list[f64] ┆ struct[1] ┆ f64      │
    ╞═══════════╪═══════════╪═══════════╪══════════╡
    │ [2]       ┆ [2.0]     ┆ {0.0}     ┆ 0.25     │
    │ [2]       ┆ [2.0]     ┆ {1.0}     ┆ 0.5      │
    │ [2]       ┆ [2.0]     ┆ {2.0}     ┆ 0.25     │
    │ [3]       ┆ [2.0]     ┆ {0.0}     ┆ 0.333333 │
    │ [3]       ┆ [2.0]     ┆ {1.0}     ┆ 0.666667 │
    └───────────┴───────────┴───────────┴──────────┘
    """
    hint_cols = [col for col in [agg_col, split_col] if col is not None]
    dataset = dataset.lazy()

    expected_cols = [owner_col, *hint_cols]
    diff, ok = _valid_columns(dataset, expected_cols)
    if not ok: raise ValueError(f"Missing columns {diff}")

    dataset_agg = dataset.group_by(owner_col, split_col).agg(
        pl.len().alias(count_col),

        _process_expr(pre_process, pl.col(agg_col).sum())
        .alias(sum_col)
    )

    # `split_col` must come first, for sorting!
    input_cols = (
        [] if split_col is None else [split_col]
    ) + [count_col, sum_col]

    struct_expr = pl.struct(input_cols)
    records = (
        dataset_agg
        .sort(input_cols)
        .select(owner_col, struct_expr.alias("record"))
        .group_by(owner_col, maintain_order=True)
        .agg("record")
    )

    el_field = lambda field: pl.element().struct.field(field)
    extract_field = lambda field: (
        pl.col("record")
        .list.eval(el_field(field))
    )

    prior_dist_owners = (
        records
        .group_by(owner_col, "record")
        .agg(pl.len())
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("p"))
    )

    record_len = (
        pl.col("record")
        .list.eval(el_field(count_col))
        .list.sum()
    )

    process_agg = _process_expr(post_process, pl.col(agg_col))
    ch_dist_owners = (
        records
        .join(dataset, on=owner_col, how="inner")
        .with_columns(process_agg.alias(agg_col))
        .group_by(owner_col, "record", *hint_cols)
        .agg(pl.len())
        .with_columns((pl.col("len") / record_len).alias("p"))
    )

    prior_owners = ProbabDist.from_polars(
        prior_dist_owners, 
        [owner_col, "record"]
    )

    ch_owners = Channel.from_polars(
        ch_dist_owners, 
        [owner_col, "record"],
        hint_cols
    )

    joint_owners = qif.push(prior_owners, ch_owners)
    joint_dist_agg = (
        joint_owners.dist
        .group_by("record", *hint_cols)
        .agg(pl.col("p").sum().alias("p"))
        .select(
            *[extract_field(a).alias(a) for a in input_cols],

            pl.struct(hint_cols)
            .struct.rename_fields(["hint_value", "hint_split"])
            .alias("hint"),

            "p"
        )
    )

    joint_agg = Joint.from_polars(joint_dist_agg, input_cols, ["hint"])
    return qif.push_back(joint_agg)
