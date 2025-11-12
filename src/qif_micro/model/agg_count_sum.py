from collections.abc import Iterable
from enum import Enum, auto

import polars as pl
from scipy.special import gammaln

from qif_micro import qif
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


# TODO: move to some utils internal lib
def _valid_columns(
    lf: pl.LazyFrame,
    required: Iterable[str]
) -> tuple[set[str], bool]:
    missing = set(required) - set(lf.collect_schema().names())
    return missing, len(missing) == 0


def _build_ch(
    dataset: pl.LazyFrame,
    count_attr: str,
    sum_attr: str
) -> pl.LazyFrame:
    n = pl.col(sum_attr)
    k = pl.col(count_attr)
    h = pl.col("hint")

    log_p = (
        (k - 1).log() - (n + 1).log() 
        + gammaln(n + k - h - 1) - gammaln(n - h + 1)
        + gammaln(n + 2) - gammaln(n + k)
    )

    dataset_with_hints = (
        dataset
        .unique()
        .with_columns(hint=pl.int_ranges(0, pl.col(sum_attr) + 1))
        .explode("hint")
        .with_columns(hint=pl.col("hint").cast(float))
    )

    ch_dist = dataset_with_hints.with_columns(
        pl
        .when(pl.col(count_attr) == 1)
        .then((pl.col(sum_attr) == pl.col("hint")).cast(float))
        .otherwise(log_p.exp())
        .alias("p")
    )

    return ch_dist.filter(pl.col("p") > 0)


def build(
    dataset: pl.LazyFrame | pl.DataFrame,
    owner_attr: str,
    count_attr: str,
    sum_attr: str,
    split_attr: str | None = None,
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
    channel that models the relation between each record and agg_attr.

    That is, this function models an attribute-inference attack, in
    which the adversary's goal is to guess a target's (count, sum).

    A record is usually considered to be 1-dimensional, but this can be
    controlled via `split_attr`, which must be one of the columns
    that "glue" together subrecords into a n-dimensional record 
    (for example, transaction categories). In this case we consider that
    the output of the channel (the adversary's extra knowledge) is a
    pair of values.

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> dataset = pl.DataFrame({
    ...     "uid": [0, 1, 2],
    ...     "count": [2, 2, 3],
    ...     "sum": [2, 2, 2]
    ... })
    >>> owner_attr = "uid"
    >>> count_attr = "count"
    >>> sum_attr = "sum"
    >>> prior, ch = model.agg_count_sum.build(
    ...     dataset,
    ...     owner_attr,
    ...     count_attr,
    ...     sum_attr
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
    ┌───────────┬───────────┬──────┬──────────┐
    │ count     ┆ sum       ┆ hint ┆ p        │
    │ ---       ┆ ---       ┆ ---  ┆ ---      │
    │ list[i64] ┆ list[i64] ┆ f64  ┆ f64      │
    ╞═══════════╪═══════════╪══════╪══════════╡
    │ [2]       ┆ [2]       ┆ 0.0  ┆ 0.333333 │
    │ [2]       ┆ [2]       ┆ 1.0  ┆ 0.333333 │
    │ [2]       ┆ [2]       ┆ 2.0  ┆ 0.333333 │
    │ [3]       ┆ [2]       ┆ 0.0  ┆ 0.5      │
    │ [3]       ┆ [2]       ┆ 1.0  ┆ 0.333333 │
    │ [3]       ┆ [2]       ┆ 2.0  ┆ 0.166667 │
    └───────────┴───────────┴──────┴──────────┘
    """
    # If there's no column to split, we create a temporary one,
    # just so we can reuse the same logic...
    split_attr = "tmp" if split_attr is None else split_attr

    # `split_attr` must come first, for sorting!
    ext_input_attrs = [split_attr, count_attr, sum_attr]

    dataset = dataset.lazy().with_columns(
        pl.coalesce(f"^{split_attr}$", pl.lit(""))
        .alias(split_attr)
    )

    expected_cols = [owner_attr, *ext_input_attrs]
    diff, valid = _valid_columns(dataset, expected_cols)
    if not valid: raise ValueError(f"Missing columns {diff}")

    dataset = dataset.with_columns(
        _process_expr(pre_process, pl.col(sum_attr))
        .alias(sum_attr)
    )

    # Iterate over each of the possible values for `split_attr`,
    # construct individual channels, and then combine them, weighting by
    # the proportion of individual values for each `split_attr` val.
    struct_expr = pl.struct(ext_input_attrs)

    records = (
        dataset
        .sort(ext_input_attrs)
        .select(owner_attr, struct_expr.alias("record"))
        .group_by(owner_attr, maintain_order=True)
        .agg("record")
        .select("record")
    )

    el_field = lambda field: pl.element().struct.field(field)
    subrecord_filter = lambda val: el_field(split_attr) == val
    extract_field_at = lambda field, at: (
        pl.col("record")
        .list.filter(subrecord_filter(at))
        .list.first()
        .struct.field(field)
    )

    build_ch_with_split =  lambda v: (
        _build_ch(records.with_columns(
            extract_field_at(count_attr, v).alias(count_attr),
            extract_field_at(sum_attr, v).alias(sum_attr),
        ).drop_nulls(), count_attr, sum_attr)
        .with_columns(pl.lit(v).alias("hint_" + split_attr))
    )

    split_vals = dataset.select(pl.col(split_attr).unique())
    extract_field = lambda field: pl.col("record").list.eval(
        el_field(field)
    )

    input_attrs = [attr for attr in ext_input_attrs if attr != "tmp"]
    output_attrs = (
        ["hint"] if split_attr == "tmp" 
        else ["hint", "hint_" + split_attr]
    )

    ch_dist = pl.concat(
        build_ch_with_split(v)
        for v in split_vals.collect().to_series()
    ).with_columns(
        pl.col("record")
        .list.eval(el_field(count_attr))
        .list.sum()
        .alias("total_count"),

        (pl.col("p") * pl.col(count_attr)).alias("p_scaled")
    ).select(
        *[extract_field(attr).alias(attr) for attr in input_attrs],
        *output_attrs,
        (pl.col("p_scaled") / pl.col("total_count")).alias("p")
    )

    prior_freq = records.group_by("record").agg(pl.len())
    prior_dist = prior_freq.select(
        *[extract_field(attr).alias(attr) for attr in input_attrs],
        (pl.col("len") / pl.col("len").sum()).alias("p")
    )

    prior = ProbabDist.from_polars(prior_dist, input_attrs)
    ch = Channel.from_polars(ch_dist, input_attrs, output_attrs)

    return prior, ch


def baseline(
    dataset: pl.DataFrame | pl.LazyFrame,
    owner_attr: str,
    count_attr: str,
    sum_attr: str,
    agg_attr: str,
    split_attr: str | None = None,
    pre_process: ProcessMethod = ProcessMethod.CEIL,
    post_process: ProcessMethod = ProcessMethod.CEIL
) -> tuple[ProbabDist, Channel]:
    """
    The input to this function is a dataset in the form of microdata.

    This function then aggregates the original data to generate
    two statistics: count and sum for some QID attribute.

    We pre-process the original aggregated sum so that it is an integer,
    and we post-process the `agg_attr` so that it is also an integer.
    By default, we take the ceil of these numbers. This can be
    controlled by the `pre_process` and `post_process` parameters.

    This returns the adversary's intermediate knowledge with respect
    to the aggregated record (after the dataset has been observed)
    and the channel that models the relation between each aggregated
    record and the QID attribute.

    We assume that the prior on raw records (group of rows per user)
    is uniform, as beforehand the adversary would have not reason
    to chosen one record over another. The prior on aggregated records
    is then the number of users whose raw record maps to the same
    aggregated records, normalised by the number of records.

    In other words, we assume that the QID that the adversary learns
    is a piece of the aggregated value that the adversary wants to infer.

    A record is usually considered to be 1-dimensional, but this can be
    controlled via `split_attr`, which must be one of the columns
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
    >>> owner_attr = "uid"
    >>> count_attr = "count"
    >>> sum_attr = "sum"
    >>> agg_attr = "transaction_cost"
    >>> prior, ch = model.agg_count_sum.baseline(
    ...     dataset,
    ...     owner_attr,
    ...     count_attr,
    ...     sum_attr,
    ...     agg_attr
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
    ┌───────────┬───────────┬───────────────────────┬──────────┐
    │ count     ┆ sum       ┆ hint_transaction_cost ┆ p        │
    │ ---       ┆ ---       ┆ ---                   ┆ ---      │
    │ list[u32] ┆ list[f64] ┆ f64                   ┆ f64      │
    ╞═══════════╪═══════════╪═══════════════════════╪══════════╡
    │ [2]       ┆ [2.0]     ┆ 0.0                   ┆ 0.25     │
    │ [2]       ┆ [2.0]     ┆ 1.0                   ┆ 0.5      │
    │ [2]       ┆ [2.0]     ┆ 2.0                   ┆ 0.25     │
    │ [3]       ┆ [2.0]     ┆ 0.0                   ┆ 0.333333 │
    │ [3]       ┆ [2.0]     ┆ 1.0                   ┆ 0.666667 │
    └───────────┴───────────┴───────────────────────┴──────────┘
    """
    hint_attrs = [agg_attr] + (
        [] if split_attr is None else [split_attr]
    )

    dataset = dataset.lazy()

    expected_cols = [owner_attr, *hint_attrs]
    diff, valid = _valid_columns(dataset, expected_cols)
    if not valid: raise ValueError(f"Missing columns {diff}")

    dataset_agg = dataset.group_by(owner_attr, split_attr).agg(
        pl.len().alias(count_attr),

        _process_expr(pre_process, pl.col(agg_attr).sum())
        .alias(sum_attr)
    )

    # `split_attr` must come first, for sorting!
    input_attrs = (
        [] if split_attr is None else [split_attr]
    ) + [count_attr, sum_attr]

    struct_expr = pl.struct(input_attrs)
    records = (
        dataset_agg
        .sort(input_attrs)
        .select(owner_attr, struct_expr.alias("record"))
        .group_by(owner_attr, maintain_order=True)
        .agg("record")
    )

    el_field = lambda field: pl.element().struct.field(field)
    extract_field = lambda field: (
        pl.col("record")
        .list.eval(el_field(field))
    )

    prior_dist_owners = (
        records
        .group_by(owner_attr, "record")
        .agg(pl.len())
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("p"))
    )

    record_len = (
        pl.col("record")
        .list.eval(el_field(count_attr))
        .list.sum()
    )

    process_agg = _process_expr(post_process, pl.col(agg_attr))
    ch_dist_owners = (
        records
        .join(dataset, on=owner_attr, how="inner")
        .with_columns(process_agg.alias(agg_attr))
        .group_by(owner_attr, "record", *hint_attrs)
        .agg(pl.len())
        .with_columns((pl.col("len") / record_len).alias("p"))
    )

    prior_owners = ProbabDist.from_polars(
        prior_dist_owners, 
        [owner_attr, "record"]
    )

    ch_owners = Channel.from_polars(
        ch_dist_owners, 
        [owner_attr, "record"],
        hint_attrs
    )

    hint_attrs_rename = {attr: "hint_" + attr for attr in hint_attrs}
    joint_owners = qif.push(prior_owners, ch_owners)
    joint_dist_agg = (
        joint_owners.dist
        .group_by("record", *hint_attrs)
        .agg(pl.col("p").sum().alias("p"))
        .rename(hint_attrs_rename)
        .select(
            *[extract_field(a).alias(a) for a in input_attrs],
            *hint_attrs_rename.values(),
            "p"
        )
    )

    joint_agg = Joint.from_polars(
        joint_dist_agg, 
        input_attrs, 
        list(hint_attrs_rename.values())
    )

    return qif.push_back(joint_agg)
