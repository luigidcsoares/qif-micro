from enum import Enum, auto

import polars

from qif_micro import qif
from qif_micro.datatypes import Channel, Joint, ProbabDist

class ProcessMethod(Enum):
    AS_INT = auto()
    CEIL = auto()
    ROUND = auto()


def _process_expr(method: ProcessMethod, col_name: str) -> polars.Expr:
    col_expr = polars.col(col_name)
    match method:
        case ProcessMethod.AS_INT: return (col_expr * 10).cast(int)
        case ProcessMethod.CEIL: return col_expr.ceil()
        case ProcessMethod.ROUND: return col_expr.round()


def _build_ch(
    dataset: polars.LazyFrame,
    count_attr: str,
    sum_attr: str
) -> polars.LazyFrame:
    # If count > 1, numerator goes from hint - count + 2 to hint - 1
    start_numerator = polars.col("hint") - polars.col(count_attr) + 2
    end_numerator = polars.col("hint")

    # If count > 1, denominator goes from to count - 2
    start_denominator = 0
    end_denominator = polars.col(count_attr) - 1

    dataset_with_hints = (
        dataset
        .unique()
        .with_columns(hint=polars.int_ranges(0, polars.col(sum_attr) + 1))
        .explode("hint")
        .with_columns(hint=polars.col("hint").cast(float))
    )

    product = polars.element().product()
    term_numerator = polars.col(sum_attr) - polars.col("idx_num")
    term_denominator = (
        polars.col(sum_attr) 
            + polars.col(count_attr) - 1
            - polars.col("idx_denom")
    )

    is_length_one = polars.col(count_attr) == 1
    is_sum_hint = polars.col("hint") == polars.col(sum_attr)

    tmp_cols = ["idx_num", "idx_denom", "prod_num", "prod_denom"]
    ch_dist = dataset_with_hints.with_columns(
        polars.int_ranges(start_numerator, end_numerator)
        .alias("idx_num"),
        
        polars.int_ranges(start_denominator, end_denominator)
        .alias("idx_denom")
    ).with_columns(
        term_numerator.list.eval(product).list.first()
        .alias("prod_num"),

        term_denominator.list.eval(product).list.first()
        .alias("prod_denom")
    ).select(
        polars.exclude(tmp_cols), 
        polars
        .when(is_length_one, is_sum_hint).then(1)
        .when(is_length_one).then(0)
        .otherwise((polars.col(count_attr) - 1) * polars.col("prod_num")
            / polars.col("prod_denom"))
        .alias("p")
    ).filter(polars.col("p") > 0)

    return ch_dist


def build(
    dataset: polars.DataFrame,
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

    >>> import polars
    >>> from qif_micro import model
    >>> dataset = polars.DataFrame({
    ...     "uid": [0, 1, 2],
    ...     "count": [2, 2, 3],
    ...     "sum": [2, 2, 2]
    ... })
    >>> owner_attr = "uid"
    >>> count_attr = "count"
    >>> sum_attr = "sum"
    >>> prior, channel = model.agg_count_sum.build(
    ...     dataset,
    ...     owner_attr,
    ...     count_attr,
    ...     sum_attr
    ... )
    >>> prior.dist.sort(by=[count_attr, sum_attr]).collect()
    shape: (2, 3)
    ┌───────┬─────┬──────────┐
    │ count ┆ sum ┆ p        │
    │ ---   ┆ --- ┆ ---      │
    │ i64   ┆ i64 ┆ f64      │
    ╞═══════╪═════╪══════════╡
    │ 2     ┆ 2   ┆ 0.666667 │
    │ 3     ┆ 2   ┆ 0.333333 │
    └───────┴─────┴──────────┘
    >>> channel.dist.sort(by=[count_attr, sum_attr, "hint"]).collect()
    shape: (6, 4)
    ┌───────┬─────┬─────┬──────────┐
    │ count ┆ sum ┆ hint ┆ p        │
    │ ---   ┆ --- ┆ --- ┆ ---      │
    │ i64   ┆ i64 ┆ f64 ┆ f64      │
    ╞═══════╪═════╪═════╪══════════╡
    │ 2     ┆ 2   ┆ 0.0 ┆ 0.333333 │
    │ 2     ┆ 2   ┆ 1.0 ┆ 0.333333 │
    │ 2     ┆ 2   ┆ 2.0 ┆ 0.333333 │
    │ 3     ┆ 2   ┆ 0.0 ┆ 0.5      │
    │ 3     ┆ 2   ┆ 1.0 ┆ 0.333333 │
    │ 3     ┆ 2   ┆ 2.0 ┆ 0.166667 │
    └───────┴─────┴─────┴──────────┘
    """
    # `split_attr` must come first, for sorting!
    input_attrs = (
        [] if split_attr is None else [split_attr]
    ) + [count_attr, sum_attr]

    diff = set([owner_attr, *input_attrs]) - set(dataset.columns)
    if len(diff) > 0: raise ValueError(f"Missing columns {diff}")

    process_sum = _process_expr(pre_process, sum_attr).name.keep()
    dataset_lazy = dataset.lazy().with_columns(process_sum)

    if split_attr is None: 
        dataset = dataset_lazy.collect()

        prior_dist = dataset_lazy.group_by(*input_attrs) .agg(
            (polars.len() / dataset.height).alias("p")
        )

        ch_dist = _build_ch(dataset_lazy, owner_attr, *input_attrs)

        prior = ProbabDist.from_polars(prior_dist, input_attrs)
        ch = Channel.from_polars(ch_dist, input_attrs, ["hint"])

        return prior, ch

    # Iterate over each of the possible values for `split_attr`,
    # construct individual channels, and then combine them, weighting by
    # the proportion of individual values for each `split_attr` val.
    struct_expr = polars.struct(input_attrs)

    records = (
        dataset_lazy
        .sort(by=input_attrs)
        .select(owner_attr, struct_expr.alias("record"))
        .group_by(owner_attr, maintain_order=True)
        .agg("record")
        .select("record")
    )

    el_field = lambda field: polars.element().struct.field(field)
    subrecord_filter = lambda val: el_field(split_attr) == val
    extract_field_at = lambda field, at: (
        polars.col("record")
        .list.filter(subrecord_filter(at))
        .list.first()
        .struct.field(field)
    )

    build_ch_with_split =  lambda v: (
        _build_ch(records.with_columns(
            extract_field_at(count_attr, v).alias(count_attr),
            extract_field_at(sum_attr, v).alias(sum_attr),
        ).drop_nulls(), count_attr, sum_attr)
        .with_columns(
            polars.col("hint").alias("hint_value"),
            polars.lit(v).alias("hint_" + split_attr)
        )
    )

    split_vals = dataset_lazy.select(polars.col(split_attr).unique())
    extract_field = lambda field: polars.col("record").list.eval(
        el_field(field)
    )

    ch_dist = polars.concat(
        build_ch_with_split(v)
        for v in split_vals.collect().to_series()
    ).with_columns(
        polars.col("record")
        .list.eval(el_field(count_attr))
        .list.sum()
        .alias("total_count"),

        (polars.col("p") * polars.col(count_attr)).alias("p_scaled")
    ).select(
        extract_field(count_attr).alias(count_attr),
        extract_field(sum_attr).alias(sum_attr),
        extract_field(split_attr).alias(split_attr),
        "hint_value",
        "hint_" + split_attr,
        (polars.col("p_scaled") / polars.col("total_count")).alias("p")
    )

    prior_freq = records.group_by("record").agg(polars.len())
    prior_dist = prior_freq.select(
        extract_field(count_attr).alias(count_attr),
        extract_field(sum_attr).alias(sum_attr),
        extract_field(split_attr).alias(split_attr),
        (polars.col("len") / polars.col("len").sum()).alias("p")
    )

    prior = ProbabDist.from_polars(prior_dist, input_attrs)
    ch = Channel.from_polars(ch_dist, input_attrs, [
        "hint_value", "hint_" + split_attr
    ])

    return prior, ch


def baseline(
    dataset: polars.DataFrame,
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

    >>> import polars
    >>> from qif_micro import model
    >>> dataset = polars.DataFrame({
    ...     "uid": [0, 0, 1, 1, 2, 2, 2],
    ...     "transaction_cost": [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0]
    ... })
    >>> owner_attr = "uid"
    >>> count_attr = "count"
    >>> sum_attr = "sum"
    >>> agg_attr = "transaction_cost"
    >>> prior, channel = model.agg_count_sum.baseline(
    ...     dataset,
    ...     owner_attr,
    ...     count_attr,
    ...     sum_attr,
    ...     agg_attr
    ... )
    >>> prior.dist.sort(by=[count_attr, sum_attr]).collect()
    shape: (2, 3)
    ┌───────┬─────┬──────────┐
    │ count ┆ sum ┆ p        │
    │ ---   ┆ --- ┆ ---      │
    │ u32   ┆ f64 ┆ f64      │
    ╞═══════╪═════╪══════════╡
    │ 2     ┆ 2.0 ┆ 0.666667 │
    │ 3     ┆ 2.0 ┆ 0.333333 │
    └───────┴─────┴──────────┘
    >>> channel.dist.sort(by=[count_attr, sum_attr, agg_attr]).collect()
    shape: (5, 4)
    ┌───────┬─────┬──────────────────┬──────────┐
    │ count ┆ sum ┆ transaction_cost ┆ p        │
    │ ---   ┆ --- ┆ ---              ┆ ---      │
    │ u32   ┆ f64 ┆ f64              ┆ f64      │
    ╞═══════╪═════╪══════════════════╪══════════╡
    │ 2     ┆ 2.0 ┆ 0.0              ┆ 0.25     │
    │ 2     ┆ 2.0 ┆ 1.0              ┆ 0.5      │
    │ 2     ┆ 2.0 ┆ 2.0              ┆ 0.25     │
    │ 3     ┆ 2.0 ┆ 0.0              ┆ 0.333333 │
    │ 3     ┆ 2.0 ┆ 1.0              ┆ 0.666667 │
    └───────┴─────┴──────────────────┴──────────┘
    """
    hint_attrs = [agg_attr] + (
        [] if split_attr is None else [split_attr]
    )

    diff = set([owner_attr, *hint_attrs]) - set(dataset.columns)
    if len(diff) > 0: raise ValueError(f"Missing columns {diff}")

    process_sum = _process_expr(pre_process, sum_attr)
    dataset_lazy = dataset.lazy()

    dataset_agg_lazy = (
        dataset_lazy.group_by(owner_attr, split_attr).agg(
            polars.len().alias(count_attr),
            polars.col(agg_attr).sum().alias(sum_attr)
        ).with_columns(process_sum.name.keep())
    )

    # `split_attr` must come first, for sorting!
    input_attrs = (
        [] if split_attr is None else [split_attr]
    ) + [count_attr, sum_attr]

    struct_expr = polars.struct(input_attrs)
    records = (
        dataset_agg_lazy
        .sort(by=input_attrs)
        .select(owner_attr, struct_expr.alias("record"))
        .group_by(owner_attr, maintain_order=True)
        .agg("record")
    )

    el_field = lambda field: polars.element().struct.field(field)
    extract_field = lambda field: (
        polars.col("record").list.eval(el_field(field))
    )

    prior_probab = polars.col("len") / polars.col("len").sum()
    prior_dist_owners = (
        records
        .group_by(owner_attr, "record")
        .agg(polars.len())
        .with_columns(prior_probab.alias("p"))
    )

    record_len = polars.col("record").list.eval(
        el_field(count_attr)
    ).list.sum()

    process_agg = _process_expr(post_process, agg_attr)
    ch_probab = polars.col("len") / record_len

    ch_dist_owners = (
        records
        .join(dataset_lazy, on=owner_attr, how="inner")
        .with_columns(process_agg.name.keep())
        .group_by(owner_attr, "record", *hint_attrs)
        .agg(polars.len())
        .with_columns(ch_probab.alias("p"))
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
        .agg(polars.col("p").sum())
        .rename(hint_attrs_rename)
        .select(
            *[extract_field(a).alias(a) for a in input_attrs],
            *hint_attrs_rename.values(),
            "p"
        )
    )

    joint_agg = Joint.from_polars(
        joint_dist_agg, input_attrs, 
        list(hint_attrs_rename.values())
    )

    return qif.push_back(joint_agg)
