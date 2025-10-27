import polars

from qif_micro.datatypes import Channel, ProbabDist

def build(
    dataset: polars.DataFrame,
    owner_field: str,
    count_field: str,
    sum_field: str
) -> tuple[ProbabDist, Channel]:
    """
    The input to this function is a dataset that is result of
    aggregating some original data by count and sum.

    We assume that the original data that has been aggregated are
    non-negative integers (>= 0). We assume also that the adversary
    knows/will learn (from external sources) one of these values.

    This function then returns the adversary's intermediate knowledge
    (i.e., after the aggregated data has been observed) and the channel that
    models the relation between the each record and the qid_field.

    That is, this function models an attribute-inference attack, in
    which the adversary's goal is to guess the (count, sum) of a target. 

    ## Example

    >>> import polars
    >>> from qif_micro import model
    >>> dataset = polars.DataFrame({
    ...     "uid": [0, 1, 2],
    ...     "count": [2, 2, 3],
    ...     "sum": [2, 2, 2]
    ... })
    >>> owner_field = "uid"
    >>> count_field = "count"
    >>> sum_field = "sum"
    >>> prior, channel = model.agg_count_sum.build(
    ...     dataset,
    ...     owner_field,
    ...     count_field,
    ...     sum_field
    ... )
    >>> prior.dist.sort(by=[count_field, sum_field]).collect()
    shape: (2, 3)
    ┌───────┬─────┬──────────┐
    │ count ┆ sum ┆ p        │
    │ ---   ┆ --- ┆ ---      │
    │ i64   ┆ i64 ┆ f64      │
    ╞═══════╪═════╪══════════╡
    │ 2     ┆ 2   ┆ 0.666667 │
    │ 3     ┆ 2   ┆ 0.333333 │
    └───────┴─────┴──────────┘
    >>> channel.dist.sort(by=[count_field, sum_field, "qid"]).collect()
    shape: (6, 4)
    ┌───────┬─────┬─────┬──────────┐
    │ count ┆ sum ┆ qid ┆ p        │
    │ ---   ┆ --- ┆ --- ┆ ---      │
    │ i64   ┆ i64 ┆ i64 ┆ f64      │
    ╞═══════╪═════╪═════╪══════════╡
    │ 2     ┆ 2   ┆ 0   ┆ 0.333333 │
    │ 2     ┆ 2   ┆ 1   ┆ 0.333333 │
    │ 2     ┆ 2   ┆ 2   ┆ 0.333333 │
    │ 3     ┆ 2   ┆ 0   ┆ 0.5      │
    │ 3     ┆ 2   ┆ 1   ┆ 0.333333 │
    │ 3     ┆ 2   ┆ 2   ┆ 0.166667 │
    └───────┴─────┴─────┴──────────┘
    """
    assert owner_field in dataset.columns

    n_records = dataset.get_column(owner_field).n_unique()
    assert n_records == dataset.height

    # We assume the dataset fits in memory, but the prior and channel
    # could be really large, so we from this point we rely on laziness
    dataset_lazy = dataset.lazy()

    # If count > 1, numerator goes from qid - count + 2 to qid - 1
    start_numerator = polars.col("qid") - polars.col(count_field) + 2
    end_numerator = polars.col("qid")

    # If count > 1, denominator goes from to count - 2
    start_denominator = 0
    end_denominator = polars.col(count_field) - 1

    dataset_with_qids = (
        dataset_lazy
        .select(count_field, sum_field)
        .unique()
        .with_columns(qid=polars.int_ranges(0, polars.col("sum") + 1))
        .explode("qid")
        .with_columns(qid=polars.col("qid").cast(float))
    )

    product = polars.element().product()
    term_numerator = polars.col(sum_field) - polars.col("idx_num")
    term_denominator = (
        polars.col(sum_field) 
        + polars.col(count_field) - 1
        - polars.col("idx_denom")
    )

    is_length_one = polars.col(count_field) == 1
    is_sum_qid = polars.col("qid") == polars.col(sum_field)

    ch_dist = dataset_with_qids.with_columns(
        idx_num=polars.int_ranges(start_numerator, end_numerator),
        idx_denom=polars.int_ranges(start_denominator, end_denominator)
    ).with_columns(
        prod_num=term_numerator.list.eval(product).list.first(),
        prod_denom=term_denominator.list.eval(product).list.first()
    ).select(count_field, sum_field, "qid", p=(
        polars
        .when(is_length_one, is_sum_qid)
        .then(1)
        .when(is_length_one)
        .then(0)
        .otherwise(
            (polars.col(count_field) - 1) 
            * polars.col("prod_num")
            / polars.col("prod_denom")
        )
    )).filter(polars.col("p") > 0)

    prior_dist = (
        dataset_lazy
        .group_by("count", "sum")
        .agg(p=polars.len() / dataset.height)
    )

    prior = ProbabDist.from_polars(prior_dist, [count_field, sum_field])
    ch = Channel.from_polars(ch_dist, [count_field, sum_field], ["qid"])

    return prior, ch
