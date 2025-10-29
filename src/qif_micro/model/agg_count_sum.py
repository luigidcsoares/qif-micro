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

def build(
    dataset: polars.DataFrame,
    owner_attr: str,
    count_attr: str,
    sum_attr: str,
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
    which the adversary's goal is to guess the (count, sum) of a target. 

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
    >>> channel.dist.sort(by=[count_attr, sum_attr, "qid"]).collect()
    shape: (6, 4)
    ┌───────┬─────┬─────┬──────────┐
    │ count ┆ sum ┆ qid ┆ p        │
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
    assert owner_attr in dataset.columns

    n_records = dataset.get_column(owner_attr).n_unique()
    assert n_records == dataset.height

    # We assume the dataset fits in memory, but the prior and channel
    # could be really large, so we from this point we rely on laziness
    pre_process_sum = _process_expr(pre_process, sum_attr)
    dataset_lazy = dataset.lazy().select(
        count_attr,
        pre_process_sum.name.keep()
    )

    # If count > 1, numerator goes from qid - count + 2 to qid - 1
    start_numerator = polars.col("qid") - polars.col(count_attr) + 2
    end_numerator = polars.col("qid")

    # If count > 1, denominator goes from to count - 2
    start_denominator = 0
    end_denominator = polars.col(count_attr) - 1

    dataset_with_qids = (
        dataset_lazy
        .unique()
        .with_columns(qid=polars.int_ranges(0, polars.col(sum_attr) + 1))
        .explode("qid")
        .with_columns(qid=polars.col("qid").cast(float))
    )

    product = polars.element().product()
    term_numerator = polars.col(sum_attr) - polars.col("idx_num")
    term_denominator = (
        polars.col(sum_attr) 
        + polars.col(count_attr) - 1
        - polars.col("idx_denom")
    )

    is_length_one = polars.col(count_attr) == 1
    is_sum_qid = polars.col("qid") == polars.col(sum_attr)

    ch_dist = dataset_with_qids.with_columns(
        idx_num=polars.int_ranges(start_numerator, end_numerator),
        idx_denom=polars.int_ranges(start_denominator, end_denominator)
    ).with_columns(
        prod_num=term_numerator.list.eval(product).list.first(),
        prod_denom=term_denominator.list.eval(product).list.first()
    ).select(count_attr, sum_attr, "qid", p=(
        polars
        .when(is_length_one, is_sum_qid)
        .then(1)
        .when(is_length_one)
        .then(0)
        .otherwise(
            (polars.col(count_attr) - 1) 
            * polars.col("prod_num")
            / polars.col("prod_denom")
        )
    )).filter(polars.col("p") > 0)

    prior_dist = (
        dataset_lazy
        .group_by(count_attr, sum_attr)
        .agg(p=polars.len() / dataset.height)
    )

    prior = ProbabDist.from_polars(prior_dist, [count_attr, sum_attr])
    ch = Channel.from_polars(ch_dist, [count_attr, sum_attr], ["qid"])

    return prior, ch


def baseline(
    dataset: polars.DataFrame,
    owner_attr: str,
    count_attr: str,
    sum_attr: str,
    agg_attr: str,
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
    assert owner_attr in dataset.columns

    pre_process_sum = _process_expr(pre_process, sum_attr)
    post_process_agg = _process_expr(post_process, agg_attr)

    dataset_lazy = dataset.lazy().with_columns(**{
        count_attr: polars.len().over(owner_attr),
        sum_attr: polars.col(agg_attr).sum().over(owner_attr)
    }).with_columns(
        pre_process_sum.name.keep(),
        post_process_agg.name.keep()
    )

    prior_dist_owners = (
        dataset_lazy
        .group_by(owner_attr, count_attr, sum_attr)
        .agg(p=1 / dataset.n_unique(owner_attr))
    )

    ch_dist_owners = (
        dataset_lazy
        .with_columns(record_len=polars.len().over(owner_attr))
        .group_by(polars.all())
        .agg(qid_freq=polars.len())
        .select(
            owner_attr, count_attr, sum_attr, agg_attr,
            p=polars.col("qid_freq") / polars.col("record_len")
        )
    )

    prior_owners = ProbabDist.from_polars(
        prior_dist_owners, 
        [owner_attr, count_attr, sum_attr]
    )

    ch_owners = Channel.from_polars(
        ch_dist_owners,
        [owner_attr, count_attr, sum_attr],
        [agg_attr]
    )

    joint_owners = qif.push(prior_owners, ch_owners)
    joint_agg_dist = (
        joint_owners.dist
        .group_by(count_attr, sum_attr, agg_attr)
        .agg(p=polars.col("p").sum())
    )

    joint_agg = Joint.from_polars(
        joint_agg_dist,
        [count_attr, sum_attr],
        [agg_attr]
    )

    return qif.push_back(joint_agg)
