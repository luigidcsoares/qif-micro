import math

import polars

from qif_micro.typing import Channel, Hyper
from qif_micro.typing import ProbabDist
from qif_micro.typing import Strategies

def _build_hyper(prior: ProbabDist, channel: Channel) -> Hyper:
    prior_fields = set(prior.collect_schema().names())
    assert "p" in prior_fields

    channel_fields = set(channel.collect_schema().names())
    assert "p" in channel_fields
    assert "qid" in channel_fields

    sensitive_fields = channel_fields - {"p", "qid"}
    joint = (
        channel
        .join(prior.lazy(), on=sensitive_fields, how="inner")
        .with_columns(polars.col("p") * polars.col("p_right"))
        .drop("p_right")
    )

    outer = joint.select("p", "qid").group_by("qid").sum()

    outer_sum = outer.select(polars.col("p").sum()).collect()
    assert math.isclose(outer_sum.item(), 1)

    posteriors = (
        joint
        .join(outer, on="qid", how="inner")
        .with_columns(p=polars.col("p") / polars.col("p_right"))
        .drop("p_right")
    )

    posteriors_sum = posteriors.select(polars.col("p").sum()).collect()
    expected_sum = outer.select(polars.col("p").len()).collect()
    assert math.isclose(posteriors_sum.item(), expected_sum.item())

    return outer, posteriors


def _build_strategies(hyper: Hyper) -> Strategies:
    outer, posteriors = hyper

    max_qids = posteriors.group_by("qid").agg(max=polars.col("p").max())
    masked = (
        posteriors
        .join(max_qids, on="qid", how="inner")
        .with_columns(p=polars.col("p") == polars.col("max"))
        .drop("max")
    )

    count_max = masked.group_by("qid").agg(cmax=polars.col("p").sum())
    strategies = (
        masked
        .join(count_max, on="qid", how="inner")
        .with_columns(p=polars.col("p") / polars.col("cmax"))
        .drop("cmax")
    )

    strategies_sum = strategies.select(polars.col("p").sum()).collect()
    expected_sum = outer.select(polars.col("p").len()).collect()
    assert math.isclose(strategies_sum.item(), expected_sum.item())

    return strategies


def posterior(
    prior: ProbabDist,
    channel: Channel,
    baseline: Hyper
) -> float:
    """
    Computes the adversary's expected chance of correctly guessing the
    secret, by constructing the adversary's strategy conditioned on each
    output, and evaluating the performance of each strategy with respect
    to the corresponding baseline state of knowledge.

    ## Example

    >>> import polars
    >>> from qif_micro import model
    >>> from qif_micro import eval

    Consider the following baseline:
    >>> baseline_outer = polars.LazyFrame({
    ...     "qid": [0, 1, 2],
    ...     "p": [4/9, 2/9, 1/3]
    ... })
    >>> baseline_post = polars.LazyFrame({
    ...     "count": [2, 2, 2, 3, 3, 3],
    ...     "sum": [2, 2, 2, 2, 2, 2],
    ...     "qid": [0, 1, 2, 0, 1, 2],
    ...     "p": [3/4, 0, 1, 1/4, 1, 0]
    ... })

    Furthermore, say that the observed dataset is
    >>> dataset = polars.DataFrame({
    ...     "uid": [0, 1, 2],
    ...     "count": [2, 2, 3],
    ...     "sum": [2, 2, 2]
    ... })

    Then, we construct a model for the attack in which the QID is one of
    the integers >= 0 that have been aggregated as a sum:

    >>> owner_field = "uid"
    >>> count_field = "count"
    >>> sum_field = "sum"
    >>> prior, channel = model.agg_count_sum.build(
    ...     dataset,
    ...     owner_field,
    ...     count_field,
    ...     sum_field
    ... )

    In this scenario, the Bayes vulnerability w.r.t. the baseline is
    >>> prior, channel = model.agg_count_sum.build(
    ...     dataset,
    ...     owner_field,
    ...     count_field,
    ...     sum_field
    ... )
    >>> eval.posterior(prior, channel, (baseline_outer, baseline_post))
    0.6666666666666666
    """
    baseline_outer, baseline_post = baseline
    outer, posteriors = _build_hyper(prior, channel)
    strategies = _build_strategies((outer, posteriors))

    join_fields = set(channel.collect_schema().names()) - {"p"}
    post_vuln = (
        baseline_post
        .join(strategies, on=join_fields, how="inner")
        .with_columns(p=polars.col("p") * polars.col("p_right"))
        .group_by("qid")
        .agg(p=polars.col("p").sum())
        .join(baseline_outer, on="qid", how="inner")
        .with_columns(p=polars.col("p") * polars.col("p_right"))
        .select(polars.col("p").sum())
        .collect()
        .item()
    )

    assert 0 <= post_vuln <= 1
    return post_vuln
