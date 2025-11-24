import polars as pl

from .probability import push
from qif_micro.datatypes import Channel, Joint, ProbabDist

def _strategies(joint: Joint) -> Channel:
    """
    Turns out that the set of all strategies can be represented as
    a channel mapping each output of the system to a distribution.
    We assume strategies are uniform on the support (optimal actions).

    Since we are assuming Bayes in general, this channel actually
    has the same dimensions than the joint.
    """
    joint_cols = joint.dist.group_by(joint.output)
    max_output = joint_cols.agg(pl.col("p").max().alias("max"))

    cols_with_max = joint.dist.join(max_output, on=joint.output)
    cols_masked = cols_with_max.with_columns(
        pl.col("p")
        .is_close(pl.col("max"))
        .cast(pl.Float64)
        .alias("p")
    )

    count_max = (
        cols_masked
        .group_by(joint.output)
        .agg(pl.col("p").sum().alias("cmax"))
    )

    st_dist = cols_masked.join(count_max, on=joint.output).select(
        *joint.input,
        *joint.output,
        (pl.col("p") / pl.col("cmax")).alias("p")
    )

    return Channel.from_polars(st_dist, joint.output, joint.input)


def posterior(model: Joint, baseline: Joint) -> float:
    """
    Computes the adversary's expected chance of correctly guessing the
    secret, by constructing the adversary's strategy conditioned on each
    output, and evaluating the performance of each strategy with respect
    to the corresponding baseline state of knowledge.

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> from qif_micro import qif
    >>> from qif_micro.datatypes import ProbabDist, Channel

    Consider the following baseline:
    >>> baseline_dist = pl.LazyFrame({
    ...     "X": ["x0", "x1"] * 3,
    ...     "Y": ["y0"] * 2 + ["y1"] * 2 + ["y2"] * 2,
    ...     "p": [1/3, 1/9, 0, 2/9, 1/3, 0]
    ... })
    >>> baseline = Joint.from_polars(baseline_dist, ["X"], ["Y"])

    Furthermore, let's say that the adversary's prior knowledge is
    >>> prior_dist = pl.LazyFrame({
    ...     "X": ["x0", "x1"],
    ...     "p": [2/3, 1/3]
    ... })
    >>> prior = ProbabDist.from_polars(prior_dist, ["X"])

    and say that the adversary has access to the following system:
    >>> ch_dist = pl.LazyFrame({
    ...     "X": ["x0"] * 3 + ["x1"] * 3,
    ...     "Y": ["y0", "y1", "y2"] * 2,
    ...     "p": [1/3, 1/3, 1/3, 1/2, 1/3, 1/6]
    ... })
    >>> ch = Channel.from_polars(ch_dist, ["X"], ["Y"])

    In this scenario the expected Bayes vuln wrt the baseline is
    >>> qif.posterior(qif.push(prior, ch), baseline)
    0.6666666666666666
    """
    strategies = _strategies(model)

    # Joint "input" matches with strategy "output" and vice-versa.
    # That is, strategy takes one of the output as input 
    # and returns the secret that the adversary will guess
    lcols = baseline.input + baseline.output
    rcols = strategies.output + strategies.input

    post_vuln = (
        baseline.dist
        .join(strategies.dist, left_on=lcols, right_on=rcols)
        .with_columns((pl.col("p") * pl.col("p_right")).alias("p"))
        .group_by(baseline.output)
        .agg(pl.col("p").sum().alias("p"))
        .select(pl.col("p").sum().alias("p"))
        .collect()
        .item()
    )

    assert 0 <= post_vuln <= 1
    return post_vuln
