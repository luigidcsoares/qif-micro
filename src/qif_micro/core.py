import polars as pl

from qif_micro import qif
from qif_micro.datatypes import channel, Channel, LazyChannel
from qif_micro.datatypes import joint
from qif_micro.datatypes import probab_dist, ProbabDist, LazyProbabDist 

def _strategies(prior: ProbabDist, ch: Channel) -> Channel:
    """
    Turns out that the set of all strategies can be represented as
    a channel mapping each output of the system to a distribution.
    We assume strategies are uniform on the support (optimal actions).

    Since we are assuming Bayes in general, this channel actually
    has the same dimensions than the joint.
    """
    joint = qif.push(prior, ch)

    expr_max = pl.col("p").max().over(joint.output).alias("max")
    joint_with_max = joint.dist.with_columns(expr_max)

    expr_mask = (pl.col("p") == pl.col("max")).cast(pl.Float64).alias("p")
    joint_masked = joint_with_max.with_columns(expr_mask)

    expr_cmax = pl.col("p").sum().over(joint.output)
    expr_uniform_p = (pl.col("p") / expr_cmax).alias("p")
    
    st_dist = joint_masked.with_columns(*joint.secret, *joint.output, expr_uniform_p)
    return channel.make(st_dist, joint.output, joint.secret)


def linkage_risk(
    prior_adv: ProbabDist | LazyProbabDist,
    prior_baseline: ProbabDist | LazyProbabDist,
    hint_ch: Channel | LazyChannel
) -> float:
    """
    Computes the adversary's expected chance of correctly guessing the
    secret, by constructing the adversary's strategy conditioned on each
    output, and evaluating the performance of each strategy with respect
    to the corresponding baseline state of knowledge.

    TODO: add gain function

    ## Example
    TODO 
    """
    prior_adv = probab_dist.collect(prior_adv)
    prior_baseline = probab_dist.collect(prior_baseline)
    hint_ch = channel.collect(hint_ch)

    baseline = joint.collect(qif.push(prior_baseline, hint_ch))
    st = _strategies(prior_adv, hint_ch)

    # Joint `secret` matches with strategy `output` and vice-versa.
    # That is, strategy takes one of the output as input 
    # and returns the secret that the adversary will guess
    lcols = baseline.secret + baseline.output
    rcols = st.output + st.secret

    return (
        baseline.dist
        .join(st.dist, left_on=lcols, right_on=rcols)
        .with_columns((pl.col("p") * pl.col("p_right")).alias("p"))
        .select(pl.col("p").sum().alias("p"))
        .collect()
        .item()
    )
