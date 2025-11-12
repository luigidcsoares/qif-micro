import polars as pl

from qif_micro.datatypes import Channel, Joint, ProbabDist

def push(prior: ProbabDist, ch: Channel) -> Joint:
    """
    Given a probability distribution `prior`: DX and a channel
    `ch`: X -> DY implementing the conditional distributions p(Y | x),
    computes the joint distribution p(X, Y) as p(x) * p(y | x).
    """
    joint_dist = ch.dist.join(
        prior.dist,
        left_on=ch.input,
        right_on=prior.outcome,
    ).select(
        *ch.input,
        *ch.output,
        (pl.col("p") * pl.col("p_right")).alias("p")
    )
    
    return Joint.from_polars(joint_dist, ch.input, ch.output)


def push_back(joint: Joint) -> tuple[ProbabDist, Channel]:
    """
    Decomposes a joint distribution into prior and channel,
    noting that p(x) = sum_x p(x, y) and p(y | x) = p(x, y) / p(x).
    """ 
    joint_rows = joint.dist.group_by(joint.input)
    prior_dist = joint_rows.agg(pl.col("p").sum().alias("p"))

    ch_dist = joint.dist.join(prior_dist, on=joint.input).select(
        *joint.input,
        *joint.output,
        (pl.col("p") / pl.col("p_right")).alias("p")
    )

    prior = ProbabDist.from_polars(prior_dist, joint.input)
    ch = Channel.from_polars(ch_dist, joint.input, joint.output)

    return prior, ch
