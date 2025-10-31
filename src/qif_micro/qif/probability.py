import polars

from qif_micro.datatypes import Channel, Joint, ProbabDist

def push(prior: ProbabDist, ch: Channel) -> Joint:
    """
    Given a probability distribution `prior`: DX and a channel
    `ch`: X -> DY implementing the conditional distributions p(Y | x),
    computes the joint distribution p(X, Y) as p(x) * p(y | x).
    """
    joint_dist = ch.dist.join(
        prior.dist,
        left_on=ch.input_names,
        right_on=prior.outcome_names,
        how="inner"
    ).select(
        polars.exclude("p", "p_right"),
        polars.col("p") * polars.col("p_right").alias("p")
    )
    

    return Joint.from_polars(
        joint_dist,
        ch.input_names,
        ch.output_names
    )


def push_back(joint: Joint) -> tuple[ProbabDist, Channel]:
    """
    Decomposes a joint distribution into prior and channel,
    noting that p(x) = sum_x p(x, y) and p(y | x) = p(x, y) / p(x).
    """ 
    prior_dist = joint.dist.group_by(joint.input_names).agg(
        polars.col("p").sum()
    )

    ch_dist = joint.dist.join(
        prior_dist, on=joint.input_names, how="inner"
    ).select(
        polars.exclude("p", "p_right"),
        (polars.col("p") / polars.col("p_right")).alias("p")
    )

    prior = ProbabDist.from_polars(prior_dist, joint.input_names)
    ch = Channel.from_polars(
        ch_dist,
        joint.input_names,
        joint.output_names
    )

    return prior, ch
