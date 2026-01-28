import polars as pl

from qif_micro.datatypes import channel, Channel, LazyChannel
from qif_micro.datatypes import probab_dist, ProbabDist, LazyProbabDist
from qif_micro.datatypes import joint, Joint, LazyJoint

def push(
    prior: ProbabDist | LazyProbabDist,
    ch: Channel | LazyChannel
 ) -> LazyJoint:
    """
    Given a probability distribution `prior`: DX and a channel
    `ch`: X -> DY implementing the conditional distributions p(Y | x),
    computes the joint distribution p(X, Y) as p(x) * p(y | x).
    """
    p_expr = (pl.col("p") * pl.col("p_right")).alias("p")
    _ = ch.dist
    _ = _.join(prior.dist, left_on=ch.secret, right_on=prior.secret)
    joint_dist = _.select(*ch.secret, *ch.output, p_expr)
    return joint.make_lazy(joint_dist, ch.secret, ch.output)    


def push_back(joint: Joint | LazyJoint) -> tuple[
    ProbabDist | LazyProbabDist,
    Channel | LazyChannel
]:
    """
    Decomposes a joint distribution into prior and channel,
    noting that p(x) = sum_x p(x, y) and p(y | x) = p(x, y) / p(x).
    """ 
    _ = joint.dist.group_by(joint.secret)
    prior_dist = _.agg(pl.col("p").sum().alias("p"))
    prior = probab_dist.make_lazy(prior_dist, joint.secret)

    p_expr = (pl.col("p") / pl.col("p_right")).alias("p")
    _ = joint.dist.join(prior_dist, on=joint.secret)
    ch_dist = _.select(*joint.secret, *joint.output, p_expr)
    ch = channel.make_lazy(ch_dist, joint.secret, joint.output)

    return prior, ch
