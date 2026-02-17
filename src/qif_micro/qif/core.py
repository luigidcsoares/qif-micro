import polars as pl

from qif_micro.datatypes import channel, Channel, LazyChannel
from qif_micro.datatypes import probab_dist, ProbabDist, LazyProbabDist
from qif_micro.datatypes import joint, Joint, LazyJoint

import numpy as np
from scipy.sparse import issparse, csr_array

from . import datatypes

# TODO: get rid of push and rename _push, and update imports
#
def _push(pi: datatypes.ProbabDist,  ch: datatypes.Channel) -> datatypes.Joint:
    """
    Pushes a prior through a channel to compute a joint distribution.

    ## Example 
    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel
    >>> from qif_micro.qif.datatypes import ProbabDist

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> ch = Channel(csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ]))
    
    >>> joint = qif.core._push(pi, ch)
    >>> joint
    Joint(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>)

    >>> joint.dist.toarray()
    array([[0.0625, 0.125 , 0.0625],
           [0.    , 0.5   , 0.    ],
           [0.    , 0.    , 0.25  ]])

    It also works if the channel is not sparse:
    >>> ch = Channel(ch.dist.toarray())
    >>> qif.core._push(pi, ch).dist
    array([[0.0625, 0.125 , 0.0625],
           [0.    , 0.5   , 0.    ],
           [0.    , 0.    , 0.25  ]])
    """
    joint_dist = pi.dist[:, np.newaxis] * ch.dist
    # If channel is sparse, the result will be in coo repr, so we convert to csr
    joint_dist = joint_dist.tocsr() if issparse(ch.dist) else joint_dist
    # At this point, failure to build the joint is an implementation error
    try: return datatypes.Joint(joint_dist)
    except Exception as e: assert False, f"Joint build failed: {e!r}"


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

