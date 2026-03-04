import polars as pl
import numpy as np

from plum import dispatch
from scipy.sparse import issparse, csr_array

from qif_micro.qif.datatypes import Channel, Joint, ProbabDist

def joint(pi: ProbabDist,  ch: Channel) -> Joint:
    """
    Pushes a prior through a channel to compute a joint distribution.

    Parameters
    ----------
    pi : ProbabDist
        Prior probability distribution over the secret space.

    ch : Channel
        Stochastic channel (matrix) mapping secrets to observable outputs.
        The channel may be sparse or dense.

    Returns
    -------
    Joint
        An object whose ``dist`` attribute holds the joint distribution
        matrix (sparse or dense depending on the input channel).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel, ProbabDist

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> ch = Channel(csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ]))
    
    >>> joint = qif.probab.joint(pi, ch)
    >>> joint
    Joint(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>)

    >>> joint.dist.toarray()
    array([[0.0625, 0.125 , 0.0625],
           [0.    , 0.5   , 0.    ],
           [0.    , 0.    , 0.25  ]])

    It also works if the channel is not sparse:
    >>> ch = Channel(ch.dist.toarray())
    >>> qif.probab.joint(pi, ch).dist
    array([[0.0625, 0.125 , 0.0625],
           [0.    , 0.5   , 0.    ],
           [0.    , 0.    , 0.25  ]])
    """
    joint_dist = pi.dist[:, np.newaxis] * ch.dist
    # If channel is sparse, the result will be in coo repr, so we convert to csr
    joint_dist = joint_dist.tocsr() if issparse(ch.dist) else joint_dist
    # At this point, failure to build the joint is an implementation error
    try: return Joint(joint_dist)
    except Exception as e: assert False, f"Joint build failed: {e!r}"


@dispatch
def hyper(pi: ProbabDist, ch: Channel) -> tuple[ProbabDist, Channel]:
    """
    Pushes a prior through a channel to compute a hyper-distribution.

    The function is overloaded:

    - ``hyper(pi, ch)`` – accepts a :class:`ProbabDist` and a :class:`Channel`.
    - ``hyper(joint)`` – accepts a pre‑computed :class:`Joint` object.

    Parameters
    ----------
    pi : ProbabDist
        Prior probability distribution over the secret space.

    ch : Channel
        Stochastic channel (matrix) that maps secrets to observable outputs.

    Returns
    -------
    tuple (ProbabDist, Channel)
        - The outer distribution over outputs.
        - The posterior distributions for each observation.
          
    See Also
    --------
    hyper(joint) : Overload that works directly on a :class:`Joint` object.
    joint : Function that builds a joint distribution from a prior and a channe

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel, ProbabDist

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> ch = Channel(csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ]))

    >>> outer, posteriors = qif.probab.hyper(pi, ch)
    >>> outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> posteriors.dist.toarray().T
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])

    It also works if the channel is not sparse:
    
    >>> ch = Channel(ch.dist.toarray())
    >>> outer, posteriors = qif.probab.hyper(pi, ch)
    >>> outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> posteriors.dist.T
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])

    This function is overloaded to take a joint instead:
    >>> outer, posteriors = qif.probab.hyper(qif.probab.joint(pi, ch))
    >>> outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> posteriors.dist.T
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])
    """
    return hyper(joint(pi, ch))


@dispatch
def hyper(joint: Joint) -> tuple[ProbabDist, Channel]:
    outer_dist = joint.dist.sum(axis=0)
    post_dists = joint.dist / outer_dist
    post_dists = post_dists.tocsr() if issparse(joint.dist) else post_dists
    return ProbabDist(outer_dist), Channel(post_dists.T)
