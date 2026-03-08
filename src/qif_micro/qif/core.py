import polars as pl
import numpy as np

from multimethod import multimethod
from scipy.sparse import issparse, csr_array

from qif_micro.qif.datatypes import (
    Channel,
    Joint,
    ProbabDist,
    Strategy
)

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
    
    >>> joint = qif.joint(pi, ch)
    >>> joint
    Joint(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>)

    >>> joint.dist.toarray()
    array([[0.0625, 0.125 , 0.0625],
           [0.    , 0.5   , 0.    ],
           [0.    , 0.    , 0.25  ]])

    It also works if the channel is not sparse:
    >>> ch = Channel(ch.dist.toarray())
    >>> qif.joint(pi, ch).dist
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


@multimethod
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

    >>> outer, posteriors = qif.hyper(pi, ch)
    >>> outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> posteriors.dist.toarray().T
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])

    It also works if the channel is not sparse:
    
    >>> ch = Channel(ch.dist.toarray())
    >>> outer, posteriors = qif.hyper(pi, ch)
    >>> outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> posteriors.dist.T
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])

    This function is overloaded to take a joint instead:
    >>> outer, posteriors = qif.hyper(qif.joint(pi, ch))
    >>> outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> posteriors.dist.T
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])
    """
    return hyper(joint(pi, ch))


@multimethod
def hyper(joint: Joint) -> tuple[ProbabDist, Channel]:
    outer_dist = joint.dist.sum(axis=0)
    post_dists = joint.dist / outer_dist
    post_dists = post_dists.tocsr() if issparse(joint.dist) else post_dists
    return ProbabDist(outer_dist), Channel(post_dists.T)


@multimethod
def strategy(belief: Joint) -> Strategy:
    """
    TODO
    """
    dist = belief.dist
    rows, cols = dist.nonzero()
    col_max = dist.max(axis=0).toarray()
    
    mask_data = dist[rows, cols] == col_max[cols]
    mask = csr_array((mask_data, (rows, cols)), shape=dist.shape)
    max_counts = mask.sum(axis=0)

    st_data = mask_data / max_counts[mask.indices]
    st_dist = csr_array((st_data, mask.indices, mask.indptr), shape=dist.shape)

    return Channel(st_dist.T)
    # FIXME: Keeping as backup. I do not really remember how this could happen.
    # 
    # It could be that the input is a joint with all-zero columns,
    # in which case there must be a strategy (uniform over all rows):
    # nz_per_col = dist.count_nonzero(axis=0)
    # allzero_cols = np.nonzero(nz_per_col == 0)[0]

    # n_allzero = allzero_cols.shape[0]
    # if n_allzero == 0: return Channel(st_dist.T)

    # st_dist = st_dist.tocoo()
    # st_data = st_dist.data
    # st_rows, st_cols = st_dist.coords

    # allzero_data = np.repeat(1 / n_allzero, n_allzero * dist.shape[0])
    # allzero_rows, allzero_cols = zip(*(
    #     (r, c) for c in allzero_cols for r in range(dist.shape[0])
    # ))

    # st_data = np.concatenate([st_data, allzero_data])
    # st_rows = np.concatenate([st_rows, allzero_rows])
    # st_cols = np.concatenate([st_cols, allzero_cols])

    # st_dist = coo_array((st_data, (st_rows, st_cols)), shape=dist.shape)
    # return Channel(st_dist.tocsr().T)


@multimethod
def strategy(belief: ProbabDist) -> Strategy:
    return strategy(Joint(belief.dist[:, np.newaxis]))
