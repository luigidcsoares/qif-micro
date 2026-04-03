from collections.abc import Sequence

import numpy as np

from multimethod import multimethod
from scipy.sparse import csr_array, issparse

from qif_micro.qif.datatypes import (
    Channel,
    Joint,
    ProbabDist,
    Strategy
)

def joint(pi: ProbabDist, ch: Channel) -> Joint:
    """
    Pushes a prior through a channel to compute a joint distribution.

    Parameters
    ----------
    pi : ProbabDist
        Prior probability distribution over the secret space.

    ch : Channel
        Stochastic channel (matrix) mapping secrets to observable outputs.

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
        with 5 stored elements and shape (3, 3)>, is_slice=False)

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
    is_slice = pi.is_slice or ch.is_slice
    is_partitioned = isinstance(ch.dist, Sequence)

    ch_dist = ch.dist if is_partitioned else [ch.dist]
    pi_dist = pi.dist[:, np.newaxis] 

    # If channel is sparse, the result will be in coo repr, so we convert to csr
    def _mk_joint(s):
        joint_slice = pi_dist * s
        return joint_slice.tocsr() if issparse(s) else joint_slice

    
    joint_dist = [_mk_joint(s) for s in ch_dist]
    joint_dist = joint_dist if len(joint_dist) > 1 else joint_dist[0]

    # At this point, failure to build the joint is an implementation error
    try: return Joint(joint_dist, is_slice)
    except Exception as e: assert False, f"Joint build failed: {e!r}"


@multimethod
def hyper(pi: ProbabDist, ch: Channel) -> tuple[ProbabDist, Channel]:
    """
    Pushes a prior through a channel to compute a hyper-distribution.

    Parameters
    ----------
    The function is overloaded:

    - ``hyper(pi, ch)``: accepts a :class:`ProbabDist` and a :class:`Channel`.
    - ``hyper(joint)``:  accepts a pre‑computed :class:`Joint` object.

    pi : ProbabDist
        Prior probability distribution over the secret space.
        (First overload)

    ch : Channel
        Stochastic channel (matrix) that maps secrets to observable outputs.
        (First overload)

    joint : Joint
        Joint distribution between secrets and observable outputs.
        (Second overload)

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
    is_slice = joint.is_slice
    is_partitioned = isinstance(joint.dist, Sequence)
    joint_dist = joint.dist if is_partitioned else [joint.dist]

    # If joint is sparse, the result will be in coo repr, so we convert to csr
    def _mk_post(s_joint, s_outer):
        # If the joint is a slice, the outer will have zeros, which will trigger
        # a division by zero. So we replace zeros with a flag 1, which shouldnt
        # change the final result (as the cells will be 0 / 1):
        s_outer[s_outer == 0] = 1
        post_slice = (s_joint / s_outer).T
        return post_slice.tocsr() if issparse(s_joint) else post_slice

    
    outer_dist = [s.sum(axis=0) for s in joint_dist]
    post_dists = [_mk_post(*p) for p in zip(joint_dist, outer_dist)]
    post_dists = post_dists if len(post_dists) > 1 else post_dists[0]
    outer_dist = np.hstack(outer_dist)

    is_slice_outer = not np.isclose(outer_dist.sum(), 1.0)
    outer = ProbabDist(outer_dist, is_slice_outer)
    ch = Channel(post_dists, is_slice)

    return outer, ch


@multimethod
def strategy(pi: ProbabDist) -> Strategy:
    """
    Constructs the adversary's strategy according to their belief
    about how secret values are distributed.

    Parameters
    ----------
    This function is overloaded:

    - ``strategy(pi)``: accepts a :class:`ProbabDist`
    - ``strategy(pi, ch)``: accepts a :class:`ProbabDist` and a :class:`Channel`
    - ``strategy(joint)``: accepts a :class:`Joint`

    pi : ProbabDist
        The adversary's belief about how secrets are distributed.
        (First and second overloads)

    ch : Channel
        Stochastic channel (matrix) mapping secrets to observable outputs.
        (Second overload)

    joint: Joint
        The adversary's belief about how secrets are correlated with
        the outputs of a system. In this case, the adversary constructs
        one strategy for each possible output that they could observe.
        (Third overload)

    Returns
    -------
    Strategy
        A stochastic matrix where each row corresponds to one strategy.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel, ProbabDist

    Given the following prior knowledge, the adversary's strategy a priori
    (assuming the identity gain function) is to guess the second secret:
    
    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> qif.strategy(pi).dist.toarray()
    array([[0., 1., 0.]])

    A strategy could also be probabilistic:

    >>> pi = ProbabDist(np.array([2/5, 1/5, 2/5]))
    >>> qif.strategy(pi).dist.toarray()
    array([[0.5, 0. , 0.5]])

    After observing the output of a channel, the adv updates their strategy:
    
    >>> ch = Channel(csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ]))

    >>> qif.strategy(pi, ch).dist.toarray()
    array([[1. , 0. , 0. ],
           [0.5, 0.5, 0. ],
           [0. , 0. , 1. ]])
    """
    return strategy(Joint(pi.dist[:, np.newaxis]))


@multimethod
def strategy(pi: ProbabDist, ch: Channel) -> Strategy:
    return strategy(joint(pi, ch))


@multimethod
def strategy(joint: Joint) -> Strategy:
    is_slice = joint.is_slice
    is_partitioned = isinstance(joint.dist, Sequence)
    joint_dist = joint.dist if is_partitioned else [joint.dist]

    def _mk_strategy(dist):
        dist = dist if issparse(dist) else csr_array(dist)

        rows, cols = dist.nonzero()
        col_max = dist.max(axis=0).toarray()
    
        mask_data = dist[rows, cols] == col_max[cols]
        mask = csr_array((mask_data, (rows, cols)), shape=dist.shape)
        max_counts = mask.sum(axis=0)

        st_data = mask_data / max_counts[mask.indices]
        csr_repr = (st_data, mask.indices, mask.indptr)

        return csr_array(csr_repr, shape=dist.shape).T

    st_dist = [_mk_strategy(s) for s in joint_dist]
    st_dist = st_dist if len(st_dist) > 1 else st_dist[0]

    return Channel(st_dist, is_slice)
