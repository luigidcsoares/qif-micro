from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
from multimethod import multimethod

from qif_micro.qif.datatypes import (
    Channel,
    Hyper,
    Joint,
    ProbabDist,
    StochMatrix,
    Strategy,
)
from qif_micro.qif.datatypes.typing import is_partitioned


def joint(pi: ProbabDist, ch: Channel) -> Joint:
    """
    Pushes a prior through a channel to compute a joint distribution.

    Parameters
    ----------
    pi : ProbabDist
        Prior probability distribution over the domain of secret values.

    ch : Channel
        Stochastic channel (matrix) mapping secrets to observable outputs.

    Returns
    -------
    Joint

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel, ProbabDist

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> ch = Channel(sp.csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ]))
    
    >>> joint = qif.joint(pi, ch)
    >>> joint
    Joint(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>, is_complete=True)

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
    ch_dist = ch.dist if is_partitioned(ch.dist) else [ch.dist]
    pi_dist = pi.dist[:, np.newaxis] 

    # If channel is sparse, the result will be in coo repr, so we convert to csr
    def _mk_joint(s):
        joint_slice = pi_dist * s
        return joint_slice.tocsr() if sp.issparse(s) else joint_slice

    
    joint_dist = [_mk_joint(s) for s in ch_dist]
    joint_dist = joint_dist if len(joint_dist) > 1 else joint_dist[0]

    # At this point, failure to build the joint is an implementation error
    try:
        return Joint(joint_dist)
    except Exception as e:
        assert False, f"Joint build failed: {e!r}"


@multimethod
def hyper(pi: ProbabDist, ch: Channel) -> Hyper:
    """
    Pushes a prior through a channel to compute a hyper-distribution.

    Parameters
    ----------
    This function is overloaded:

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
    Hyper
        The hyper-distribution containing both the outer distribution over
        outputs and the posterior distributions for each observation.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel, ProbabDist

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> ch = Channel(sp.csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ]))

    >>> h = qif.hyper(pi, ch)
    >>> h.outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> h.posteriors.dist.toarray()
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])

    It also works if the channel is not sparse:

    >>> ch = Channel(ch.dist.toarray())
    >>> h = qif.hyper(pi, ch)
    >>> h.outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> h.posteriors.dist
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])

    This function is overloaded to take a joint instead:

    >>> h = qif.hyper(qif.joint(pi, ch))
    >>> h.outer.dist
    array([0.0625, 0.625 , 0.3125])

    >>> h.posteriors.dist
    array([[1. , 0.2, 0.2],
           [0. , 0.8, 0. ],
           [0. , 0. , 0.8]])
    """
    return hyper(joint(pi, ch))


@multimethod
def hyper(j: Joint) -> Hyper:  # noqa: F811
    j_dist = j.dist if is_partitioned(j.dist) else [j.dist]

    # If joint is sparse, the result will be in coo repr, so we convert to csr
    def _mk_post(s_joint, s_outer):
        # If the joint is a slice, the outer may have zeros, which will trigger
        # a division by zero. So we replace zeros with a flag 1, which shouldnt
        # change the final result (as the cells will be 0 / 1):
        s_outer = s_outer.copy()
        s_outer[s_outer == 0] = 1
        post_slice = s_joint / s_outer
        return post_slice.tocsr() if sp.issparse(s_joint) else post_slice

    outer_dist = [s.sum(axis=0) for s in j_dist]
    post_dists = [_mk_post(*p) for p in zip(j_dist, outer_dist)]
    post_dists_combined = (
        post_dists if len(post_dists) > 1 else post_dists[0]
    )
    outer_dist_combined = np.hstack(outer_dist)

    outer = ProbabDist(outer_dist_combined)
    posteriors = StochMatrix(post_dists_combined, dist_orient=0)

    return Hyper(outer, posteriors)


def _mk_strategy(dist):
    if not sp.issparse(dist):
        dist = sp.csr_array(dist)

    rows, cols = dist.nonzero()
    col_max = dist.max(axis=0).toarray()

    mask_data = dist[rows, cols] == col_max[cols]
    mask = sp.csr_array((mask_data, (rows, cols)), shape=dist.shape)
    max_counts = mask.sum(axis=0)

    st_data = mask_data / max_counts[mask.indices]
    csr_repr = (st_data, mask.indices, mask.indptr)

    return sp.csr_array(csr_repr, shape=dist.shape)


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
    >>> import scipy.sparse as sp
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel, ProbabDist, Strategy

    Given the following prior knowledge, the adversary's strategy a priori
    (assuming the identity gain function) is to guess the second secret:
    
    A strategy may be deterministic or probabilistic:
    
    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> qif.strategy(pi).dist
    array([0., 1., 0.])

    >>> pi = ProbabDist(np.array([2/5, 1/5, 2/5]))
    >>> qif.strategy(pi).dist
    array([0.5, 0. , 0.5])

    After observing the output of a channel, the adv updates their strategy.
    With multiple outputs, the strategy is a 2D array:
    
    >>> ch = Channel(sp.csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ]))

    >>> st = qif.strategy(pi, ch)
    >>> st
    Strategy(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>, is_complete=True)

    >>> st.dist.toarray()
    array([[1. , 0.5, 0. ],
           [0. , 0.5, 0. ],
           [0. , 0. , 1. ]])
    """
    st_dist = _mk_strategy(pi.dist[:, np.newaxis])
    return Strategy(st_dist.toarray().ravel())


@multimethod
def strategy(pi: ProbabDist, ch: Channel) -> Strategy:  # noqa: F811
    return strategy(joint(pi, ch))


@multimethod
def strategy(joint: Joint) -> Strategy:  # noqa: F811
    is_partitioned = isinstance(joint.dist, Sequence)
    joint_dist = joint.dist if is_partitioned else [joint.dist]

    st_dist = [_mk_strategy(s) for s in joint_dist]
    if len(st_dist) == 1: st_dist = st_dist[0]

    return Strategy(st_dist)
