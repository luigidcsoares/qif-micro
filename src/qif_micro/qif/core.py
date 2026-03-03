import polars as pl

from qif_micro.datatypes import channel, Channel, LazyChannel
from qif_micro.datatypes import probab_dist, ProbabDist, LazyProbabDist
from qif_micro.datatypes import joint, Joint, LazyJoint

import numpy as np
from scipy.sparse import issparse, csr_array

from . import datatypes

def push(pi: datatypes.ProbabDist,  ch: datatypes.Channel) -> datatypes.Joint:
    """
    Pushes a prior through a channel to compute a joint distribution.

    Examples
    --------
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
    
    >>> joint = qif.push(pi, ch)
    >>> joint
    Joint(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>)

    >>> joint.dist.toarray()
    array([[0.0625, 0.125 , 0.0625],
           [0.    , 0.5   , 0.    ],
           [0.    , 0.    , 0.25  ]])

    It also works if the channel is not sparse:
    >>> ch = Channel(ch.dist.toarray())
    >>> qif.push(pi, ch).dist
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
