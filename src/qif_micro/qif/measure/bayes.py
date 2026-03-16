from collections.abc import Sequence

import numpy as np

from multimethod import multimethod

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist

def prior(pi: ProbabDist) -> np.floating:
    """
    The prior Bayes vulnerability is just the maximum prior probability. 

    Parameters
    ----------
    pi : ProbabDist
        The adversary's prior knowledge on secrets.

    Returns
    -------
    np.floating

    Examples
    --------
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import ProbabDist
    >>> from qif_micro.qif.measure import bayes

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> bayes.prior(pi)
    np.float64(0.5)
    """
    return pi.dist.max()


@multimethod
def posterior(pi: ProbabDist, ch: Channel) -> np.floating:
    """
    The expected posterior Bayes vulnerability is computed as
    the sum of the column maxima in the joint distribution.

    Parameters
    ----------
    This function is overloaded:

    - ``posterior(pi, ch)``: accepts a :class:`ProbabDist` and a :class:`Channel`
    - ``posterior(joint)``: accepts a :class:`Joint`

    pi : ProbabDist
        The adversary's prior knowledge on secrets.
        (First overload)

    ch : Channel
        Stochastic channel (matrix) mapping secrets to observable outputs.
        (First overload)

    joint : Joint
        Joint distribution between secrets and observable outputs.
        (Second overload)

    Returns
    -------
    np.floating

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> from qif_micro.qif.datatypes import Channel
    >>> from qif_micro.qif.datatypes import ProbabDist
    >>> from qif_micro.qif.measure import bayes

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> ch = Channel(csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ]))
    
    >>> bayes.posterior(pi, ch)
    np.float64(0.8125)
    """
    return posterior(qif.joint(pi, ch))


@multimethod
def posterior(joint: Joint) -> np.floating:
    is_partitioned = isinstance(joint.dist, Sequence)
    joint_dist = joint.dist if is_partitioned else [joint.dist]
    return sum(s.max(axis=0).sum() for s in joint_dist)
