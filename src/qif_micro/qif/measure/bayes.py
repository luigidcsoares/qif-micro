import numpy as np
from multimethod import multimethod

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist
from qif_micro.qif.datatypes.typing import Slice


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
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import ProbabDist

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> qif.measure.bayes.prior(pi)
    np.float64(0.5)
    """
    return pi.dist.max()


@multimethod
def posterior(pi: ProbabDist, ch: Channel) -> np.floating:
    """
    The expected posterior Bayes vulnerability is computed as the sum of
    the column maxima in the joint distribution.

    This measures the probability that an adversary can successfully guess
    the secret given an observable output, averaged over all possible outputs.

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
        Expected posterior Bayes vulnerability. Values in [0, 1].

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
    
    >>> posterior_vuln = qif.measure.bayes.posterior(pi, ch)
    >>> posterior_vuln
    np.float64(0.8125)

    This value indicates 81.25% chance of inferring the secret value,
    on average over all possible outputs.
    """
    return posterior(qif.joint(pi, ch))


@multimethod
def posterior(j: Joint) -> np.floating:  # noqa: F811
    if isinstance(j.dist, Slice.__value__.__args__): 
        dist = [j.dist]
    else:
        dist = j.dist

    return sum(s.max(axis=0).sum() for s in dist)
