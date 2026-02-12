import numpy as np

from qif_micro.qif.datatypes import ProbabDist, Channel
from qif_micro.qif.core import _push

def prior(pi: ProbabDist) -> np.floating:
    """
    The prior bayes vulnerability is just the maximum prior probability. 

    ## Example
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import ProbabDist
    >>> from qif_micro.qif.measure import bayes

    >>> pi = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> bayes.prior(pi)
    np.float64(0.5)
    """
    return pi.dist.max()


def posterior(pi: ProbabDist, ch: Channel) -> np.floating:
    """
    The expected posterior Bayes vulnerability is computed as
    the sum of the column maxima in the joint distribution.

    ## Example
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
    return _push(pi, ch).dist.max(axis=0).sum()
