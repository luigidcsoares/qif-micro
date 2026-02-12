from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._internal import _is_dist_valid, _ProbabDistError

@dataclass(frozen=True)
class ProbabDist:
    """
    ## Example
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import ProbabDist
    >>> ProbabDist(np.array([1/4, 1/2, 1/4]))
    ProbabDist(dist=array([0.25, 0.5 , 0.25]))
    """
    dist: NDArray[np.floating]

    def __post_init__(self):
        dist_check = _is_dist_valid(self.dist[np.newaxis, :])

        if dist_check is _ProbabDistError.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check is  _ProbabDistError.ROW_SUM_MISMATCH:
            raise ValueError("Rows do not add up to 1!")
