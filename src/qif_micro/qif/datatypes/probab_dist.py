from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse

from qif_micro.qif.datatypes.typing import Slice 
from qif_micro.qif.datatypes._internal import _is_dist_valid, _ProbabDistError

@dataclass(frozen=True)
class ProbabDist:
    """
    Examples
    --------
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import ProbabDist
    >>> ProbabDist(np.array([1/4, 1/2, 1/4]))
    ProbabDist(dist=array([0.25, 0.5 , 0.25]), is_slice=False)
    """
    dist: Slice
    is_slice: bool = False

    def __post_init__(self):
        dist = self.dist[np.newaxis, :]
        dist = dist.tocsr() if issparse(dist) else dist
        dist_check = _is_dist_valid(dist, self.is_slice)

        if dist_check is _ProbabDistError.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check is  _ProbabDistError.ROW_SUM_MISMATCH:
            raise ValueError("Rows do not add up to 1!")
