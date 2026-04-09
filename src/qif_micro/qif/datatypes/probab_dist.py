from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from qif_micro.qif.datatypes.typing import Slice 
from qif_micro.qif.datatypes._internal import _is_dist_valid, _Error

@dataclass(frozen=True)
class ProbabDist:
    """
    Examples
    --------
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import ProbabDist
    >>> ProbabDist(np.array([1/4, 1/2, 1/4]))
    ProbabDist(dist=array([0.25, 0.5 , 0.25]), is_complete=True)
    """
    dist: Slice
    is_complete: bool = field(init=False)

    def __post_init__(self):
        dist = self.dist[np.newaxis, :]
        dist = dist.tocsr() if sp.issparse(dist) else dist
        dist_check = _is_dist_valid(dist)

        if dist_check.error is _Error.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check.error is _Error.ROW_SUM_MISMATCH:
            raise ValueError("Sum of probability distribution exceeds 1!")

        row_sum = dist_check.row_sum
        is_complete = np.isclose(row_sum, 1.0).all()
        object.__setattr__(self, "is_complete", bool(is_complete))
