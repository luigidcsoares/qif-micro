import enum

import numpy as np

from numpy.typing import NDArray
from scipy.sparse import issparse, csr_array

class _ProbabDistError(enum.Enum):
    OK = 0
    NEGATIVE_VALUES = 1
    ROW_SUM_MISMATCH = 2

    
def _is_dist_valid(dist: NDArray[np.floating] | csr_array) -> _ProbabDistError:
    inner_data = dist.data if issparse(dist) else dist

    if np.any(inner_data < 0):
        return _ProbabDistError.NEGATIVE_VALUES

    if not np.isclose(dist.sum(axis=1), 1).all():
        return _ProbabDistError.ROW_SUM_MISMATCH

    return _ProbabDistError.OK
    
