from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from qif_micro.qif.datatypes.typing import Slice 
from qif_micro.qif.datatypes._internal import _is_dist_valid, _Error

@dataclass(frozen=True)
class ProbabDist:
    """
    Represents a probability distribution over a discrete domain.

    A probability distribution is a vector of non-negative values that sum to
    at most 1 (exactly 1 for complete distributions). It can be stored in
    dense format (numpy array) or sparse format (scipy.sparse.csr/csc_array).

    Attributes
    ----------
    dist : Slice, init? Yes
        The probability vector. Can be 1D dense (numpy.ndarray) or sparse
        (scipy.sparse.csr/csc_array). Represents probabilities over domain values.

    is_complete : bool, init? No
        True if the sum of probabilities equals 1.0 (within numerical tolerance),
        indicating a complete probability distribution. False if the sum is
        strictly less than 1.0, indicating a sub-probability distribution.

    Pre-conditions
    ---------------
    - ``dist`` must be 1-dimensional
    - All entries must be non-negative
    - The sum of entries must not exceed 1.0

    Post-conditions
    ----------------
    - ``is_complete`` is True iff the sum equals 1.0 (within numerical tolerance)

    Examples
    --------
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import ProbabDist

    >>> pd = ProbabDist(np.array([1/4, 1/2, 1/4]))
    >>> pd
    ProbabDist(dist=array([0.25, 0.5 , 0.25]), is_complete=True)

    >>> pd.is_complete
    True

    >>> sub_prob = ProbabDist(np.array([0.2, 0.3]))
    >>> sub_prob.is_complete
    False
    """
    dist: Slice
    is_complete: bool = field(init=False)

    def __post_init__(self):
        # ====================================================================
        # Pre-conditions
        # ====================================================================
        if self.dist.ndim != 1:
            raise ValueError("``dist`` must be 1-dimensional!")
        
        dist = self.dist[np.newaxis, :]
        dist = dist.tocsr() if sp.issparse(dist) else dist
        dist_check = _is_dist_valid(dist)

        if dist_check.error is _Error.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check.error is _Error.ROW_SUM_MISMATCH:
            raise ValueError("Sum of probability distribution exceeds 1!")
        # ====================================================================

        row_sum = dist_check.row_sum
        is_complete = np.isclose(row_sum, 1.0).all()
        object.__setattr__(self, "is_complete", bool(is_complete))
