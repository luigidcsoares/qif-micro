from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from qif_micro.qif.datatypes._validation import _is_dist_valid, _Error


@dataclass(frozen=True)
class ProbabDist:
    """
    Represents a probability distribution over a discrete domain.

    A probability distribution is a vector of non-negative values that sum to
    at most 1 (exactly 1 for complete distributions). Stored in dense format
    as a numpy array.

    Attributes
    ----------
    dist : np.ndarray, init? Yes
        The 1D probability vector as a dense numpy array. Represents
        probabilities over domain values.

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

    Consider the following probability distribution:

    >>> dist = [1/4, 0, 3/4]
    >>> ProbabDist(np.array(dist))
    ProbabDist(dist=array([0.25, 0.  , 0.75]), is_complete=True)

    The distribution need not be complete; it can be a slice:

    >>> ProbabDist(np.array([0.2, 0.3]))
    ProbabDist(dist=array([0.2, 0.3]), is_complete=False)
    """
    dist: np.ndarray
    is_complete: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.dist, np.ndarray):
            raise ValueError("``dist`` must be ndarray!")

        if self.dist.ndim != 1:
            raise ValueError("``dist`` must be 1-dimensional!")

        dist_check = _is_dist_valid([self.dist])

        if dist_check.error is _Error.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check.error is _Error.AXIS_SUM_MISMATCH:
            raise ValueError("Sum of probability distribution exceeds 1!")
        # ====================================================================

        axis_sum = dist_check.axis_sum
        is_complete = np.isclose(axis_sum, 1.0).all()
        object.__setattr__(self, "is_complete", bool(is_complete))
