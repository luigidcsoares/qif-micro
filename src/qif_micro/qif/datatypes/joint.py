from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from qif_micro.qif.datatypes.typing import Slice
from qif_micro.qif.datatypes._internal import _is_dist_valid, _Error

@dataclass(frozen=True)
class Joint:
    """
    Represents a joint probability distribution over pairs of random variables.

    A joint distribution is a matrix where entry (i, j) represents the
    probability of the pair (i, j). Entries are non-negative and the total sum
    is at most 1 (exactly 1 for complete distributions).

    The matrix may be stored in dense format (numpy array) or sparse format
    (scipy.sparse.csr/csc_array) for memory efficiency.

    Attributes
    ----------
    dist : Slice | Sequence[Slice], init? Yes
        The joint distribution matrix or sequence of distributions.

    is_complete : bool, init? No
        True if the total sum of all probabilities equals 1.0 (within numerical
        tolerance), indicating a complete joint distribution. False if the sum
        is strictly less than 1.0, indicating a sub-probability distribution.

    Pre-conditions
    ---------------
    - ``dist`` must be 2-dimensional (if partitioned, every slice must be 2D)
    - All entries must be non-negative
    - The total sum of all entries must not exceed 1.0

    Post-conditions
    ----------------
    - ``is_complete`` is True iff the total sum equals 1.0 (within numerical tolerance)

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> from qif_micro.qif.datatypes import Joint

    >>> joint = Joint(sp.csr_array([
    ...     [1/16, 1/8, 1/16], # First row
    ...     [0,    1/2,    0], # Second row
    ...     [0,      0,  1/4]  # Third row
    ... ]))

    >>> joint
    Joint(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>, is_complete=True)

    >>> joint.dist.toarray()
    array([[0.0625, 0.125 , 0.0625],
           [0.    , 0.5   , 0.    ],
           [0.    , 0.    , 0.25  ]])

    >>> joint.is_complete
    True
    """
    dist: Slice | Sequence[Slice]
    is_complete: bool = field(init=False)

    def __post_init__(self):
        # ====================================================================
        # Pre-conditions
        # ====================================================================
        is_partitioned = isinstance(self.dist, Sequence)
        dist = self.dist if is_partitioned else [self.dist]

        for i, s in enumerate(dist):
            msg = "``dist`` must be 2-dimensional!"
            if s.ndim != 2: raise ValueError(msg)


        dist = [
            (s.data if sp.issparse(s) else s).ravel()[np.newaxis, :] 
            for s in dist
        ]
            
        dist_check = _is_dist_valid(dist)

        if dist_check.error is _Error.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check.error is _Error.ROW_SUM_MISMATCH:
            raise ValueError("Sum of joint distribution exceeds 1!")
        # ====================================================================

        row_sum = dist_check.row_sum
        is_complete = np.isclose(row_sum, 1.0).all()
        object.__setattr__(self, "is_complete", bool(is_complete))
