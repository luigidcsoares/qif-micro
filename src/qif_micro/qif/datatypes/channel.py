from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from qif_micro.qif.datatypes.typing import Slice
from qif_micro.qif.datatypes._internal import _is_dist_valid, _Error

@dataclass(frozen=True)
class Channel:
    """
    Represents a stochastic channel as a matrix mapping inputs to outputs.

    A stochastic channel is a matrix where each row represents an input value
    and each column represents an output value. Each row is a probability
    distribution over outputs, so entries are non-negative and row sums are
    at most 1 (exactly 1 for complete channels).

    The matrix may be stored in dense format (numpy array) or sparse format
    (scipy.sparse.csr/csc_array) for memory efficiency with large domains.

    Attributes
    ----------
    dist : Slice | Sequence[Slice], init? Yes
        The stochastic matrix. Can be 2D dense (numpy.ndarray) or sparse
        (scipy.sparse.csr/csc_array). Shape is (n_inputs, n_outputs).

    is_complete : bool, init? No
        True if all row sums equal 1.0 (within numerical tolerance), indicating
        a complete probability distribution. False if row sums are strictly less
        than 1.0, indicating a sub-probability distribution.

    Pre-conditions
    ---------------
    - ``dist`` must be 2-dimensional (if partitioned, every slice must be 2D)
    - All entries must be non-negative
    - Each row must sum to at most 1.0

    Post-conditions
    ----------------
    - ``is_complete`` is True iff all rows sum to 1.0 (within numerical tolerance)

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> from qif_micro.qif.datatypes import Channel

    >>> ch = Channel(sp.csr_array([
    ...     [1/4, 1/2, 1/4],   # First row
    ...     [0,   1,   0],     # Second row
    ...     [0,   0,   1],     # Third row
    ... ]))

    >>> ch
    Channel(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>, is_complete=True)

    >>> ch.dist.toarray()
    array([[0.25, 0.5 , 0.25],
           [0.  , 1.  , 0.  ],
           [0.  , 0.  , 1.  ]])

    >>> ch.is_complete
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

        for s in dist:
            msg = "``dist`` must be 2-dimensional!"
            if s.ndim != 2: raise ValueError(msg)
        
        dist_check = _is_dist_valid(dist)

        if dist_check.error is _Error.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check.error is _Error.ROW_SUM_MISMATCH:
            raise ValueError("Sum of rows exceeds 1!")
        # ====================================================================

        row_sum = dist_check.row_sum
        is_complete = np.isclose(row_sum, 1.0).all()
        object.__setattr__(self, "is_complete", bool(is_complete))
