from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from qif_micro.qif.datatypes.typing import Slice
from qif_micro.qif.datatypes._internal import _is_dist_valid, _Error

@dataclass(frozen=True)
class Channel:
    """
    The ``Channel`` class stores an stochastic matrix.
    The matrix may be dense (numpy array) or sparse (scipy csr_array).

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
    """
    dist: Slice | Sequence[Slice]
    is_complete: bool = field(init=False)

    def __post_init__(self):
        dist_check = _is_dist_valid(self.dist)

        if dist_check.error is _Error.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check.error is _Error.ROW_SUM_MISMATCH:
            raise ValueError("Sum of rows exceeds 1!")

        row_sum = dist_check.row_sum
        is_complete = np.isclose(row_sum, 1.0).all()
        object.__setattr__(self, "is_complete", bool(is_complete))
