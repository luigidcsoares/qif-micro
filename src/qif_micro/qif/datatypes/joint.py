from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from qif_micro.qif.datatypes.typing import Slice
from qif_micro.qif.datatypes._internal import _is_dist_valid, _Error

@dataclass(frozen=True)
class Joint:
    """
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
    """
    dist: Slice | Sequence[Slice]
    is_complete: bool = field(init=False)

    def __post_init__(self):
        dist = [s.data if sp.issparse(s) else s for s in self.dist]
        dist = [s.ravel()[np.newaxis, :] for s in dist]
        dist_check = _is_dist_valid(dist)

        if dist_check.error is _Error.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check.error is _Error.ROW_SUM_MISMATCH:
            raise ValueError("Sum of joint distribution exceeds 1!")

        row_sum = dist_check.row_sum
        is_complete = np.isclose(row_sum, 1.0).all()
        object.__setattr__(self, "is_complete", bool(is_complete))

