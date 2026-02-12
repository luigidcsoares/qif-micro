from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from ._internal import _is_dist_valid, _ProbabDistError

@dataclass(frozen=True)
class Channel:
    """
    ## Example
    >>> from scipy.sparse import csr_array
    >>> from qif_micro.qif.datatypes import Channel

    >>> ch_dist = csr_array([
    ...     [1/4, 1/2, 1/4], # First row
    ...     [0,     1,   0], # Second row
    ...     [0,     0,   1]  # Third row
    ... ])
    >>> ch = Channel(ch_dist)

    >>> ch
    Channel(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>)

    >>> ch.dist.toarray()
    array([[0.25, 0.5 , 0.25],
           [0.  , 1.  , 0.  ],
           [0.  , 0.  , 1.  ]])
    """
    dist: NDArray[np.floating] | csr_array

    def __post_init__(self):
        dist_check = _is_dist_valid(self.dist)

        if dist_check is _ProbabDistError.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check is  _ProbabDistError.ROW_SUM_MISMATCH:
            raise ValueError("Rows do not add up to 1!")
