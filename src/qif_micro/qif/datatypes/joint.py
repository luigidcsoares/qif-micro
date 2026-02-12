from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse, csr_array

from ._internal import _is_dist_valid, _ProbabDistError

@dataclass(frozen=True)
class Joint:
    """
    ## Example
    >>> from scipy.sparse import csr_array
    >>> from qif_micro.qif.datatypes import Joint

    >>> joint_dist = csr_array([
    ...     [1/16, 1/8, 1/16], # First row
    ...     [0,    1/2,    0], # Second row
    ...     [0,      0,  1/4]  # Third row
    ... ])
    >>> joint = Joint(joint_dist)

    >>> joint
    Joint(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>)

    >>> joint.dist.toarray()
    array([[0.0625, 0.125 , 0.0625],
           [0.    , 0.5   , 0.    ],
           [0.    , 0.    , 0.25  ]])
    """
    dist: NDArray[np.floating] | csr_array

    def __post_init__(self):
        inner_data = self.dist.data if issparse(self.dist) else self.dist
        dist_check = _is_dist_valid(inner_data.ravel()[np.newaxis, :])

        if dist_check is _ProbabDistError.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check is  _ProbabDistError.ROW_SUM_MISMATCH:
            raise ValueError("Rows do not add up to 1!")

