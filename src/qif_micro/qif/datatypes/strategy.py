from dataclasses import dataclass
from collections.abc import Sequence
import re
import scipy.sparse as sp

import numpy as np

from .probab_dist import ProbabDist
from .stoch_matrix import StochMatrix
from .typing import Slice


@dataclass(frozen=True, repr=False)
class Strategy:
    """
    Represents an adversary's strategy: a probability distribution over secrets.

    A strategy is always a probability distribution over secrets. If 1D, it
    represents a single distribution. If 2D, each column is a probability
    distribution over secrets, one for each possible observed output.

    The constructor accepts raw data and internally wraps it as either:
    - ProbabDist: if the data is 1D or has a single column
    - StochMatrix: if the data is 2D with multiple columns (dist_orient=0)

    Parameters
    ----------
    dist : Slice | Sequence[Slice]
        The strategy distribution data. Can be 1D or 2D dense (numpy.ndarray),
        sparse (scipy.sparse.csr/csc_array), or a sequence of such matrices
        when partitioned by columns.

    Pre-conditions
    ---------------
    - If 1D, it represents a probability distribution over secrets
    - If 2D, each column is a probability distribution over secrets
    - All entries must be non-negative
    - If 1D: sum must be at most 1.0
    - If 2D: each column sum must be at most 1.0

    Examples
    --------
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import Strategy

    Single output (1D strategy):

    >>> Strategy(np.array([0.5, 0.3, 0.2]))
    Strategy(dist=array([0.5, 0.3, 0.2]), is_complete=True)
    
    Multiple outputs (2D strategy):

    >>> matrix = [[1/2, 0], [1/2, 1], [0, 0]]
    >>> Strategy(np.array(matrix))
    Strategy(dist=array([[0.5, 0. ],
           [0.5, 1. ],
           [0. , 0. ]]), is_complete=True)
    """
    _inner: ProbabDist | StochMatrix

    def __init__(self, dist: Slice | Sequence[Slice]):
        # ProbabDist will never be partitioned, so assume it is StochMatrix.
        is_2d = isinstance(dist, Sequence) or dist.ndim > 1
        inner = StochMatrix(dist, dist_orient=0) if is_2d else ProbabDist(dist)
        object.__setattr__(self, "_inner", inner)


    @property
    def dist(self):
        """Access the underlying distribution data."""
        return self._inner.dist

    @property
    def is_complete(self):
        """Access the completeness flag."""
        return self._inner.is_complete

    def __repr__(self):
        inner_repr = repr(self._inner)
        # Replace inner type with Strategy and remove dist_orient
        inner_repr = re.sub(r"ProbabDist|StochMatrix", "Strategy", inner_repr)
        inner_repr = re.sub(r",\s*dist_orient=\d", "", inner_repr)
        return inner_repr
