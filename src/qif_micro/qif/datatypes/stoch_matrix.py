from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from qif_micro.qif.datatypes._validation import _Error, _is_dist_valid
from qif_micro.qif.datatypes.typing import Slice, is_partitioned


@dataclass(frozen=True)
class StochMatrix:
    """
    Represents a stochastic matrix with configurable orientation.

    A stochastic matrix is a matrix where probabilities sum to 1.0 along
    a specified axis. The orientation determines which axis contains the
    probability distributions:
    - dist_orient=0: Column-oriented (e.g., posteriors P(input|output))
    - dist_orient=1: Row-oriented (e.g., channels P(output|input))

    The matrix may be stored in dense format (numpy array) or sparse format
    (scipy.sparse.csr/csc_array) for memory efficiency with large domains.

    Attributes
    ----------
    dist : Slice | Sequence[Slice], init? Yes
        The stochastic matrix. Can be 2D dense (numpy.ndarray), sparse
        (scipy.sparse.csr/csc_array), or a sequence of such matrices when
        partitioned by columns. Shape is (n_rows, n_cols).

    dist_orient : int, init? Yes
        Specifies which axis represents probability distributions:
        - dist_orient=0: Column sums = 1.0 (column-oriented)
        - dist_orient=1: Row sums = 1.0 (row-oriented)

    is_complete : bool, init? No
        True if all sums along dist_orient equal 1.0 (within numerical
        tolerance), indicating complete probability distributions.

    Pre-conditions
    ---------------
    - ``dist`` must be 2-dimensional (if partitioned, every slice must be 2D)
    - ``dist_orient`` must be 0 or 1
    - All entries must be non-negative
    - If partitioned, all slices must have the same number of rows
    - Sums along dist_orient must be at most 1.0

    Post-conditions
    ----------------
    - ``is_complete`` is True iff all sums along dist_orient equal 1.0

    Examples
    --------
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import StochMatrix

    Row-oriented matrix (channel: P(output|input)):

    >>> matrix = np.array([[1/4, 1/2, 1/4], [0, 1, 0], [0, 0, 1]])
    >>> StochMatrix(matrix, dist_orient=1)
    StochMatrix(dist=array([[0.25, 0.5 , 0.25],
           [0.  , 1.  , 0.  ],
           [0.  , 0.  , 1.  ]]), dist_orient=1, is_complete=True)

    Column-oriented matrix (posterior: P(input|output)):

    >>> matrix = np.array([[2/3, 1/3], [1/3, 2/3]])
    >>> StochMatrix(matrix, dist_orient=0)
    StochMatrix(dist=array([[0.66666667, 0.33333333],
           [0.33333333, 0.66666667]]), dist_orient=0, is_complete=True)
    """
    dist: Slice | Sequence[Slice]
    dist_orient: int
    is_complete: bool = field(init=False)

    def __post_init__(self):
        # ====================================================================
        # Pre-conditions
        # ====================================================================
        if self.dist_orient not in (0, 1):
            raise ValueError("``dist_orient`` must be 0 or 1!")

        dist = self.dist if is_partitioned(self.dist) else [self.dist]

        # Check all slices are 2D
        n_rows = dist[0].shape[0]
        msg_dim = "``dist`` must be 2-dimensional!"
        msg_rows = "All partitions must have the same number of rows!"

        for s in dist:
            if s.ndim != 2: raise ValueError(msg_dim)
            if s.shape[0] != n_rows: raise ValueError(msg_rows)

        dist_check = _is_dist_valid(dist, dist_orient=self.dist_orient)

        if dist_check.error is _Error.NEGATIVE_VALUES:
            raise ValueError("Negative entries!")

        if dist_check.error is _Error.AXIS_SUM_MISMATCH:
            msg_kind = "columns" if self.dist_orient == 0 else "rows"
            raise ValueError(f"Sum of {msg_kind} exceeds 1!")
        # ====================================================================

        axis_sum = dist_check.axis_sum
        assert axis_sum is not None
        
        is_complete = np.isclose(axis_sum, 1.0).all()
        object.__setattr__(self, "is_complete", bool(is_complete))
