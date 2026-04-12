from dataclasses import dataclass
from collections.abc import Sequence
import re

from qif_micro.qif.datatypes.stoch_matrix import StochMatrix
from qif_micro.qif.datatypes.typing import Slice


@dataclass(frozen=True, repr=False)
class Channel:
    """
    Represents a stochastic channel as a matrix mapping inputs to outputs.

    A stochastic channel is a matrix where each row represents an input value
    and each column represents an output value. Each row is a probability
    distribution over outputs, so entries are non-negative and row sums are
    at most 1 (exactly 1 for complete channels).

    The matrix may be stored in dense format (numpy array) or sparse format
    (scipy.sparse.csr/csc_array) for memory efficiency with large domains.

    Parameters
    ----------
    dist : Slice | Sequence[Slice]
        The stochastic matrix. Can be 2D dense (numpy.ndarray) or sparse
        (scipy.sparse.csr/csc_array). Shape is (n_inputs, n_outputs).

    Pre-conditions
    ---------------
    - ``dist`` must be 2-dimensional (if partitioned, every slice must be 2D)
    - All entries must be non-negative
    - Each row must sum to at most 1.0
    - If partitioned, all slices must have same number of rows

    Post-conditions
    ----------------
    - ``is_complete`` is True iff all rows sum to 1.0 (within numerical tolerance)

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from qif_micro.qif.datatypes import Channel

    Consider the following channel matrix:
    
    >>> matrix = [
    ...     [1/4, 1/2, 1/4],   # First row
    ...     [0,   1,   0],     # Second row
    ...     [0,   0,   1],     # Third row
    ... ]
    
    We can construct a dense representation of the channel:

    >>> Channel(np.array(matrix))
    Channel(dist=array([[0.25, 0.5 , 0.25],
           [0.  , 1.  , 0.  ],
           [0.  , 0.  , 1.  ]]), is_complete=True)

    Or we can construct a sparse representation:

    >>> ch = Channel(sp.csr_array(matrix))
    >>> ch
    Channel(dist=<Compressed Sparse Row sparse array of dtype 'float64'
        with 5 stored elements and shape (3, 3)>, is_complete=True)

    >>> ch.dist.toarray()
    array([[0.25, 0.5 , 0.25],
           [0.  , 1.  , 0.  ],
           [0.  , 0.  , 1.  ]])

    The channel need not be complete, it can be a slice (by columns):

    >>> Channel(np.array(matrix)[:, [0, 2]])
    Channel(dist=array([[0.25, 0.25],
           [0.  , 0.  ],
           [0.  , 1.  ]]), is_complete=False)

    And it may be partitioned (also by columns; perhaps for memory reasons):

    >>> part0 = np.array(matrix)[:, [0, 1]]
    >>> part1 = np.array(matrix)[:, [2]]
    >>> Channel([part0, part1])
    Channel(dist=[array([[0.25, 0.5 ],
           [0.  , 1.  ],
           [0.  , 0.  ]]), array([[0.25],
           [0.  ],
           [1.  ]])], is_complete=True)
    """
    _inner: StochMatrix

    def __init__(self, dist: Slice | Sequence[Slice]):
        # Create a StochMatrix with dist_orient=1 (row-oriented)
        object.__setattr__(self, "_inner", StochMatrix(dist, dist_orient=1))


    @property
    def dist(self): return self._inner.dist

    @property
    def is_complete(self): return self._inner.is_complete

    def __repr__(self):
        inner_repr = repr(self._inner)
        # Replace StochMatrix with Channel and remove dist_orient
        inner_repr = re.sub(r"StochMatrix", "Channel", inner_repr)
        inner_repr = re.sub(r",\s*dist_orient=\d", "", inner_repr)
        return inner_repr
