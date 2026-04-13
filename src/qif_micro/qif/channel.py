from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from qif_micro.qif.datatypes import Channel


def identity(n: np.uint64) -> Channel:
    """
    Parameters
    ----------
    n: uint64
        Number of rows and columns.

    Returns
    -------
    Channel
        A new deterministic channel with probability 1 in the diagonal.

    Examples
    --------
    >>> from qif_micro import qif
    >>> qif.channel.identity(3).dist.toarray()
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    data = np.repeat(1.0, n)    
    indices = np.arange(n)
    indptr = np.arange(n + 1)
    ch_dist = sp.csr_array((data, indices, indptr), shape=(n, n))
    return Channel(ch_dist)


def reduced(ch: Channel) -> Channel:
    """
    Reduce the columns of a channel by merging scalar multiples.

    Merges columns that are scalar multiples of each other (i.e., proportionally
    identical), combining their contributions while preserving the channel's
    probabilistic semantics. This reduces dimensionality without changing the
    information flow properties.

    Warning
    -------
    For channels with many columns this operation can be costly in both
    execution time and, more importantly, memory consumption. Complexity is
    O(n_cols^2) time and O(n_cols * n_rows) memory.

    Note
    ----
    Partitioned channels are not yet supported (TODO).

    Parameters
    ----------
    ch : Channel
        Stochastic channel (matrix) mapping secrets to observable outputs.

    Returns
    -------
    Channel
        A new channel with scalar multiple columns merged, preserving the
        original probabilistic semantics and representation (sparse/dense).

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel

    >>> ch = Channel(sp.csr_array([[1/2, 1/4, 1/8, 1/8], [2/3, 1/6, 1/6, 0]]))
    >>> qif.channel.reduced(ch).dist.toarray()
    array([[0.625     , 0.25      , 0.125     ],
           [0.83333333, 0.16666667, 0.        ]])

    Here, the first and fourth columns are scalar multiples (1/2 and 1/4),
    so they are merged.

    As expected, it also works with a dense channel, preserving the dense
    representation:

    >>> qif.channel.reduced(Channel(ch.dist.toarray())).dist
    array([[0.625     , 0.25      , 0.125     ],
           [0.83333333, 0.16666667, 0.        ]])
    """
    # Keep track if the original channel was sparse or not, to preserve repr.
    # TODO: preserve partitions
    is_partitioned = isinstance(ch.dist, Sequence)
    ch_dist = ch.dist if is_partitioned else [ch.dist]

    keep_sparse = np.any([sp.issparse(s) for s in ch_dist])
    ch_dist = sp.hstack([sp.csr_array(s) for s in ch_dist])
    
    # We start by dividing each col by the first non-zero entry.
    # This guarantees that, if col_i = k * col_j, we get rid of k,
    # and they will be equal after that.
    ch_cols = ch_dist.tocsc()
    first_nz = ch_cols.indptr[:-1]
    norm_cols = (ch_dist / ch_cols.data[first_nz][np.newaxis, :]).tocsc()

    # Then, we compute a group id for columns that are equal.
    # We use their byte representation for a lookup.
    #
    # TODO: Any vectorised implementation of this?
    def to_bytes(col): return np.ascontiguousarray(col.toarray()).tobytes()
    
    n_cols = norm_cols.shape[1]
    bytes_to_id = {}
    col_ids = np.empty(n_cols, dtype=np.uint64)
    
    for j in range(n_cols):
        col_bytes = to_bytes(norm_cols[:, j])
        col_ids[j] = bytes_to_id.setdefault(col_bytes, j)
        
    # We now have columns that are scalar multiple of each other
    # identified by a unique hash value. Let's find those indices:
    indices = np.unique(col_ids, return_inverse=True)[1]

    # Then we create a matrix that, for each column (hash id)
    # has one's at each index (channel col) that has that hash,
    # and zero everywhere else. We use this to agg the chan cols.
    n_unique_cols = indices.max() + 1
    rows = np.arange(n_cols, dtype=np.uint64)
    cols = indices
    data = np.ones(n_cols, dtype=ch_cols.dtype)

    coo_repr = (data, (rows, cols))
    shape = (n_cols, n_unique_cols)
    agg = sp.coo_array(coo_repr, shape=shape).tocsc()

    reduced_dist = ch_dist @ agg
    reduced_dist = reduced_dist if keep_sparse else reduced_dist.toarray()
    return Channel(reduced_dist)
