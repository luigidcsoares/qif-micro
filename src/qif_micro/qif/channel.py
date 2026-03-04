import numpy as np

from scipy.sparse import issparse, coo_array, csr_array

from qif_micro.qif.datatypes import Channel
from qif_micro.qif._utils import _duplicate_indices

def reduced(ch: Channel) -> Channel:
    """
    Reduce the columns of a channel by merging those that can be combined
    without altering the channel’s meaning.

    Warning
    -------
    For channels with many columns this operation can be costly in both
    execution time and, more importantly, memory consumption.

    Parameters
    ----------
    channel : Channel
        The input channel.

    Returns
    -------
    Channel
        A new channel whose columns have been reduced while preserving the
        original probabilistic semantics.

    Warning
    -------
    For channels with many columns, this may be expensive, both in terms
    of execution time and (most importantly) space.

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel

    >>> ch = Channel(csr_array([[1/2, 1/4, 1/8, 1/8], [2/3, 1/6, 1/6, 0]]))
    >>> qif.channel.reduced(ch).dist.toarray()
    array([[0.625     , 0.25      , 0.125     ],
           [0.83333333, 0.16666667, 0.        ]])

    As expected, it also works with dense channel, preserving the dense repr:

    >>> qif.channel.reduced(Channel(ch.dist.toarray())).dist
    array([[0.625     , 0.25      , 0.125     ],
           [0.83333333, 0.16666667, 0.        ]])
    """
    # Keep track if the original channel was sparse or not, to preserve repr.
    keep_sparse = issparse(ch.dist)
    ch_dist = csr_array(ch.dist)
    
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
    to_bytes = lambda col: np.ascontiguousarray(col.toarray()).tobytes()
    
    n_cols = norm_cols.shape[1]
    bytes_to_id = {}
    col_ids = np.empty(n_cols, dtype=np.uint64)
    
    for j in range(n_cols):
        col_bytes = to_bytes(norm_cols[:, j])
        col_ids[j] = bytes_to_id.setdefault(col_bytes, j)
        
    # We now have columns that are scalar multiple of each other
    # identified by a unique hash value. Let's find those indices:
    indices = _duplicate_indices(col_ids)

    # Then we create a matrix that, for each column (hash id)
    # has one's at each index (channel col) that has that hash,
    # and zero everywhere else. We use this to agg the chan cols.
    n_unique_cols = indices.max() + 1
    rows = np.arange(n_cols, dtype=np.uint64)
    cols = indices
    data = np.ones(n_cols, dtype=ch_cols.dtype)
    agg = coo_array((data, (rows, cols)), shape=(n_cols, n_unique_cols)).tocsc()

    reduced_dist = ch_dist @ agg
    return Channel(reduced_dist if keep_sparse else reduced_dist.toarray())
