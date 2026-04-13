from itertools import chain

import numpy as np
import scipy.sparse as sp

from qif_micro.qif.datatypes import Channel
from qif_micro.qif.datatypes.typing import is_partitioned


def _dense_parallel(lhs: np.ndarray, rhs: np.ndarray):
    n_rows = lhs.shape[0]
    assert n_rows == rhs.shape[0]

    # Clean-up all-zero columns
    dist = np.einsum("xy,xz->xyz", lhs, rhs).reshape(n_rows, -1)
    keep = dist.any(axis=0)

    return dist[:, keep]

    
def _sparse_parallel(lhs: sp.csr_array, rhs: sp.csr_array, return_cols: bool):
    n_rows = lhs.shape[0]
    assert n_rows == rhs.shape[0]
    
    # The following implements row-wise outer product (parallel composition)
    # for scipy sparse matrices. There isn't yet an official implementation in scipy,
    # so this was obtained from: https://stackoverflow.com/questions/57099722/row-wise-outer-product-on-sparse-matrices
    split_lhs = lhs.indptr[1:-1]
    split_rhs = rhs.indptr[1:-1]

    lhs_data_by_row = np.split(lhs.data, split_lhs)
    rhs_data_by_row = np.split(rhs.data, split_rhs)
    zip_data_by_row = zip(lhs_data_by_row, rhs_data_by_row)

    def mk_data(a, b): return np.outer(a, b).ravel()
    data_by_row = [mk_data(a, b) for a, b in zip_data_by_row]
    data = np.concatenate(data_by_row)
    
    lhs_indices_by_row = np.split(lhs.indices, split_lhs)
    rhs_indices_by_row = np.split(rhs.indices, split_rhs)
    zip_indices_by_row = zip(lhs_indices_by_row, rhs_indices_by_row)

    def broadcast_lhs(a, b): return np.repeat(a, b.shape[0])
    def broadcast_rhs(a, b): return np.tile(b, a.shape[0])
    def mk_col_pairs(a, b): return (broadcast_lhs(a, b), broadcast_rhs(a, b))
    def mk_indices(a, b): return np.column_stack(mk_col_pairs(a, b))
    indices_by_row = [mk_indices(a, b) for a, b in zip_indices_by_row]

    row_len = (r.shape[0] for r in indices_by_row)
    indptr = np.fromiter(chain((0,), row_len), np.uint64).cumsum()

    # The indices at this point may have gaps. This would lead to
    # all-zero columns, which is bad memory-wise, so we remap them: 
    sparse_indices = np.vstack(indices_by_row)
    flat_indices = sparse_indices[:, 0] * rhs.shape[1] + sparse_indices[:, 1]

    if not return_cols:
        _, indices = np.unique(flat_indices, return_inverse=True)
        n_cols = indices.max() + 1
        ch_dist = sp.csr_array((data, indices, indptr), shape=(n_rows, n_cols))
        return ch_dist, np.empty(shape=(0, 2), dtype=np.uint64)
        
    _, first_pos, indices = np.unique(
        flat_indices,
        return_index=True,
        return_inverse=True
    )

    n_cols = indices.max() + 1
    ch_dist = sp.csr_array((data, indices, indptr), shape=(n_rows, n_cols))

    # Also return the column pairs that correspond to each new column
    return ch_dist, sparse_indices[first_pos]


def parallel(
    lhs: Channel,
    rhs: Channel,
    opt_memory: bool = True,
    return_cols: bool = False,
) -> Channel | tuple[Channel, np.ndarray]:
    """
    Parallel composition of two channels ``lhs`` and ``rhs``.

    In parallel composition, each row of the result corresponds to the
    Cartesian product of outputs from both channels.

    Parameters
    ----------
    lhs : Channel
        The left‑hand side channel to be composed.

    rhs : Channel
        The right‑hand side channel to be composed.

    opt_memory : bool, optional (default: True)
        If sparse channels are involved (otherwise, this is ignored):
        - By default (``True``) the parallel optimisation is enabled.  
        - When ``False`` the function assumes that memory is not a
          concern and disables the optimisation.

    return_cols : bool, optional (default: False)
        If ``True`` and sparse channels are involved, the function returns
        the number column pairs as labels. If optimisation is enabled and
        some columns have been simplified, these will be paired with -1.

    Returns
    -------
    Channel
        The result of the parallel composition of ``lhs`` and ``rhs``.

    tuple (Channel, np.ndarray | int)
        - The result of the parallel composition;
        - If ``return_columns`` enabled: The columns labels (pairs) OR

    tuple (Channel, np.ndarray, int)
        - The result of the parallel composition;
        - The column labels (pairs);
        - The number of columns reduced.

    Notes
    -----
    Partitioned channels (sequences of channel slices) are supported. In such
    cases, the result will also be partitioned by the Cartesian product of
    output partitions from both channels.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from qif_micro import qif
    >>> from qif_micro.qif.datatypes import Channel

    Let us first consider the case of a dense representation:
    
    >>> lhs = Channel(np.array([[1/2, 1/4, 0, 1/4], [0, 1/6, 2/3, 1/6]]))
    >>> rhs = Channel(np.array([[2/3, 1/6, 1/6], [2/3, 1/3, 0]]))
    >>> qif.compose.parallel(lhs, rhs).dist
    array([[0.33333333, 0.08333333, 0.08333333, 0.16666667, 0.04166667,
            0.04166667, 0.        , 0.        , 0.16666667, 0.04166667,
            0.04166667],
           [0.        , 0.        , 0.        , 0.11111111, 0.05555556,
            0.        , 0.44444444, 0.22222222, 0.11111111, 0.05555556,
            0.        ]])

    Now notice how columns are reduced with a sparse repr:
    
    >>> lhs = Channel(sp.csr_array(lhs.dist))
    >>> rhs = Channel(sp.csr_array(rhs.dist))
    >>> sp.hstack(parallel(lhs, rhs).dist).toarray()
    array([[0.5       , 0.        , 0.16666667, 0.04166667, 0.04166667,
                0.16666667, 0.04166667, 0.04166667],
               [0.        , 0.66666667, 0.11111111, 0.05555556, 0.        ,
                0.11111111, 0.05555556, 0.        ]])

    It is possible to retrieve the column pairs, noting that columns that
    have been reduced (on either side) will be paired with -1:

    >>> qif.compose.parallel(lhs, rhs, return_cols=True)[1]
    array([[ 0, -1],
           [ 2, -1],
           [ 1,  0],
           [ 1,  1],
           [ 1,  2],
           [ 3,  0],
           [ 3,  1],
           [ 3,  2]])
    """
    # If any of the partitions is sparse, treat all of them as sparse.
    lhs_dist = lhs.dist if is_partitioned(lhs.dist) else [lhs.dist]
    rhs_dist = rhs.dist if is_partitioned(rhs.dist) else [rhs.dist]

    is_lhs_sparse = np.any([sp.issparse(s) for s in lhs_dist])
    is_rhs_sparse = np.any([sp.issparse(s) for s in rhs_dist])

    # Pre-condition: number of rows must match
    n_rows = lhs_dist[0].shape[0]
    if rhs_dist[0].shape[0] != n_rows:
        raise ValueError("Number of rows do not match!")

    # If channel is not sparse, memory is not a concern,
    # so just keep in the numpy realm:
    if not (is_lhs_sparse or is_rhs_sparse):
        par_dist = [
            _dense_parallel(s_lhs_dist, s_rhs_dist)
            for s_lhs_dist in lhs_dist
            for s_rhs_dist in rhs_dist
        ]

        par_dist = par_dist if len(par_dist) > 1 else par_dist[0]
        return Channel(par_dist)
        
    # Make both sparses, if one of them is not:
    lhs_dist = list(map(sp.csr_array, lhs_dist))
    rhs_dist = list(map(sp.csr_array, rhs_dist))

    # If memory is not a concern (even though channels are sparse),
    # just do the parallel composition without any optimisation.
    if not opt_memory:
        result_it = (
            _sparse_parallel(s_lhs_dist, s_rhs_dist, return_cols)
            for s_lhs_dist in lhs_dist
            for s_rhs_dist in rhs_dist
        )

        par_dist, cols = zip(*result_it)
        par_dist = par_dist if len(par_dist) > 1 else par_dist[0]
        cols = np.vstack(cols)

        ch = Channel(par_dist)
        return (ch, cols) if return_cols else ch
    
    # Otherwise, parallel optimisation is enabled.
    # 
    # We start by finding which columns in the lhs_dist have exactly one non-zero cell,
    # so that we can optimise them memory-wise by anticipating a reduction.
    # 
    # And then we split the lhs_dist into two sub-channels,
    # one with the cols that we can optimise and the other with the remanining cols. 
    nz_per_col = [s.count_nonzero(axis=0) for s in lhs_dist]
    
    determ_nz_cols_lhs = [np.nonzero(s == 1)[0] for s in nz_per_col]
    probab_nz_cols_lhs = [np.nonzero(s > 1)[0] for s in nz_per_col]
    
    reduced_lhs_dist = [
        lhs_dist[i][:, c]
        for i, c in enumerate(determ_nz_cols_lhs)
    ]

    reduced_lhs_dist = sp.hstack(reduced_lhs_dist)

    unreduced_lhs_dist = [
        lhs_dist[i][:, s_cols] for i, s_cols in enumerate(probab_nz_cols_lhs)
        if s_cols.size > 0
    ]

    # We then repeat for the rhs_dist, but only those columns that do not match
    # with the reduced columns from the lhs_dist (that is, non-zero at different index).
    # Otherwise we could be "reducing twice" the same columns of the parallel comp.
    excluded_rows = np.unique(reduced_lhs_dist.nonzero()[0])
    all_rows = np.arange(n_rows)
    safe_rows = np.setdiff1d(all_rows, excluded_rows, assume_unique=True)
    
    nz_per_col = [s.count_nonzero(axis=0) for s in rhs_dist]
    nz_per_col_safe = [s[safe_rows, :].count_nonzero(axis=0) for s in rhs_dist]
    safe_cols = [
        s_nz == s_safe
        for s_nz, s_safe in zip(nz_per_col, nz_per_col_safe)
    ]
    
    determ_nz_cols_rhs = [
        np.nonzero((s_nz == 1) & s_safe)[0]
        for s_nz, s_safe in zip(nz_per_col, safe_cols)
    ]

    probab_nz_cols_rhs = [
        np.nonzero((s_nz > 1) | ~s_safe)[0]
        for s_nz, s_safe in zip(nz_per_col, safe_cols)
    ]

    reduced_rhs_dist = sp.hstack([
        rhs_dist[i][:, c]
        for i, c in enumerate(determ_nz_cols_rhs)
    ])

    unreduced_rhs_dist = [
        rhs_dist[i][:, s_cols] for i, s_cols in enumerate(probab_nz_cols_rhs)
        if s_cols.size > 0
    ]

    pairs = (
        (s_lhs_dist, s_rhs_dist)
        for s_lhs_dist in unreduced_lhs_dist
        for s_rhs_dist in unreduced_rhs_dist
    )

    result_it = (_sparse_parallel(*p, return_cols) for p in pairs)
    par_dist, cols = zip(*result_it)

    # We keep the two reduced slices as two separate partitions:
    par_dist = [s for s in [reduced_lhs_dist, reduced_rhs_dist, *par_dist] if s.nnz > 0]
    par_dist = par_dist if len(par_dist) > 1 else par_dist[0]

    # We need to remap the columns, as _sparse_parallel received slices
    # of the original channels.
    k = 0
    for i in range(len(lhs_dist)):
        if probab_nz_cols_lhs[i].size == 0: continue

        for j in range(len(rhs_dist)):
            if probab_nz_cols_rhs[j].size == 0: continue

            cols[k][:, 0] = probab_nz_cols_lhs[i][cols[k][:, 0]]
            cols[k][:, 1] = probab_nz_cols_rhs[j][cols[k][:, 1]]

            # Skip column 0 in the lhs_dist, as there's no offset:
            if i > 0: cols[k][:, 0] += lhs_dist[i - 1].shape[1]

            # Skip column 0 in the rhs_dist, as there's no offset
            if j > 0: cols[k][:, 1] += rhs_dist[j - 1].shape[1]

            k += 1
    
    # Then we compute the pairs of columns for the reduced slices:
    cols_reduced_lhs_dist = []
    cols_reduced_rhs_dist = []

    for i in range(len(lhs_dist)):
        if determ_nz_cols_lhs[i].size == 0: continue

        s_cols = determ_nz_cols_lhs[i]
        s_cols += 0 if i == 0 else lhs_dist[i - 1].shape[1]

        fill_rhs_dist = np.repeat(-1, s_cols.shape[0])
        cols_reduced_lhs_dist.append(np.column_stack((s_cols, fill_rhs_dist)))

    for i in range(len(rhs_dist)):
        if determ_nz_cols_rhs[i].size == 0: continue

        s_cols = determ_nz_cols_rhs[i]
        s_cols += 0 if i == 0 else rhs_dist[i - 1].shape[1]

        fill_lhs_dist = np.repeat(-1, s_cols.shape[0])
        cols_reduced_rhs_dist.append(np.column_stack((fill_lhs_dist, s_cols)))

        
    ch = Channel(par_dist)
    cols = np.vstack([*cols_reduced_lhs_dist, *cols_reduced_rhs_dist, *cols])
    return (ch, cols) if return_cols else ch
