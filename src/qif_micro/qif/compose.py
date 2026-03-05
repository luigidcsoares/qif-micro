import itertools
import math

import numpy as np
import polars as pl

from numpy.typing import NDArray
from scipy.sparse import csr_array, hstack, issparse

from qif_micro.qif.datatypes import Channel
    
def _sparse_parallel(lhs: NDArray[np.floating], rhs: NDArray[np.floating]):
    # The following implements row-wise outer product (parallel composition)
    # for scipy sparse matrices. There isn't yet an official implementation in scipy,
    # so this was obtained from: https://stackoverflow.com/questions/57099722/row-wise-outer-product-on-sparse-matrices
    lhs_data_by_row = np.split(lhs.data, lhs.indptr[1:-1])
    rhs_data_by_row = np.split(rhs.data, rhs.indptr[1:-1])
    zip_data_by_row = zip(lhs_data_by_row, rhs_data_by_row)

    mk_data = lambda a, b: np.outer(a, b).ravel()
    data_by_row = [mk_data(a, b) for a, b in zip_data_by_row]
    data = np.concatenate(data_by_row)
    
    lhs_indices_by_row = np.split(lhs.indices, lhs.indptr[1:-1])
    rhs_indices_by_row = np.split(rhs.indices, rhs.indptr[1:-1])
    zip_indices_by_row = zip(lhs_indices_by_row, rhs_indices_by_row)

    indices_by_row = [
        np.column_stack((np.repeat(a, b.shape[0]), np.tile(b, a.shape[0])))
        for a, b in zip_indices_by_row
    ]

    row_len = (r.shape[0] for r in indices_by_row)
    indptr = np.fromiter(itertools.chain((0,), row_len), np.uint64).cumsum()

    # The indices at this point may have gaps. This would lead to
    # all-zero columns, which is bad memory-wise, so we remap them: 
    sparse_indices = np.vstack(indices_by_row)
    _, first_pos, indices = np.unique(
        sparse_indices[:, 0] * rhs.shape[1] + sparse_indices[:, 1],
        return_index=True,
        return_inverse=True
    )

    n_rows = lhs.shape[0]
    n_cols = indices.max() + 1
    ch_dist = csr_array((data, indices, indptr), shape=(n_rows, n_cols))

    # Also return the column pairs that correspond to each new column
    return ch_dist, sparse_indices[first_pos]

type Columns = NDArray[int]
type ReturnParallel = (
    Channel
    | [Channel, Columns | int]
    | [Channel, Columns, int]
)
    
def parallel(
    lhs: Channel,
    rhs: Channel,
    opt_memory: bool = True,
    n_partitions: int = 1,
    return_cols: bool = False,
    return_n_opt: bool = False
) -> ReturnParallel:
    """
    Parallel composition of two channels ``lhs`` and ``rhs``.

    Parameters
    ----------
    lhs : Channel
        The left‑hand side channel to be composed.

    rhs : Channel
        The right‑hand side channel to be composed.

    opt_memory : bool, optional (default: ``True``)
        If sparse channels are involved (otherwise, this is ignored):
        - By default (``True``) the parallel optimisation is enabled.  
        - When ``False`` the function assumes that memory is not a
          concern and disables the optimisation.

    n_partitions : int, optional (default: ``1```)
        If sparse channels are involved (otherwise, this is ignored),
        controls the number of partitions to use for ``lhs`` when the
        result of the parallel composition would be too large.

    return_cols : bool, optional (default: ``False``)
        If ``True`` and sparse channels are involved, the function returns
        the number column pairs as labels. If optimisation is enabled and
        some columns have been simplified, these will be paired with -1.

    return_n_opt : bool, optional (default: ``False``)
        If ``True`` and sparse channels are involved, the function returns
        the number of output columns that were reduced. This value can be
        used to slice the resulting channel into two parts. The optimiser
        always places the optimised columns at the beginning of the output.

    Returns
    -------
    Channel
        The result of the parallel composition of ``lhs`` and ``rhs``.

    tuple (Channel, Columns | int)
        - The result of the parallel composition;
        - If ``return_columns`` enabled: The columns labels (pairs) OR
          If ``return_n_opt`` enabled: the number of columns reduced
          (including both ``lhs`` and  `rhs``; can be used to slice the ch).

    tuple (Channel, Columns, int)
        - The result of the parallel composition;
        - The column labels (pairs);
        - The number of columns reduced.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> from qif_micro.qif.compose import parallel
    >>> from qif_micro.qif.datatypes import Channel

    Let us first consider the case of a dense representation:
    
    >>> lhs = Channel(np.array([[1/2, 1/4, 0, 1/4], [0, 1/6, 2/3, 1/6]]))
    >>> rhs = Channel(np.array([[2/3, 1/6, 1/6], [2/3, 1/3, 0]]))
    >>> parallel(lhs, rhs).dist
    array([[0.33333333, 0.08333333, 0.08333333, 0.16666667, 0.04166667,
                0.04166667, 0.        , 0.        , 0.        , 0.16666667,
                0.04166667, 0.04166667],
               [0.        , 0.        , 0.        , 0.11111111, 0.05555556,
                0.        , 0.44444444, 0.22222222, 0.        , 0.11111111,
                0.05555556, 0.        ]])

    Now notice how columns are reduced with a sparse repr:
    
    >>> lhs = Channel(csr_array(lhs.dist))
    >>> rhs = Channel(csr_array(rhs.dist))
    >>> parallel(lhs, rhs).dist.toarray()
    array([[0.5       , 0.        , 0.16666667, 0.04166667, 0.04166667,
                0.16666667, 0.04166667, 0.04166667],
               [0.        , 0.66666667, 0.11111111, 0.05555556, 0.        ,
                0.11111111, 0.05555556, 0.        ]])

    It is possible to retrieve the column pairs, noting that columns that
    have been reduced (on either side) will be paired with -1:

    >>> parallel(lhs, rhs, return_cols=True)[1]
    array([[ 0, -1],
           [ 2, -1],
           [ 1,  0],
           [ 1,  1],
           [ 1,  2],
           [ 3,  0],
           [ 3,  1],
           [ 3,  2]])

    And it is possible to retrieve the count of reduced columns:

    >>> parallel(lhs, rhs, return_n_opt=True)[1]
    2
    """
    # Pre-condition: number of rows must match
    n_rows = lhs.dist.shape[0]
    if rhs.dist.shape[0] != n_rows:
        raise ValueError("Number of rows do not match!")

    # If channel is not sparse, memory is not a concern,
    # so just keep in the numpy realm:
    if not (issparse(lhs.dist) or issparse(rhs.dist)):
        parallel_dist = np.einsum("xy,xz->xyz", lhs.dist, rhs.dist)
        return Channel(parallel_dist.reshape(n_rows, -1))
        
    # Retrieve the inner objects and convert to sparse if necessary:
    lhs = lhs.dist if issparse(lhs.dist) else csr_array(lhs.dist)
    rhs = rhs.dist if issparse(rhs.dist) else csr_array(rhs.dist)

    # If memory is not a concern (even though channels are sparse),
    # just do the parallel composition without any optimisation.
    if not opt_memory:
        ch_dist, cols = _sparse_parallel(lhs, rhs)
        return (Channel(ch_dist), cols) if return_cols else Channel(ch_dist)
    
    # Otherwise, parallel optimisation is enabled.
    # 
    # We start by finding which columns in the lhs have exactly one non-zero cell,
    # so that we can optimise them memory-wise by anticipating a reduction.
    # 
    # And then we split the lhs into two sub-channels,
    # one with the cols that we can optimise and the other with the remanining cols. 
    nz_per_col = lhs.count_nonzero(axis=0)
    
    determ_nz_cols_lhs = np.nonzero(nz_per_col == 1)[0]
    probab_nz_cols_lhs = np.nonzero(nz_per_col > 1)[0]
    
    lhs, reduced_lhs = lhs[:, probab_nz_cols_lhs], lhs[:, determ_nz_cols_lhs]

    # We then repeat for the rhs, but only those columns that do not match
    # with the reduced columns from the lhs (that is, non-zero at different index).
    # Otherwise we could be "reducing twice" the same columns of the parallel comp.
    excluded_rows = np.unique(reduced_lhs.nonzero()[0])
    all_rows = np.arange(n_rows)
    safe_rows = np.setdiff1d(all_rows, excluded_rows)
    
    nz_per_col = rhs.count_nonzero(axis=0)
    nz_per_col_safe = rhs[safe_rows, :].count_nonzero(axis=0)
    safe_cols = nz_per_col == nz_per_col_safe
    
    determ_nz_cols_rhs = np.nonzero((nz_per_col == 1) & safe_cols)[0]
    probab_nz_cols_rhs = np.nonzero((nz_per_col > 1) | ~safe_cols)[0]
    
    rhs, reduced_rhs = rhs[:, probab_nz_cols_rhs], rhs[:, determ_nz_cols_rhs]
    
    # We then compute the partial parallel composition
    n_partitions = min(n_partitions, lhs.shape[1])
    part_size = math.ceil(lhs.shape[1] / n_partitions)

    part_indptr = [i*part_size for i in range(n_partitions)] + [lhs.shape[1]]
    part_ranges = zip(part_indptr[:-1], part_indptr[1:])
    partitions, partition_cols = zip(*[
        _sparse_parallel(lhs[:, i:j], rhs)
        for i, j in part_ranges
    ])
    
    # Finally, we combine column-wise the reduced and unreduced slices.
    # Pos-condition: optimised slice goes into the beginning of the matrix
    parallel_dist = hstack([reduced_lhs, reduced_rhs, *partitions])
    ch = Channel(parallel_dist)

    # Number of optimised columns, considering both sides of the composition
    # This can be used to split the channel into opt and non-opt slices.
    n_opt = reduced_lhs.shape[1] + reduced_rhs.shape[1]

    # Finally, we construct the column pairs. In the case of the optimised
    # columns, we pair them with -1.
    cols_opt = np.column_stack((
        np.hstack([determ_nz_cols_lhs, determ_nz_cols_rhs]),
        np.repeat(-1, n_opt)
    ))

    # For the columns that haven't been optimised, we got the pairs
    # from _sparse_parallel, but we need to remap the columns, as
    # _sparse_parallel received slices of the original channels.
    cols_unreduced = np.vstack(partition_cols)
    cols_unreduced[:, 0] = probab_nz_cols_lhs[cols_unreduced[:, 0]]
    cols_unreduced[:, 1] = probab_nz_cols_rhs[cols_unreduced[:, 1]]

    cols = np.vstack([cols_opt, cols_unreduced])

    if return_cols and return_n_opt: return ch, cols, n_opt
    if return_cols: return ch, cols
    if return_n_opt: return ch, n_opt

    return ch
