import itertools
import math

from typing import Any

import numpy as np
import polars as pl

from numpy.typing import NDArray
from scipy.sparse import csr_array, hstack, issparse

from qif_micro.qif.datatypes import Channel

def _duplicate_indices(arr: NDArray[Any]):
    # The following code is based on numpy's implementation of the inverse indices
    # returned by np.unique, which contains the indices in the original array
    # that corresponds to the unique values. We use this as a dense rank.
    #
    # Since this is usually called with sparse channels, meaning that memory
    # is a concern, we use heapsort as it has better space complexity.
    perm = np.argsort(arr, kind="heapsort")
    sorted_indices = arr[perm] # Should be just a view, no extra memory
    
    mask = np.empty(sorted_indices.shape, dtype=bool)
    mask[0] = True
    mask[1:] = sorted_indices[1:] != sorted_indices[:-1] # Is i + 1 = i?

    indices = np.empty(mask.shape, dtype=np.intp)
    indices[perm] = np.cumsum(mask) - 1

    return indices
    

def _unreduced_parallel(lhs: NDArray[np.floating], rhs: NDArray[np.floating]):
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

    m, n = lhs.shape[1], rhs.shape[1]
    mk_indices = lambda a, b: np.ravel_multi_index(np.ix_(a, b), (m, n)).ravel()
    indices_by_row = [mk_indices(a, b) for a, b in zip_indices_by_row]

    indptr = np.fromiter(itertools.chain((0,), map(len, indices_by_row)), int).cumsum()

    # The indices at this point may have gaps. This would lead to
    # all-zero columns, which is bad memory-wise, so we remap them: 
    indices = _duplicate_indices(np.concatenate(indices_by_row))

    n_rows = lhs.shape[0]
    n_cols = indices.max() + 1
    return csr_array((data, indices, indptr), shape=(n_rows, n_cols))

    
def parallel(
    lhs: Channel,
    rhs: Channel,
    *,
    opt_memory: bool = True,
    n_partitions: int = 1,
    return_n_opt: bool = False
) -> Channel:
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

    return_n_opt : bool, optional (default: ``False``)
        If ``True`` and sparse channels are involved, the function returns
        the number of output columns that were reduced. This value can be
        used to slice the resulting channel into two parts. The optimiser
        always places the optimised columns at the beginning of the output.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> from qif_micro.qif.compose import parallel
    >>> from qif_micro.qif.datatypes import Channel

    Let us first consider the case of a dense representation:
    
    >>> lhs = Channel(np.array([[1/2, 1/2], [0, 1]]))
    >>> rhs = Channel(np.array([[1/3, 2/3], [1, 0]]))
    >>> parallel(lhs, rhs).dist
    array([[0.16666667, 0.33333333, 0.16666667, 0.33333333],
           [0.        , 0.        , 1.        , 0.        ]])

    Now notice how columns are reduced with a sparse repr:
    
    >>> lhs = Channel(csr_array(lhs.dist))
    >>> rhs = Channel(csr_array(rhs.dist))
    >>> parallel(lhs, rhs).dist.toarray()
    array([[0.5       , 0.16666667, 0.33333333],
           [0.        , 1.        , 0.        ]])

    And it is possible to retrieve the count of reduced columns:
    >>> parallel(lhs, rhs, return_n_opt=True)[1]
    1
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
    if not opt_memory: return Channel(_unreduced_parallel(lhs, rhs))
    
    # Otherwise, parallel optimisation is enabled.
    # 
    # We start by finding which columns in the lhs have exactly one non-zero cell,
    # so that we can optimise them memory-wise by anticipating a reduction.
    # 
    # And then we split the lhs into two sub-channels,
    # one with the cols that we can optimise and the other with the remanining cols. 
    nz_per_col = lhs.count_nonzero(axis=0)
    
    determ_nz_cols = np.nonzero(nz_per_col == 1)[0]
    probab_nz_cols = np.nonzero(nz_per_col > 1)[0]
    
    lhs, reduced_lhs = lhs[:, probab_nz_cols], lhs[:, determ_nz_cols]

    # We then repeat for the rhs, but only those columns that do not match
    # with the reduced columns from the lhs (that is, non-zero at different index).
    # Otherwise we could be "reducing twice" the same columns of the parallel comp.
    excluded_rows = np.unique(reduced_lhs.nonzero()[0])
    all_rows = np.arange(n_rows)
    safe_rows = np.setdiff1d(all_rows, excluded_rows)
    
    nz_per_col = rhs.count_nonzero(axis=0)
    nz_per_col_safe = rhs[safe_rows, :].count_nonzero(axis=0)
    safe_cols = nz_per_col == nz_per_col_safe
    
    determ_nz_cols = np.nonzero((nz_per_col == 1) & safe_cols)[0]
    probab_nz_cols = np.nonzero((nz_per_col > 1) | ~safe_cols)[0]
    
    rhs, reduced_rhs = rhs[:, probab_nz_cols], rhs[:, determ_nz_cols]
    
    # We then compute the partial parallel composition
    n_partitions = min(n_partitions, lhs.shape[1])
    part_size = math.ceil(lhs.shape[1] / n_partitions)

    part_indptr = [i*part_size for i in range(n_partitions)] + [lhs.shape[1]]
    part_ranges = zip(part_indptr[:-1], part_indptr[1:])
    parts = [_unreduced_parallel(lhs[:, i:j], rhs) for i, j in part_ranges]
    
    # Finally, we combine column-wise the reduced and unreduced slices of the parallel composition:
    # Pos-condition: optimised slice goes into the beginning of the matrix (first cols)
    parallel_dist = hstack([reduced_lhs, reduced_rhs, *parts])
    ch = Channel(parallel_dist)

    # Number of optimised columns, considering both sides of the composition
    # This can be used to split the channel into opt and non-opt slices.
    n_opt = reduced_lhs.shape[1] + reduced_rhs.shape[1]

    return (ch, n_opt) if return_n_opt else ch
