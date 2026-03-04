from typing import Any

import numpy as np
from numpy.typing import NDArray

def _duplicate_indices(arr: NDArray[Any]):
    # The following code is based on numpy's implementation of the inverse indices
    # returned by np.unique, which contains the indices in the original array
    # that corresponds to the unique values. We use this as a dense rank.
    #
    # Since this is usually called with sparse channels, meaning that memory
    # is a concern, we use heapsort as it has better space complexity.
    perm = np.argsort(arr, kind="heapsort")
    sorted_indices = arr[perm]
    
    mask = np.empty(sorted_indices.shape, dtype=bool)
    mask[0] = True
    mask[1:] = sorted_indices[1:] != sorted_indices[:-1] # Is i + 1 = i?

    indices = np.empty(mask.shape, dtype=np.intp)
    indices[perm] = np.cumsum(mask) - 1

    return indices
