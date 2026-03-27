import enum

from collections.abc import Sequence
from functools import reduce

import numpy as np

from multimethod import multimethod
from scipy.sparse import issparse, csr_array

from qif_micro.qif.datatypes.typing import Slice

class _ProbabDistError(enum.Enum):
    OK = 0
    NEGATIVE_VALUES = 1
    ROW_SUM_MISMATCH = 2

    
@multimethod
def _is_dist_valid(
    dist: Sequence[Slice],
    is_slice: bool = False
) -> _ProbabDistError:
    inner_data = [s.data if issparse(s) else s for s in dist]
    
    has_negative = np.any([np.any(s < 0) for s in inner_data])
    if has_negative: return _ProbabDistError.NEGATIVE_VALUES

    if is_slice: return _ProbabDistError.OK
    
    reduce_fn = lambda acc, s: acc + s.sum(axis=1)
    row_sum = reduce(reduce_fn, dist[1:], dist[0].sum(axis=1))
    sum_to_one = np.isclose(row_sum, 1).all()
    if not sum_to_one: return _ProbabDistError.ROW_SUM_MISMATCH

    return _ProbabDistError.OK
    

@multimethod
def _is_dist_valid(dist: Slice, is_slice: bool = False) -> _ProbabDistError:
    return _is_dist_valid([dist], is_slice)
