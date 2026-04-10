from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import reduce

from multimethod import multimethod
import numpy as np
import scipy.sparse as sp

from qif_micro.qif.datatypes.typing import Slice

class _Error(Enum):
    NEGATIVE_VALUES = 1
    ROW_SUM_MISMATCH = 2


@dataclass(frozen=True)
class _Check:
    error: _Error | None = None
    row_sum: np.ndarray | None = None

    
@multimethod
def _is_dist_valid(dist: Sequence[Slice]) -> _Check:
    for s in dist:
        data = s.data if sp.issparse(s) else s
        if np.any(data < 0): return _Check(error=_Error.NEGATIVE_VALUES)

    reduce_fn = lambda acc, s: acc + s.sum(axis=1)
    row_sum = reduce(reduce_fn, dist[1:], dist[0].sum(axis=1))
    sum_is_one = np.isclose(row_sum, 1)
    sum_exceeds_one = ((row_sum > 1) & ~sum_is_one).any()
    if sum_exceeds_one: return _Check(error=_Error.ROW_SUM_MISMATCH)

    return _Check(row_sum=row_sum)
    

@multimethod
def _is_dist_valid(dist: Slice) -> _Error: return _is_dist_valid([dist])
