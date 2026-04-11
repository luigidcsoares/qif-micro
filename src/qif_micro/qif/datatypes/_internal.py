from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import operator

import numpy as np
import scipy.sparse as sp

from .typing import Slice


class _Error(Enum):
    NEGATIVE_VALUES = 1
    AXIS_SUM_MISMATCH = 2


@dataclass(frozen=True)
class _Check:
    error: _Error | None = None
    axis_sum: np.ndarray | None = None


def _is_dist_valid(
    dist: Sequence[Slice],
    dist_orient: int | None = None
) -> _Check:
    assert (dist_orient is None) or (dist_orient in [0, 1])

    for s in dist:
        data = s.data if sp.issparse(s) else s
        if np.any(data < 0): return _Check(error=_Error.NEGATIVE_VALUES)

    # For dist_orient=1 (Channel/Joint): add row sums across partitions
    # For dist_orient=0 (Hyper): check each partition's column sums
    # independently (don't add across partitions)
    reduce_fn = lambda acc, s: acc + s.sum(axis=dist_orient)
    axis_sum = [s.sum(axis=dist_orient) for s in dist]
    if dist_orient == 0: axis_sum = np.hstack(axis_sum)
    else: axis_sum = reduce(operator.add, axis_sum)

    sum_is_one = np.isclose(axis_sum, 1)
    sum_exceeds_one = ((axis_sum > 1) & ~sum_is_one).any()
    if sum_exceeds_one: return _Check(error=_Error.AXIS_SUM_MISMATCH)

    return _Check(axis_sum=axis_sum)
