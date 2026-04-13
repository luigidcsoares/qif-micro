from collections.abc import Sequence
from typing import TypeIs

import numpy as np
import scipy.sparse as sp

type Slice = np.ndarray | sp.csc_array | sp.csr_array


def is_partitioned(x: Slice | Sequence[Slice]) -> TypeIs[Sequence[Slice]]:
    return isinstance(x, Sequence)
