from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_array

from qif_micro.qif.datatypes import Channel, Joint, ProbabDist

def _strategy(belief: ProbabDist | Joint) -> csr_array:
    dist = (
        belief.dist if isinstance(belief, Joint)
        else csr_array(belief.dist[:, np.newaxis])
    )
    
    rows, cols = dist.nonzero()
    col_max = dist.max(axis=0).toarray()
    
    mask_data = dist[rows, cols] == col_max[cols]
    mask = csr_array((mask_data, (rows, cols)), shape=dist.shape)
    max_counts = mask.sum(axis=0)

    st_data = mask_data / max_counts[mask.indices]
    st_dist = csr_array((st_data, mask.indices, mask.indptr), shape=dist.shape)

    # It could be that the input is a joint with all-zero columns,
    # in which case there must be a strategy (uniform over all rows):
    nz_per_col = dist.count_nonzero(axis=0)
    allzero_cols = np.nonzero(nz_per_col == 0)[0]

    n_allzero = allzero_cols.shape[0]
    if n_allzero == 0: return st_dist

    st_dist = st_dist.tocoo()
    st_data = st_dist.data
    st_rows, st_cols = st_dist.coords

    allzero_data = np.repeat(1 / n_allzero, n_allzero * dist.shape[0])
    allzero_rows, allzero_cols = zip(*(
        (r, c) for c in allzero_cols for r in range(dist.shape[0])
    ))

    st_data = np.concatenate([st_data, allzero_data])
    st_rows = np.concatenate([st_rows, allzero_rows])
    st_cols = np.concatenate([st_cols, allzero_cols])

    return coo_array((st_data, (st_rows, st_cols)), shape=dist.shape).tocsr()


type MapOwners = pl.DataFrame | pl.LazyFrame
type MapLabels = pl.DataFrame | pl.LazyFrame
type Model = tuple[ProbabDist, Channel, MapOwners, MapLabels]

def linkage_risk(
    dataset: Model | Sequence[Model],
    baseline: Model | Sequence[Model] | None = None
) -> np.floating:
    pass
