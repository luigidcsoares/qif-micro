from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

from multimethod import multimethod
from scipy.sparse import coo_array

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist, Strategy

from qif_micro.model import baseline
from qif_micro.model._internal import _mk_long_dataset, _mk_records
from qif_micro.typing import Dataset, Model, Record
from qif_micro._utils import _valid_columns

def build(
    pi: ProbabDist,
    records: Sequence[Record],
    mechanism: Channel,
    baseline_dataset: Dataset,
    sanitised_dataset: Dataset,
    hint: Iterable[str],
    owner_col: str = "owner_id",
) -> Model:
    """
    TODOS
    -----
    Support longitudinal and multiple mechanisms.

    Examples
    --------
    >>> from functools import partial
    >>> import numpy as np
    >>> import polars as pl
    >>> from qif_micro.qif.datatypes import ProbabDist
    >>> from qif_micro import mechanism
    >>> from qif_micro import model

    Consider the following domain of records:

    >>> records = [
    ...     [{"q": 0, "s": 0}],
    ...     [{"q": 0, "s": 1}],
    ...     [{"q": 1, "s": 0}],
    ...     [{"q": 1, "s": 1}],
    ...     [{"q": 2, "s": 0}],
    ...     [{"q": 2, "s": 1}],
    ... ]

    We first construct a prior on the domain of records:

    >>> domain_size = len(records)
    >>> pi = ProbabDist(np.repeat(1/domain_size, domain_size))
    
    Then we construct the record-level mechanism:
     
    >>> rr_q = partial(mechanism.random_response, p=2/3)
    >>> rr_s = partial(mechanism.random_response, p=3/4)
    >>> m = mechanism.record(records, {"q": rr_q, "s": rr_s})

    Finally, we define the input and output datasets that we want to analyse:

    >>> b_dataset = pl.DataFrame({
    ...     "owner_id": [0, 1, 2, 3],
    ...     "q":        [0, 0, 1, 2],
    ...     "s":        [0, 0, 0, 1]
    ... })

    >>> s_dataset = pl.DataFrame({
    ...     "owner_id": [0, 1, 2, 3],
    ...     "q":        [0, 0, 1, 1],
    ...     "s":        [0, 0, 1, 1]
    ... })
    
    >>> baseline_joint, adv_st = model.generic(
    ...     pi, records, m, b_dataset, s_dataset, ["q"]
    ... )

    >>> baseline_joint.dist.toarray()
    array([[0.5 , 0.  , 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25]])

    For ``q = 0`` and ``q = 2``, the adversary would guess the record
    correctly, but the noise makes them guess incorrectly for ``q = 1``:
    
    >>> adv_st.dist.toarray()
    array([[1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1.]])
    """
    # ========================================================================
    # Pre-conditions
    # ========================================================================

    # We assume that each index i in the prior ``pi`` corresponds to the i-th
    # record in the domain of records. So, their length must be the same.
    if pi.dist.shape[0] != len(records):
        raise ValueError("Incompatible prior ``pi`` and ``records``!")

    # Standardise ``hint`` input
    hint = [hint] if isinstance(hint, str) else hint

    # Check the required attributes for the baseline and sanitised datasets
    required = [owner_col, *hint]

    ok, missing = _valid_columns(baseline_dataset, required)
    if not ok: raise ValueError(f"Baseline missing attributes: {missing}")

    ok, missing = _valid_columns(sanitised_dataset, required)
    if not ok: raise ValueError(f"Sanitised missing attributes: {missing}")
    
    # We must also be sure that the baseline and sanitised datasets are
    # compatible, according to the mechanism under analysis, and
    # also with respect to the the domain of records:
    as_df = lambda i, r: pl.DataFrame({"record": [r], "rid": [i]})
    records = (as_df(i, r) for i, r in enumerate(records))
    records_df = pl.concat(records, how="vertical_relaxed")

    baseline_records = (
        _mk_records(baseline_dataset, owner_col)
        .join(records_df, on="record", how="left")
    )
    
    sanitised_records = (
        _mk_records(sanitised_dataset, owner_col)
        .join(records_df, on="record", how="left")
    )

    hyper_mechanism = qif.hyper(pi, mechanism)[1].dist.tocoo()
    data, rows, cols = hyper_mechanism.data, *hyper_mechanism.coords
    hyper_mechanism_df = pl.DataFrame({"row": rows, "col": cols, "p": data})

    transf_records = (
        baseline_records
        .join(sanitised_records, on=owner_col)
        .rename({"rid": "row", "rid_right": "col"})
        .rename({"record": "row_label", "record_right": "col_label"})
        .join(hyper_mechanism_df, on=["row", "col"], how="left")
    )

    # There are three ways of getting a null probability:
    # a. The mapping input record -> output record is impossible
    # b. The input record is not possible according to the prior
    # c. The input or output records are not even in the domain
    if transf_records["p"].has_nulls():
        raise ValueError("Incompatible baseline and sanitised datasets!")

    # ========================================================================
    
    # The adversary's intermediate knowledge on a particular record is given
    # by the sum of the posterior probability of that record, given
    # the observed sanitised records, weighted by the dataset length.
    p_expr = pl.col("p").sum() / sanitised_dataset.height
    delta_metadata = (
        sanitised_records.drop(owner_col, "record")
        .join(hyper_mechanism_df, left_on="rid", right_on="col")
        .group_by("row").agg(p_expr)
        .join(records_df, left_on="row", right_on="rid")
        .sort("row")
    )

    # To construct the hint channel, we follow the same approach implemented
    # in the baseline model, but we re-implement it here so that we can
    # restrict it to the hints that are possible in practice, but still
    # keeping all records possible from the adversary's perspective.
    len_expr = pl.col("record").list.len().alias("len")
    extract_hints_expr = pl.col("record").struct.field(hint)
    hint_label_expr = pl.struct(hint).alias("hint_label")
    hint_expr = pl.col("hint_label").rank("dense").alias("hint") - 1
    p_expr = (pl.len() / pl.col("len").first()).alias("p")

    valid_hints = baseline_dataset.select(hint_label_expr).unique()
    ch_metadata = (
        delta_metadata.with_columns(len_expr)
        .explode("record")
        .select(pl.col("row").alias("record"), extract_hints_expr, "len")
        .select("record", hint_label_expr, "len")
        
        # Filter hints so that we get only the ones that are possible
        # in practice, according to the baseline:
        .join(valid_hints, on="hint_label")

        # Compute the probability of each cell in the channel
        .group_by("record", "hint_label")
        .agg(p_expr)

        # and transform the hint labels into col indices:
        .select("record", hint_expr, "p")
    )

    # The metadata above gives us just a slice of the hint channel,
    # so it is not really a valid channel. But, we can construct from
    # it the baseline joint (which will be a valid joint):
    p_expr = (
        pl.when(pl.col("record_right").is_null()).then(0)
        .otherwise(pl.col("p") / baseline_records.height)
        .alias("p")
    )

    baseline_records = baseline_records.select(pl.col("rid").alias("record"))
    baseline_joint_metadata = (
        ch_metadata
        .join(baseline_records, on="record", how="left", coalesce=False)
        .with_columns(p_expr)
        .group_by("record", "hint").agg(pl.col("p").sum())
    )

    n_rows = baseline_joint_metadata["record"].max() + 1
    n_cols = baseline_joint_metadata["hint"].max() + 1

    data = baseline_joint_metadata["p"].to_numpy()
    rows = baseline_joint_metadata["record"].to_numpy()
    cols = baseline_joint_metadata["hint"].to_numpy()
    coo_repr = (data, (rows, cols))

    baseline_joint_dist = coo_array(coo_repr, shape=(n_rows, n_cols))
    baseline_joint = Joint(baseline_joint_dist.tocsr())
    
    # As for the adversary's strategy, we temporarily add a fake column,
    # just so we can get a proper channel (to rely on the pyqif lib stuff):
    row_sum = pl.col("p").sum()
    remaining_p_expr = (
        # Avoid negative entries due to float errors
        pl.when(row_sum.is_close(1)).then(0)
        .otherwise(1 - row_sum)
        .alias("p")
    )

    remaining_p = (
        ch_metadata
        .group_by("record")
        .agg(remaining_p_expr)
        .filter(pl.col("p") > 0)
    )

    data = np.hstack([
        ch_metadata["p"].to_numpy(),
        remaining_p["p"].to_numpy()
    ])

    rows = np.hstack([
        ch_metadata["record"].to_numpy(),
        remaining_p["record"].to_numpy()
    ])
    
    cols = np.hstack([
        ch_metadata["hint"].to_numpy(),
        np.repeat(n_cols, remaining_p.height)
    ])

    coo_repr = (data, (rows, cols))
    shape = (n_rows, n_cols + (1 if remaining_p.height > 0 else 0))
    hint_ch_dist = coo_array(coo_repr, shape=shape)
    hint_ch = Channel(hint_ch_dist.tocsr())

    delta = ProbabDist(delta_metadata["p"].to_numpy().ravel())
    adv_joint = qif.joint(delta, hint_ch)
    adv_st = Strategy(qif.strategy(adv_joint).dist[:n_cols, :])

    return baseline_joint, adv_st
