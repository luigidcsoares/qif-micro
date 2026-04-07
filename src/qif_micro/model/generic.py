from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

from multimethod import multimethod
from scipy.sparse import coo_array, csr_array, hstack

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist, Strategy

from qif_micro.model import baseline
from qif_micro.model._internal import _mk_long_dataset, _mk_records
from qif_micro.typing import DataFrame, Model, Record
from qif_micro._utils import _valid_columns

@multimethod
def build(
    pi: ProbabDist,
    records: DataFrame,
    mechanism: Channel,
    baseline_dataset: DataFrame,
    sanitised_dataset: DataFrame,
    hint: Iterable[str],
    owner_col: str = "owner_id",
    record_col: str = "record_id",
    entry_col: str = "entry_id",
) -> Model:
    """
    This function builds a generic adversary model from records and datasets.

    This function is overloaded:

    - ``build(pi, records, ...)``: accepts a :class:`pl.DataFrame`
    - ``build(pi, records, ...)``: accepts an iterable of records

    TODOS
    -----
    Support longitudinal and multiple mechanisms.

    Parameters
    ----------
    pi : ProbabDist
        Prior distribution over the domain of records.

    records : DataFrame
        A DataFrame where each row represents an entry in a record.
        The DataFrame must have columns ``record_col`` and ``entry_col``
        (by default ``"record_id"`` and ``"entry_id"``) identifying each
        record and entry. Other columns represent the record attributes.

    mechanism : Channel
        The mechanism to apply to the records.

    baseline_dataset : DataFrame
        The baseline dataset containing the actual records.

    sanitised_dataset : DataFrame
        The sanitised dataset after applying the mechanism.

    hint : str | iterable of str
        Column names that represent the adversary's auxiliary information.

    owner_col : str, optional (default: "owner_id")
        Column name for the owner identifier.

    record_col : str, optional (default: "record_id")
        Column name for the record identifier.

    entry_col : str, optional (default: "entry_id")
        Column name for the entry identifier within each record.

    Returns
    -------
    tuple[Joint, Strategy]
        A pair (baseline_joint, adv_st) representing:
        - baseline_joint: the baseline joint distribution
        - adv_st: the adversary's strategy (posterior)

    Pre-conditions
    --------------
    - ``records`` must be in "wide" format: each row is an entry of a record,
      columns are record attributes.
    - ``records`` must have ``record_col`` and ``entry_col`` columns identifying
      records and entries (defaults: "record_id", "entry_id").
    - ``baseline_dataset`` and ``sanitised_dataset`` must have ``owner_col``
      and ``entry_col`` columns (defaults: "owner_id", "entry_id").
    - ``pi`` prior size must equal the number of records in the record domain.
    - All ``hint`` columns must exist in records and datasets.
    - The mechanism must be compatible with the records and datasets.
    - Baseline and sanitised datasets must produce compatible records with the
      record domain under the given mechanism.

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
    >>> m = mechanism.record(records, q=rr_q, s=rr_s)

    Finally, we define the input and output datasets that we want to analyse:

    >>> b_dataset = pl.DataFrame({
    ...     "owner_id": [0, 1, 2, 3],
    ...     "entry_id": [0, 0, 0, 0],
    ...     "q":        [0, 0, 1, 2],
    ...     "s":        [0, 0, 0, 1]
    ... })

    >>> s_dataset = pl.DataFrame({
    ...     "owner_id": [0, 1, 2, 3],
    ...     "entry_id": [0, 0, 0, 0],
    ...     "q":        [0, 0, 1, 1],
    ...     "s":        [0, 0, 1, 1]
    ... })
    
    >>> result = model.generic(pi, records, m, b_dataset, s_dataset, ["q"])
    >>> baseline_joint, adv_st = result

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
    array([[1. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0.5, 0.5]])

    The domain of records can also be a DataFrame, in which case each row
    represents an entry and must have ``record_id`` and ``entry_id`` columns:

    >>> records = pl.from_records([
    ...     {"record_id": 0, "entry_id": 0, "q": 0, "s": 0},
    ...     {"record_id": 1, "entry_id": 0, "q": 0, "s": 1},
    ...     {"record_id": 2, "entry_id": 0, "q": 1, "s": 0},
    ...     {"record_id": 3, "entry_id": 0, "q": 1, "s": 1},
    ...     {"record_id": 4, "entry_id": 0, "q": 2, "s": 0},
    ...     {"record_id": 5, "entry_id": 0, "q": 2, "s": 1},
    ... ])

    >>> result = model.generic(pi, records, m, b_dataset, s_dataset, ["q"])
    >>> baseline_joint, adv_st = result
           
    >>> baseline_joint.dist.toarray()
    array([[0.5 , 0.  , 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25]])

    >>> adv_st.dist.toarray()
    array([[1. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0.5, 0.5]])
    """
    # ========================================================================
    # Pre-conditions
    # ========================================================================
    records = records.lazy()
    baseline_dataset = baseline_dataset.lazy()
    sanitised_dataset = sanitised_dataset.lazy()

    # Standardise ``hint`` input
    hint = [hint] if isinstance(hint, str) else hint

    # Check the required attributes for the domain of records
    required = [record_col, entry_col, *hint]
    ok, missing = _valid_columns(records, required)
    if not ok: raise ValueError(f"Records missing attributes: {missing}")

    # We assume that each index i in the prior ``pi`` corresponds to the i-th
    # record in the domain of records. So, their length must be the same.
    n_records = (
        records
        .select(pl.col(record_col).len())
        .collect(engine="streaming")
        .item()
    )
    
    if pi.dist.shape[0] != n_records:
        raise ValueError("Incompatible prior ``pi`` and ``records``!")

    # Check the required attributes for the baseline and sanitised datasets
    required = [owner_col, entry_col, *hint]

    ok, missing = _valid_columns(baseline_dataset, required)
    if not ok: raise ValueError(f"Baseline missing attributes: {missing}")

    ok, missing = _valid_columns(sanitised_dataset, required)
    if not ok: raise ValueError(f"Sanitised missing attributes: {missing}")
    
    # We must also be sure that the baseline and sanitised datasets are
    # compatible, according to the mechanism under analysis, and
    # also with respect to the the domain of records:
    # Convert DataFrame to list of records format first
    id_cols = [record_col, owner_col, entry_col]
    attr_cols = [c for c in records.collect_schema() if c not in id_cols]
    entry_expr = pl.col(entry_col).cast(pl.UInt64)

    records = (
        records
        .sort(record_col, entry_col)
        .select(record_col, entry_expr, *attr_cols)
        .pipe(_mk_records, record_col)
    )

    baseline_records = (
        baseline_dataset
        .sort(owner_col, entry_col)
        .select(owner_col, entry_expr, *attr_cols)
        .pipe(_mk_records, owner_col)
        .join(records, on="record", how="left")
    )
    
    sanitised_records = (
        sanitised_dataset
        .sort(owner_col, entry_col)
        .select(owner_col, entry_expr, *attr_cols)
        .pipe(_mk_records, owner_col)
        .join(records, on="record", how="left")
    )

    # Compute the posteriors and transpose, since the result is a channel
    # with outputs as the rows and we need them as columns.
    hyper_mechanism = qif.hyper(pi, mechanism)[1].dist.T.tocoo()
    data, rows, cols = hyper_mechanism.data, *hyper_mechanism.coords
    hyper_mechanism_df = pl.LazyFrame({"row": rows, "col": cols, "p": data})

    transf_records = (
        baseline_records
        .join(sanitised_records, on=owner_col)
        .rename({record_col: "row", record_col + "_right": "col"})
        .rename({"record": "row_label", "record_right": "col_label"})
        .join(hyper_mechanism_df, on=["row", "col"], how="left")
    )

    # There are three ways of getting a null probability:
    # a. The mapping input record -> output record is impossible
    # b. The input record is not possible according to the prior
    # c. The input or output records are not even in the domain
    has_nulls = (
        transf_records.select(pl.col("p").has_nulls())
        .collect(engine="streaming")
        .item()
    )

    if has_nulls:
        raise ValueError("Incompatible baseline and sanitised datasets!")

    # ========================================================================
    # The adversary's intermediate knowledge on a particular record is given
    # by the sum of the posterior probability of that record, given
    # the observed sanitised records, weighted by the dataset length.
    n_records = (
        sanitised_dataset
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )

    p_expr = pl.col("p").mul("len").sum() / n_records
    delta_metadata = (
        sanitised_records.group_by(record_col).agg(pl.len())
        .join(hyper_mechanism_df, left_on=record_col, right_on="col")
        .group_by("row").agg(p_expr)
        .join(records, left_on="row", right_on=record_col)
        .sort("row")
        .collect(engine="streaming")
    )

    # To construct the hint channel, we follow the same approach implemented
    # in the baseline model, but we re-implement it here so that we can
    # restrict it to the hints that are possible in practice, but still
    # keeping all records possible from the adversary's perspective.
    extract_hints_expr = pl.col("record").struct.field(hint)
    hint_label_expr = pl.struct(hint).alias("hint_label")
    hint_expr = pl.col("hint_label").rank("dense").alias("hint") - 1

    dense_row_expr = pl.col("row").rank("dense").alias("dense_row") - 1
    
    len_expr = pl.col("record").list.len().alias("len")
    p_expr = (pl.len() / pl.col("len").first()).alias("p")
    
    valid_hints = baseline_dataset.lazy().select(hint_label_expr).unique()
    ch_metadata = (
        delta_metadata.lazy()
        .with_columns(len_expr)
        .explode("record")
        .select("row", extract_hints_expr, "len")
        .select("row", hint_label_expr, "len")
        
        # Filter hints so that we get only the ones that are possible
        # in practice, according to the baseline.
        .join(valid_hints, on="hint_label")

        # Compute the probability of each cell in the channel.
        .group_by("row", "hint_label")
        .agg(p_expr)

        # and transform the hint labels into col indices:
        .select("row", dense_row_expr, hint_expr, "p")
        .collect(engine="streaming")
    )
    
    # The metadata above gives us just a slice of the hint channel,
    # so it is not really a valid channel. But, we can construct from
    # it the baseline joint (which will be a valid joint):
    p_expr = (pl.col("p") / n_records).alias("p")

    baseline_records = baseline_records.lazy().rename({record_col: "row"})
    baseline_joint_metadata = (
        ch_metadata.lazy()
        .join(baseline_records, on="row")
        .with_columns(p_expr)
        .group_by("row", "hint")
        .agg(pl.col("p").sum(), pl.col("dense_row").first())
        .collect(engine="streaming")
    )

    n_rows = delta_metadata.height
    n_cols = baseline_joint_metadata["hint"].max() + 1

    data = baseline_joint_metadata["p"].to_numpy()
    cols = baseline_joint_metadata["hint"].to_numpy()
    rows = baseline_joint_metadata["dense_row"].to_numpy()

    coo_repr = (data, (rows, cols))
    baseline_joint_dist = coo_array(coo_repr, shape=(n_rows, n_cols))
    baseline_joint = Joint(baseline_joint_dist.tocsr())

    data = ch_metadata["p"].to_numpy()
    cols = ch_metadata["hint"].to_numpy()
    rows = ch_metadata["dense_row"].to_numpy()

    shape = (n_rows, n_cols)
    coo_repr = (data, (rows, cols))
    dist = coo_array(coo_repr, shape=shape).tocsr()

    hint_ch = Channel(dist, is_slice=True)

    delta = ProbabDist(delta_metadata["p"].to_numpy().ravel())
    adv_joint = qif.joint(delta, hint_ch)
    adv_st = qif.strategy(adv_joint)

    return baseline_joint, adv_st


@multimethod
def build(
    pi: ProbabDist,
    records: list | tuple,
    mechanism: Channel,
    baseline_dataset: DataFrame,
    sanitised_dataset: DataFrame,
    hint: Iterable[str],
    owner_col: str = "owner_id",
    record_col: str = "record_id",
    entry_col: str = "entry_id",
) -> Model:
    as_df = lambda i, r:  pl.LazyFrame(r).with_columns(
        pl.lit(i).alias(record_col),
        pl.row_index(entry_col)
    )

    records = (as_df(i, r) for i, r in enumerate(records))
    records = pl.concat(records, how="diagonal")

    return build(
        pi,
        records,
        mechanism,
        baseline_dataset,
        sanitised_dataset,
        hint,
        owner_col=owner_col,
        record_col=record_col,
        entry_col=entry_col,
    )
