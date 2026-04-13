from collections.abc import Iterable

import polars as pl
import scipy.sparse as sp

from qif_micro import qif
from qif_micro._utils import _valid_columns
from qif_micro.mechanism import Mechanism
from qif_micro.model._internal import _mk_records
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist
from qif_micro.typing import DataFrame, is_dataframe, Model, Record


def build(
    pi: ProbabDist,
    records: Iterable[Record] | DataFrame,
    mechanism: Mechanism,
    baseline_dataset: DataFrame,
    sanitised_dataset: DataFrame,
    hint: str | Iterable[str],
    owner_col: str = "owner_id",
    record_col: str = "record_id",
    entry_col: str = "entry_id",
) -> Model:
    """
    Build a generic adversary model from records and datasets.

    Constructs a joint distribution and posterior strategy for an adversary
    who observes a privacy-preserving mechanism applied to a dataset. This
    is the most flexible model, supporting arbitrary mechanisms.

    Parameters
    ----------
    pi : ProbabDist
        Prior distribution over the record domain. Length must equal the
        number of distinct records.

    records : Iterable[Record] | DataFrame
        The domain of possible records. Either:
        - An iterable of records (each a dict with attribute names as keys)
        - A DataFrame where each row is an entry, with ``record_col`` and
          ``entry_col`` columns identifying records and entries

    mechanism : Mechanism
        A callable that takes (input_domain, output_domain) and returns a
        Channel representing the privacy mechanism's output distribution.

    baseline_dataset : DataFrame
        Original dataset in wide format: one row per entry, columns are
        attributes, with owner and entry identifier columns. Represents
        the "true" data before privacy mechanism application.

    sanitised_dataset : DataFrame
        Output of applying the privacy mechanism to the baseline dataset.
        Must have the same structure as baseline_dataset.

    hint : str | Iterable[str]
        Column name(s) representing the adversary's auxiliary information
        (observations). Typically corresponds to non-sensitive columns or
        post-processed outputs.

    owner_col : str, optional (default: "owner_id")
        Column name for the owner identifier (in datasets).

    record_col : str, optional (default: "record_id")
        Column name for the record identifier (in record domain).

    entry_col : str, optional (default: "entry_id")
        Column name for the entry identifier within records.

    Returns
    -------
    tuple[Joint, Strategy]
        A pair (baseline_joint, adv_st) where:
        - baseline_joint: Joint distribution over records and hints
        - adv_st: Adversary's strategy (posterior) for inferring records

    Pre-conditions
    --------------
    - ``records`` must be in wide format: one row per entry, columns are
      record attributes, with ``record_col`` and ``entry_col`` columns.

    - ``records`` must have ``record_col`` and ``entry_col`` columns.

    - ``baseline_dataset`` and ``sanitised_dataset`` must have ``owner_col``
      and ``entry_col`` columns.

    - ``pi`` size must equal the number of unique records in the record domain.

    - All ``hint`` columns must exist in both datasets.

    - The mechanism must be compatible with record and dataset structures.

    - ``baseline_dataset`` and ``sanitised_dataset`` must produce records
      compatible with the record domain under the mechanism.

    - Both datasets must not have incompatibilities with the mechanism.

    Examples
    --------
    >>> from functools import partial
    >>> import numpy as np
    >>> import polars as pl
    >>> from qif_micro import qif
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
     
    >>> eps_q = np.log(2) # p = 2/3
    >>> eps_s = np.log(3) # p = 3/4
    >>> rr_q = partial(qif.dp.random_response, eps=eps_q, domain_size=3)
    >>> rr_s = partial(qif.dp.random_response, eps=eps_s, domain_size=2)
    >>> m = partial(mechanism.record, q=rr_q, s=rr_s)

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
    
    >>> result = model.generic(pi, records, m, b_dataset, s_dataset, "q")
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
    array([[1. , 0. , 0. ],
           [0. , 0. , 0. ],
           [0. , 0. , 0. ],
           [0. , 1. , 0. ],
           [0. , 0. , 0.5],
           [0. , 0. , 0.5]])

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

    >>> result = model.generic(pi, records, m, b_dataset, s_dataset, "q")
    >>> baseline_joint, adv_st = result
           
    >>> baseline_joint.dist.toarray()
    array([[0.5 , 0.  , 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25]])

    >>> adv_st.dist.toarray()
    array([[1. , 0. , 0. ],
           [0. , 0. , 0. ],
           [0. , 0. , 0. ],
           [0. , 1. , 0. ],
           [0. , 0. , 0.5],
           [0. , 0. , 0.5]])
    """
    # ========================================================================
    # Pre-processing inputs
    # ========================================================================
    def as_df(i, r): return pl.LazyFrame(r).with_columns(
        pl.lit(i).alias(record_col),
        pl.row_index(entry_col)
    )

    if not is_dataframe(records):
        records = (as_df(i, r) for i, r in enumerate(records))
        records = pl.concat(records, how="diagonal")

    records = records.lazy()
    baseline_dataset = baseline_dataset.lazy()
    sanitised_dataset = sanitised_dataset.lazy()

    # Standardise ``hint`` input
    hint = [hint] if isinstance(hint, str) else list(hint)

    # ========================================================================
    # Pre-conditions
    # ========================================================================
    # Check the required attributes for the domain of records
    required = [record_col, entry_col, *hint]
    ok, missing = _valid_columns(records, required)
    if not ok: raise ValueError(f"Records missing attributes: {missing}")

    # We assume that each index i in the prior ``pi`` corresponds to the i-th
    # record in the domain of records. So, their length must be the same.
    n_records = (
        records
        .select(pl.col(record_col).n_unique())
        .collect(engine="streaming")
        .item()  # ty:ignore[unresolved-attribute]
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
        .collect(engine="streaming").lazy() # Cache  # ty:ignore[unresolved-attribute]
    )

    baseline_records = (
        baseline_dataset
        .sort(owner_col, entry_col)
        .select(owner_col, entry_expr, *attr_cols)
        .pipe(_mk_records, owner_col).lazy()
        .join(records, on="record", how="left")
        .collect(engine="streaming").lazy() # Cache  # ty:ignore[unresolved-attribute]
    )
    
    sanitised_records = (
        sanitised_dataset
        .sort(owner_col, entry_col)
        .select(owner_col, entry_expr, *attr_cols)
        .pipe(_mk_records, owner_col).lazy()
        .join(records, on="record", how="left")
        .collect(engine="streaming").lazy() # Cache  # ty:ignore[unresolved-attribute]
    )

    # The input ``mechanism`` is a function that takes input and output
    # domains of records, and returns the concrete mechanism (channel):
    def to_long_format(df): return df.explode("record").unnest("record")
    input_domain = to_long_format(records)
    output_domain = to_long_format(sanitised_records)
    
    mechanism_ch = mechanism(
        input_domain=input_domain,
        output_domain=output_domain.drop(owner_col).unique()
    )

    # Compute the posteriors and transpose, since the result is a channel
    # with outputs as the rows and we need them as columns.
    hyper_mechanism = qif.hyper(pi, mechanism_ch).posteriors.dist.tocoo()
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
        .item()  # ty:ignore[unresolved-attribute]
    )

    if has_nulls:
        raise ValueError("Incompatible baseline and sanitised datasets!")

    # ========================================================================
    # The adversary's intermediate knowledge on a particular record is given
    # by the sum of the posterior probability of that record, given
    # the observed sanitised records, weighted by the dataset length.
    n_records = (
        sanitised_records
        .select(pl.len())
        .collect(engine="streaming")
        .item()  # ty:ignore[unresolved-attribute]
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
        delta_metadata.lazy()  # ty:ignore[unresolved-attribute]
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
        ch_metadata.lazy()  # ty:ignore[unresolved-attribute]
        .join(baseline_records, on="row")
        .with_columns(p_expr)
        .group_by("row", "hint")
        .agg(pl.col("p").sum(), pl.col("dense_row").first())
        .collect(engine="streaming")
    )

    n_rows = delta_metadata.height  # ty:ignore[unresolved-attribute]
    n_cols = baseline_joint_metadata.select(pl.col("hint").max() + 1).item()  # ty:ignore[unresolved-attribute]

    data = baseline_joint_metadata["p"].to_numpy()  # ty:ignore[not-subscriptable]
    cols = baseline_joint_metadata["hint"].to_numpy()  # ty:ignore[not-subscriptable]
    rows = baseline_joint_metadata["dense_row"].to_numpy()  # ty:ignore[not-subscriptable]

    coo_repr = (data, (rows, cols))
    baseline_joint_dist = sp.coo_array(coo_repr, shape=(n_rows, n_cols))
    baseline_joint = Joint(baseline_joint_dist.tocsr())

    data = ch_metadata["p"].to_numpy()  # ty:ignore[not-subscriptable]
    cols = ch_metadata["hint"].to_numpy()  # ty:ignore[not-subscriptable]
    rows = ch_metadata["dense_row"].to_numpy()  # ty:ignore[not-subscriptable]

    shape = (n_rows, n_cols)
    coo_repr = (data, (rows, cols))
    dist = sp.coo_array(coo_repr, shape=shape).tocsr()

    hint_ch = Channel(dist)

    delta = ProbabDist(delta_metadata["p"].to_numpy().ravel())  # ty:ignore[not-subscriptable]
    adv_joint = qif.joint(delta, hint_ch)
    adv_st = qif.strategy(adv_joint)

    return baseline_joint, adv_st
