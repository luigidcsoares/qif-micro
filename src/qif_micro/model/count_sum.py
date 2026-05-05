from collections.abc import Iterable
from functools import reduce

from multimethod import multimethod
from scipy.special import gammaln
import numpy as np
import polars as pl
import scipy.sparse as sp

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist

from qif_micro.model import baseline
from qif_micro.model._internal import _mk_long_dataset, _mk_records
from qif_micro.typing import DataFrame, Model
from qif_micro._utils import _valid_columns, _filter_optional

@multimethod
def build(
    datasets: Iterable[DataFrame],
    agg_col: str = "agg",
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id",
    entry_col: str = "entry_id",
) -> Model:
    """
    Build adversary strategies from a count-sum query as follows

    .. code-block:: sql
        SELECT count(*) as count_col, sum(agg_col) as sum_col
        FROM datasets
        GROUP_BY owner_col, group_by_col

    Constructs a joint distribution and strategy representing an adversary's
    knowledge after observing aggregated statistics (count and sum) grouped
    by owner and optional group attribute.

    Parameters
    ----------
    This function is overloaded:

    - ``build(dataset, ...)``: accepts a single :class:`DataFrame`
    - ``build([d0, d1, ...], ...)``: accepts an iterable of DataFrames

    dataset : DataFrame
        A dataset in wide format where each row is an entry of a record,
        and each column is a record attribute.
        (First overload)

    datasets : Iterable[DataFrame]
        One or more datasets with the same structure
        (all containing the same set of owners).
        (Second overload)

    agg_col : str, optional (default: "agg")
        Column name used for aggregation (sum). Must contain integers.

    count_col : str, optional (default: "count")
        Name for the count aggregation result column.

    sum_col : str, optional (default: "sum")
        Name for the sum aggregation result column.

    group_by_col : str | None, optional (default: None)
        Optional column name for the group-by operation.

    owner_col : str, optional (default: "owner_id")
        Column name for the owner identifier.

    entry_col : str, optional (default: "entry_id")
        Column name for the entry identifier within each record.

    Returns
    -------
    tuple[Joint, Strategy]
        A pair (baseline_joint, adv_st) where:
        - baseline_joint: Joint distribution over aggregated records and hints
        - adv_st: Adversary's strategy (posterior) for inferring records

    Pre-conditions
    --------------
    - Each dataset must be in wide format: one row per entry, columns are
      attributes, with owner and entry identifier columns.

    - Owner column (default "owner_id") must exist.

    - Entry column (default "entry_id") must exist.

    - The ``agg_col`` must be an integer-typed column.

    - If ``group_by_col`` is provided, it must exist in all datasets.

    - All datasets must contain the same set of owners.

    - At least one dataset must be provided.

    Examples
    -------
    >>> import polars as pl
    >>> from qif_micro import model

    Consider the following histograms, and one of the original datasets:

    >>> dataset = pl.DataFrame({
    ...     "owner_id": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ...     "entry_id": [0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 3],
    ...     "agg":      [0, 2, 1, 1, 0, 2, 0, 2, 1, 0, 1],
    ...     "group":    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ... })

    >>> baseline_joint, adv_st = model.count_sum(
    ...     dataset,
    ...     agg_col="agg",
    ...     group_by_col="group"
    ... )

    >>> baseline_joint.dist.toarray()
    array([[0.125     , 0.25      , 0.        , 0.125     ],
       [0.16666667, 0.        , 0.        , 0.08333333],
       [0.0625    , 0.0625    , 0.0625    , 0.0625    ]])

    >>> adv_st.dist.toarray()
    array([[1., 1., 0., 1.],
           [0., 0., 0., 0.],
           [0., 0., 1., 0.]])

    We can also construct a longitudinal model. Consider a second dataset
    (with the restriction that ``agg_col`` and ``group_col`` must be the same):

    >>> dataset_rhs = pl.DataFrame({
    ...     "owner_id": [0, 1, 1, 2, 3],
    ...     "entry_id": [0, 0, 1, 0, 0],
    ...     "agg":      [5, 5, 3, 0, 0],
    ...     "group":    [0, 0, 0, 0, 1]
    ... })

    >>> datasets = [dataset, dataset_rhs]
    >>> baseline_joint, adv_st = model.count_sum(
    ...     datasets,
    ...     agg_col="agg",
    ...     group_by_col="group"
    ... )

    >>> baseline_joint.dist.toarray()
    array([[0.        , 0.        , 0.125     , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.125     ],
           [0.        , 0.        , 0.        , 0.        , 0.125     ,
            0.125     , 0.        , 0.        , 0.        , 0.        ],
           [0.16666667, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.08333333, 0.        , 0.        ],
           [0.        , 0.0625    , 0.        , 0.0625    , 0.        ,
            0.        , 0.0625    , 0.        , 0.0625    , 0.        ]])

    >>> adv_st.dist.toarray()
    array([[0., 0., 1., 0., 0., 1., 0., 0., 0., 1.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 1., 0., 1., 0., 0., 1., 0., 1., 0.]])
    """
    # =============================================================
    # Pre-conditions: The dataset must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    #
    # If more than one dataset, they must contain the same owners.
    # =============================================================
    datasets = [df.lazy() for df in datasets]
    if len(datasets) == 0: raise ValueError("Empty sequence of datasets!")

    owners_expr = pl.col(owner_col).unique()
    owners = set(
         datasets[0]
         .select(owners_expr)
         .collect(engine="streaming")
         .to_series()  # ty:ignore[unresolved-attribute]
     )

    orig_cols = _filter_optional([agg_col, group_by_col])
    for dataset in datasets:
        required = [owner_col, entry_col, *orig_cols]
        ok, missing = _valid_columns(dataset, required)

        if not ok:
            msg = f"Missing the following attributes: {missing}!"
            raise ValueError(msg)

        schema = dataset.collect_schema()
        if not schema[agg_col].is_integer():
            msg = f"``agg_col`` ({agg_col}) must be integer!"
            raise ValueError(msg)

        owners_i = set(
             dataset
             .select(owners_expr)
             .collect(engine="streaming")
             .to_series() # ty:ignore[unresolved-attribute] we dont have InProcessQuery
         )

        if owners_i != owners:
            raise ValueError("All datasets must have the same owners!")

    # =============================================================
    # End pre-conditions
    # =============================================================
    
    # We begin by constructing the baseline model. We also request
    # the map from hint labels to the columns in the baseline.
    # This gives us only the labels that are possible in practice,
    # which means that we do not need to construct the whole adv model.
    joint_orig, map_owners, map_labels = baseline(
        # Dataset with ``agg_col`` as the hints   
        datasets, orig_cols, owner_col=owner_col, entry_col=entry_col,
        return_owners=True, return_labels=True,
        # We disable opt_memory, so that we keep labels around for aligning.
        opt_memory=False
    )

    # Then we need to remap the prior and joint, so that the inputs
    # are aggregated records, not the detailed records from ``orig``.
    # 
    # This can be done by summing over records that map to same agg.
    # There's no need for normalisation, as the gain fn induces eq classes.
    sum_expr = pl.col(agg_col).sum().alias(sum_col)
    count_expr = pl.len().alias(count_col)
    histogram_cols = _filter_optional([owner_col, group_by_col])
    sort_cols = _filter_optional([owner_col, group_by_col])

    agg_entries_seq = [
        df.group_by(histogram_cols).agg(count_expr, sum_expr)
        .sort(sort_cols)
        .with_columns(pl.row_index(entry_col).over(owner_col))
        .pipe(_mk_records, owner_col, entry_col)
        for df in datasets
    ]

    long_agg_dataset = (
        _mk_long_dataset(agg_entries_seq, owner_col)
        .rename({"record": "agg_record"})
    )
    
    pi_orig_dist = joint_orig.dist.sum(axis=1)
    pi_agg = ProbabDist(
        pl.LazyFrame({"p": pi_orig_dist})
        .with_row_index("record")
        .join(map_owners, on="record")
        .join(long_agg_dataset.lazy(), on=owner_col)
        .drop(owner_col)
        .unique()
        .group_by("agg_record").agg(pl.col("p").sum())
        .sort("agg_record")
        .select("p")
        .collect(engine="streaming")
        .to_numpy()  # ty:ignore[unresolved-attribute]
        .ravel()
    )

    joint_orig_dist = joint_orig.dist.tocoo()
    data = joint_orig_dist.data
    rows, cols = joint_orig_dist.coords

    joint_agg_metadata = (
        pl.LazyFrame({"record": rows, "hint": cols, "p": data})
        .join(map_owners, on="record")
        .join(long_agg_dataset.lazy(), on=owner_col)
        .drop(owner_col)
        .unique()
        .group_by("agg_record", "hint")
        .agg(pl.col("p").sum())
        .collect(engine="streaming")
    )
    
    n_rows = joint_agg_metadata.select(pl.col("agg_record").max() + 1).item()  # ty:ignore[unresolved-attribute]
    n_cols = joint_agg_metadata.select(pl.col("hint").max() + 1).item()  # ty:ignore[unresolved-attribute]

    data = joint_agg_metadata["p"].to_numpy()  # ty:ignore[not-subscriptable]
    rows = joint_agg_metadata["agg_record"].to_numpy()  # ty:ignore[not-subscriptable]
    cols = joint_agg_metadata["hint"].to_numpy()  # ty:ignore[not-subscriptable]

    shape = (n_rows, n_cols)
    coo_repr = (data, (rows, cols))
    joint_agg_dist = sp.coo_array(coo_repr, shape=shape)
    baseline_joint = Joint(joint_agg_dist.tocsr())

    def _with_count_sum(ch_metadata, i):
        agg_entries = (
            agg_entries_seq[i]
            .lazy()
            .explode("record")
            .unnest("record")
        )

        cols = _filter_optional([count_col, sum_col, group_by_col])
        agg_expr = (
            pl.coalesce("^agg_entries$", pl.lit([]))
            .list.concat(pl.struct(cols))
            .alias("agg_entries")
        )

        return (
            ch_metadata
            .join(agg_entries, on=owner_col)
            .select(owner_col, "agg_record", agg_expr)
        )


    def _compute_next_hint_p(ch_metadata):
        ch_metadata = ch_metadata.with_columns(
            pl.col("agg_entries").list.first().struct.field(count_col),
            pl.col("agg_entries").list.first().struct.field(sum_col),
            pl.col("agg_entries").list.slice(1),
            pl.col("hint_label").list.first().struct.unnest(),
            pl.col("hint_label").list.slice(1)
        )

        # We need the sum, count and hint value (which are the vals that were agg):
        n = pl.col(sum_col)
        k = pl.col(count_col)
        h = pl.col(agg_col)

        # The prob of the hint is [h == n] if the count is 1 / record_count;
        # else, we follow the formula from the paper, using ln of gamma
        # (we use log for precision).
        # 
        # Given the way we have constructed the hints, there will
        # be no cell for the case (k == 1) ^ (h != n). Similarly,
        # there will be no cell for the case (k > 1) ^ (h > n).
        rlen_expr = (
            pl.struct(*_filter_optional([count_col, group_by_col]))
            .unique()
            .struct.field(count_col)
            .sum()
            .over("agg_record")
        )

        next_p_log_expr = (
            k.log() - rlen_expr.log() + (k - 1).log()
            + gammaln(n + k - h - 1) - gammaln(n - h + 1)
            + gammaln(n + 1) - gammaln(n + k)
        )

        next_p_expr = (
            pl.when(k == 1)
            .then(1 / rlen_expr)
            .otherwise(next_p_log_expr.exp())
        )

        p_expr = (pl.coalesce("^p$", 1.0) * next_p_expr).alias("p")
        return ch_metadata.with_columns(p_expr)


    def get_agg(i):
        return pl.col("agg_entries").list.get(i).struct


    def get_hint(i):
        return pl.col("hint_label").list.get(i).struct


    def mk_pred_group(i):
        if group_by_col is None: return pl.lit(True)
        group_hint_expr = get_hint(i).field(group_by_col)
        group_agg_expr = get_agg(i).field(group_by_col)
        return group_hint_expr == group_agg_expr


    def mk_pred_agg(i):
        count_expr = get_agg(i).field(count_col)
        sum_expr = get_agg(i).field(sum_col)
        agg_expr = get_hint(i).field(agg_col)

        # If there is only one transaction, we only consider the case of a hint
        # whose agg value is exactly the sum. Else, we check for <= sum
        pred_one_entry = ((count_expr == 1) & (agg_expr == sum_expr))
        pred_entries = ((count_expr > 1) & (agg_expr <= sum_expr))

        return pred_one_entry | pred_entries


    pred_hint = [
        mk_pred_group(i) & mk_pred_agg(i)
        for i in range(len(datasets))
    ]

    ch_metadata = reduce(
        _with_count_sum,
        range(len(datasets)),
        long_agg_dataset.lazy().unique("agg_record")
    )

    map_labels = map_labels.with_columns(
        pl.concat_list("hint_label").alias("hint_label")
    )

    ch_metadata = ch_metadata.join_where(map_labels, *pred_hint)    

    ch_metadata = reduce(
        lambda acc, _: _compute_next_hint_p(acc),
        range(len(datasets)),
        ch_metadata
    )

    ch_metadata = (
        ch_metadata
        .select("agg_record", "hint", "p")
        .collect(engine="streaming")
    )

    n_rows = ch_metadata.select(pl.col("agg_record").max() + 1).item()  # ty:ignore[unresolved-attribute]
    n_cols = ch_metadata.select(pl.col("hint").max() + 1).item()  # ty:ignore[unresolved-attribute]

    data = ch_metadata["p"].to_numpy()  # ty:ignore[not-subscriptable]
    rows = ch_metadata["agg_record"].to_numpy()  # ty:ignore[not-subscriptable]
    cols = ch_metadata["hint"].to_numpy()  # ty:ignore[not-subscriptable]

    coo_repr = (data, (rows, cols))
    hint_ch_dist = sp.coo_array(coo_repr, shape=(n_rows, n_cols))
    hint_ch = Channel(hint_ch_dist.tocsr())

    adv_joint = qif.joint(pi_agg, hint_ch)
    adv_st = qif.strategy(adv_joint)

    return baseline_joint, adv_st


@multimethod
def build(  # noqa: F811
    dataset: DataFrame,
    agg_col: str = "agg",
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id",
    entry_col: str = "entry_id",
) -> Model:
    return build(
        [dataset],
        agg_col, count_col, sum_col, group_by_col,
        owner_col, entry_col
    )
