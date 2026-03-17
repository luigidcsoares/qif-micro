from collections.abc import Sequence
from functools import reduce

import numpy as np
import polars as pl

from multimethod import multimethod
from scipy.sparse import coo_array
from scipy.special import gammaln

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist, Strategy

from qif_micro.model import baseline
from qif_micro.model.typing import Dataset, Model
from qif_micro.model._internal import _mk_long_dataset, _mk_records
from qif_micro._utils import _valid_columns, _filter_optional

def _mk_agg_entries(
    dataset: Dataset,
    agg_col: str,
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id"
) -> Dataset:
    sum_expr = pl.col(agg_col).sum().alias(sum_col)
    count_expr = pl.len().alias(count_col)
    histogram_cols = _filter_optional([owner_col, group_by_col])
    return  dataset.group_by(histogram_cols).agg(count_expr, sum_expr)


@multimethod
def build(
    datasets: Sequence[Dataset],
    agg_col: str = "agg",
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id",
    return_owners: bool = False,
    return_labels: bool = False
) -> Model:
    """
    Build the adversary's strategies given the result of a query of the form

    .. code-block:: sql
        SELECT count(*) as count_col, sum(agg_col) as sum_col
        FROM datasets
        GROUP_BY owner_col, group_by_col

    We only return the adversary's strategies with respect to the outputs that
    are possible in practice, considering the original datasets.

    This function assumes that the adversary's prior knowledge on records
    (and consequently on datasets) is uniform.

    Parameters
    ----------
    This function is overloaded:

    - ``build(dataset, ...)``: accepts a single :class:`Dataset`
    - ``build([d0, d1, ...], ...)``: accepts a sequence of :class:`Dataset`

    dataset : Dataset
        A dataset containing owners, hints and sensitive attributes.
        (First overload)

    datasets : Sequence[Dataset]
        A dataset containing owners, hints and sensitive attributes.
        (Second overload)

    agg_col : str, optional (Default: ``agg``)
        Column name used in the sum aggregation (must be an integer).
        
    count_col : str, optional (default: ``"count"``)
        Column name used as alias for the result of the count aggregation.

    sum_col : str, optional (default: ``"sum"``)
        Column name used as alias for the result of the sum aggregation.

    group_by_col : str | None, optional (default: None)
        Column name used in the group-by operation.
        
    owner_col : str, optional (default: ``"owner_id"``)
        Column name for the owner identifier.

    return_owners : bool, optional (default: ``False``)
        If true, the result includes a map from owners to row_indices.

    return_labels : bool, optional (default: ``False``)
        If true, the result includes a map from hint labels to column indices.

    Returns
    -------
    Joint
        The baseline joint knowledge.

    Strategy
        The adversary's strategies for each valid output (given the baseline).

    tuple (Joint, Strategy, MapOwners | MapLabels)
        - The baseline joint knowledge;
        - The adversary's strategies for each valid output (given the baseline);
        - If ``map_owners`` enabled: map from owners to row indices OR
          If ``map_labels`` enabled: map from hint labels to indices.

    tuple (Joint, MapOwners, MapLabels)
        - The baseline joint knowledge;
        - The adversary's strategies for each valid output (given the baseline);
        - Map from owners to row indices;
        - Map from hint labels to indices.
    
    Examples
    -------
    >>> import polars as pl
    >>> from qif_micro import model

    Consider the following histograms, and one of the original datasets:

    >>> dataset = pl.DataFrame({
    ...     "owner_id": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
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
    array([[1., 0., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.]])

    We can also construct a longitudinal model. Consider a second dataset
    (with the restriction that ``agg_col`` and ``group_col`` must be the same):

    >>> dataset_rhs = pl.DataFrame({
    ...     "owner_id": [0, 1, 1, 2, 3],
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
    array([[0.16666667, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.08333333, 0.        , 0.        ],
           [0.        , 0.        , 0.125     , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.125     ],
           [0.        , 0.        , 0.        , 0.        , 0.125     ,
            0.125     , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.0625    , 0.        , 0.0625    , 0.        ,
            0.        , 0.0625    , 0.        , 0.0625    , 0.        ]])

    >>> adv_st.dist.toarray()
    array([[1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 1., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 1., 0., 0.]])

    We can get the map from owners to record ids (rows):

    >>> m = model.count_sum(
    ...    datasets,
    ...    agg_col="agg",
    ...    group_by_col="group",
    ...    return_owners=True
    ... )[2]
    >>> m.sort("owner_id").collect()
    shape: (4, 2)
    ┌──────────┬────────┐
    │ owner_id ┆ record │
    │ ---      ┆ ---    │
    │ i64      ┆ u32    │
    ╞══════════╪════════╡
    │ 0        ┆ 2      │
    │ 1        ┆ 3      │
    │ 2        ┆ 0      │
    │ 3        ┆ 1      │
    └──────────┴────────┘

    And the map from hint labels to the corresponding cols in the channel:
    
    >>> m = model.count_sum(
    ...    datasets,
    ...    agg_col="agg",
    ...    group_by_col="group",
    ...    return_labels=True
    ... )[2]
    >>> m.sort("hint_label").collect()
    shape: (10, 2)
    ┌─────────────────┬──────┐
    │ hint_label      ┆ hint │
    │ ---             ┆ ---  │
    │ list[struct[2]] ┆ u32  │
    ╞═════════════════╪══════╡
    │ [{0,0}, {0,0}]  ┆ 0    │
    │ [{0,0}, {0,1}]  ┆ 1    │
    │ [{0,0}, {5,0}]  ┆ 2    │
    │ [{1,0}, {0,1}]  ┆ 3    │
    │ [{1,0}, {3,0}]  ┆ 4    │
    │ [{1,0}, {5,0}]  ┆ 5    │
    │ [{1,1}, {0,1}]  ┆ 6    │
    │ [{2,0}, {0,0}]  ┆ 7    │
    │ [{2,0}, {0,1}]  ┆ 8    │
    │ [{2,0}, {5,0}]  ┆ 9    │
    └─────────────────┴──────┘
    """
    # =============================================================
    # Pre-conditions: The dataset must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    #
    # If more than one dataset, they must contain the same owners.
    # =============================================================
    if len(datasets) == 0: raise ValueError("Empty sequence of datasets!")

    datasets = [d.lazy() for d in datasets]

    owners_expr = pl.col(owner_col).unique()
    owners = set(datasets[0].select(owners_expr).collect().to_series())

    for i, dataset in enumerate(datasets):
        orig_cols = _filter_optional([agg_col, group_by_col])
        ok, missing = _valid_columns(dataset, [owner_col, *orig_cols])

        if not ok:
            msg = f"Missing the following attributes: {missing}!"
            raise ValueError(msg)

        schema = dataset.collect_schema()
        if not schema[agg_col].is_integer():
            msg = f"``agg_col`` ({agg_col}) must be integer!"
            raise ValueError(msg)

        owners_i = set(dataset.select(owners_expr).collect().to_series())
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
        datasets, list(_filter_optional([agg_col, group_by_col])),
        owner_col=owner_col, return_owners=True, return_labels=True,
        # We disable opt_memory, so that we keep labels around for aligning.
        opt_memory=False
    )

    # Then we need to remap the prior and joint, so that the inputs
    # are aggregated records, not the detailed records from ``orig``.
    # 
    # This can be done by summing over records that map to same agg.
    # There's no need for normalisation, as the gain fn induces eq classes.
    sort_cols = _filter_optional([owner_col, group_by_col, count_col, sum_col])
    agg_entries_seq = [
        _mk_agg_entries(d, agg_col, count_col, sum_col, group_by_col, owner_col)
        .sort(sort_cols)
        .pipe(_mk_records, owner_col)
        for d in datasets
    ]

    long_agg_dataset = (
        _mk_long_dataset(agg_entries_seq, owner_col)
        .rename({"record": "agg_record"})
    )
    
    pi_orig_dist = joint_orig.dist.sum(axis=1)
    pi_agg = ProbabDist(
        pl.LazyFrame({"p": pi_orig_dist}).with_row_index("record")
        .join(map_owners, on="record")
        .join(long_agg_dataset, on=owner_col)
        .drop(owner_col)
        .unique()
        .group_by("agg_record").agg(pl.col("p").sum())
        .sort("agg_record")
        .select("p")
        .collect()
        .to_numpy()
        .ravel()
    )

    joint_orig_dist = joint_orig.dist.tocoo()
    data = joint_orig_dist.data
    rows, cols = joint_orig_dist.coords

    joint_agg_metadata = (
        pl.LazyFrame({ "record": rows, "hint": cols, "p": data })
        .join(map_owners, on="record")
        .join(long_agg_dataset, on=owner_col)
        .drop(owner_col)
        .unique()
        .group_by("agg_record", "hint")
        .agg(pl.col("p").sum())
        .collect()
    )

    n_rows = joint_agg_metadata["agg_record"].max() + 1
    n_cols = joint_agg_metadata["hint"].max() + 1

    data = joint_agg_metadata["p"].to_numpy()
    rows = joint_agg_metadata["agg_record"].to_numpy()
    cols = joint_agg_metadata["hint"].to_numpy()

    joint_agg_dist = coo_array((data, (rows, cols)), shape=(n_rows, n_cols))
    baseline_joint = Joint(joint_agg_dist.tocsr())

    # Now that we have the baseline joint, we can construct
    # the adversary's strategy but only for a subset of valid hints.
    #
    # We first collect the valid columns (non-zero cells)
    # for each row (aggregated record) in the baseline.
    indices = baseline_joint.dist.indices
    sections = baseline_joint.dist.indptr[1:-1]
    valid_cols = np.split(indices, sections)

    # Then we construct the metadata for the hint channnel:
    # for each aggregated record, we need the hint labels.
    # We standardise the hints as a list (in case this is not longitudinal).
    labels_schema = map_labels.collect_schema()
    as_list = lambda c: c if labels_schema[c] == pl.List else pl.concat_list(c)

    ch_metadata = (
        pl.LazyFrame({ "agg_record": range(n_rows), "hint": valid_cols })
        .explode("hint")
        .join(map_labels, on="hint")
        .with_columns(as_list("hint_label"))
        # We need ``count_col`` and ``sum_col``, but not per owner,
        # only per aggregated record, so we filter owners with same record.
        # Therefore, we drop owners with same record and keep on repr
        .join(long_agg_dataset.unique("agg_record"), on="agg_record")
    )

    def _with_count_sum(ch_metadata, i):
        agg_entries = agg_entries_seq[i].explode("record").unnest("record")

        pred_group = pl.lit(True) if group_by_col is None else (
            pl.col("hint_label").list.get(i).struct.field(group_by_col)
            == pl.col(group_by_col)
        )

        agg_expr = (
            pl.coalesce("^agg_entries$", pl.lit([]))
            .list.concat(pl.struct(count_col, sum_col))
            .alias("agg_entries")
        )

        return (
            ch_metadata
            .join(agg_entries, on=owner_col)
            .filter(pred_group)
            .select(owner_col, "agg_record", "hint", "hint_label", agg_expr)
        )


    def _compute_next_hint_p(ch_metadata):
        ch_metadata = ch_metadata.with_columns(
            pl.col("agg_entries").list.first().struct.unnest(),
            pl.col("agg_entries").list.slice(1),
            pl.col("hint_label").list.first().struct.unnest(),
            pl.col("hint_label").list.slice(1)
        )

        # We need the sum, count and hint value (which are the vals that were agg):
        n = pl.col(sum_col)
        k = pl.col(count_col)
        h = pl.col(agg_col)

        # The prob of the hint is [h == n] if the count is 1 / record_count;
        # else, we follow the formula above using ln of gamma
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


    ch_metadata = reduce(_with_count_sum, range(len(datasets)), ch_metadata)
    ch_metadata = reduce(
        lambda acc, _: _compute_next_hint_p(acc),
        range(len(datasets)),
        ch_metadata
    )

    ch_metadata = ch_metadata.select("agg_record", "hint", "p").collect()

    # We only have a slice of the actual channel (from the adv prespective),
    # we temporarily add a fake column, just so we can get a proper channel
    # (to rely on the pyqif lib stuff)
    row_sum = pl.col("p").sum()
    remaining_p_expr = (
        # Avoid negative entries due to float errors
        pl.when(row_sum.is_close(1)).then(0)
        .otherwise(1 - row_sum)
        .alias("p")
    )

    remaining_p = ch_metadata.group_by("agg_record").agg(remaining_p_expr)

    n_rows = ch_metadata["agg_record"].max() + 1
    n_cols = ch_metadata["hint"].max() + 2

    data = np.hstack([
        ch_metadata["p"].to_numpy(),
        remaining_p["p"].to_numpy()
    ])

    rows = np.hstack([
        ch_metadata["agg_record"].to_numpy(),
        remaining_p["agg_record"].to_numpy()
    ])
    
    cols = np.hstack([
        ch_metadata["hint"].to_numpy(),
        np.repeat(n_cols - 1, n_rows)
    ])

    hint_ch_dist = coo_array((data, (rows, cols)), shape=(n_rows, n_cols))
    hint_ch = Channel(hint_ch_dist.tocsr())

    hint_joint = qif.joint(pi_agg, hint_ch)
    adv_st = Strategy(qif.strategy(hint_joint).dist[:-1, :])

    # Both map_owners and map_labels should be the same as for the baseline,
    # and the prior is also the same.
    if return_owners and return_labels:
        return baseline_joint, adv_st, map_owners, map_labels

    if return_owners: return baseline_joint, adv_st, map_owners
    if return_labels: return baseline_joint, adv_st, map_labels

    return baseline_joint, adv_st


@multimethod
def build(
    dataset: Dataset,
    agg_col: str = "agg",
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id",
    return_owners: bool = False,
    return_labels: bool = False
) -> Model:
    return build(
        [dataset],
        agg_col, count_col, sum_col, group_by_col, owner_col,
        return_owners, return_labels
    )
