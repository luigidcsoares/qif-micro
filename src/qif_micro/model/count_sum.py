from collections.abc import Sequence
from functools import reduce

import numpy as np
import polars as pl

from multimethod import multimethod
from scipy.sparse import coo_array
from scipy.special import gammaln

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, ProbabDist, Strategy

from qif_micro.model import baseline
from qif_micro.model.datatypes import Dataset, MapLabels, MapOwners
from qif_micro.model._internal import _mk_long_dataset, _mk_records
from qif_micro._utils import _valid_columns, _filter_optional


def _get_owners(dataset : Dataset, owner_col: str = "owner_id"):
    owners_expr = pl.col(owner_col).unique()
    return set(dataset.select(owners_expr).collect().to_series())


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


def _validate_dataset(
    dataset: Dataset,
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id"
):
    # ``dataset```: The dataset must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    #
    # If ``group_by_col`` is not defined, records must have len one.
    #
    # The type of ``count_col`` and ``sum_col`` must be an integer.
    prefix_msg = "Dataset :: "

    schema = dataset.collect_schema()
    dataset_cols = _filter_optional([count_col, sum_col, group_by_col])

    required_cols = [owner_col, *dataset_cols]
    missing_cols = set(required_cols) - set(schema.keys())

    if len(missing_cols) > 0:
        msg = f"Missing the following attrs: {missing_cols}!"
        raise ValueError(prefix_msg + msg)

    rlens = dataset.group_by(owner_col).agg(pl.len())
    max_rlen = rlens.select(pl.col("len").max()).collect().item()

    if (group_by_col is None) and (max_rlen > 1):
        msg = "Record length must be 1, unless ``group_by_col`` is set`!"
        raise ValueError(prefix_msg + msg)

    if not schema[count_col].is_integer():
        msg = f"``count_col`` ({count_col}) must be integer!"
        raise ValueError(prefix_msg + msg)

    if not schema[sum_col].is_integer():
        msg = f"``sum_col`` ({sum_col}) must be integer!"
        raise ValueError(prefix_msg + msg)


def _validate_orig(
   dataset: Dataset,
   agg_col: str,
   group_by_col: str | None = None,
   owner_col: str = "owner_id"
):
    # ``dataset``: Must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    #
    # ``agg_col`` must be an integer
    prefix_msg = "Original :: "

    schema = dataset.collect_schema()
    orig_cols = _filter_optional([agg_col, group_by_col])

    required_cols = [owner_col, *orig_cols]
    missing_cols = set(required_cols) - set(schema.keys())

    if len(missing_cols) > 0:
        msg = f"Missing the following attrs: {missing_cols}!"
        raise ValueError(prefix_msg + msg)

    if not schema[agg_col].is_integer():
        msg = f"``agg_col`` ({agg_col}) must be integer!"
        raise ValueError(prefix_msg + msg)


type ReturnModel = (
    tuple[ProbabDist, Channel, Strategy]
    | tuple[ProbabDist, Channel, Strategy, MapOwners | MapLabels]
    | tuple[ProbabDist, Channel, Strategy, MapOwners, MapLabels]
)

@multimethod
def build(
    datasets: Sequence[Dataset],
    origs: Sequence[Dataset],
    agg_col: str,
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id",
    return_owners: bool = False,
    return_labels: bool = False
) -> ReturnModel:
    """
    TODO

    Examples
    -------
    >>> import polars as pl
    >>> from qif_micro import model

    Consider the following histograms, and one of the original datasets:

    >>> orig = pl.DataFrame({
    ...     "owner_id": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ...     "agg":      [0, 2, 1, 1, 0, 2, 0, 2, 1, 0, 1],
    ...     "group":    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ... })

    >>> dataset = pl.DataFrame({
    ...     "owner_id": [0, 1, 2, 3, 3],
    ...     "count":    [2, 2, 3, 3, 1],
    ...     "sum":      [2, 2, 2, 2, 1],
    ...     "group":    [0, 0, 0, 0, 1]
    ... })

    >>> pi, baseline_ch, adv_st = model.count_sum(
    ...     dataset, orig, "agg", group_by_col="group"
    ... )
    >>> pi.dist

    >>> baseline_ch.dist.toarray()

    >>> adv_st
    """
    # =============================================================
    # Pre-conditions
    # =============================================================
    datasets = [d.lazy() for d in datasets]
    origs = [d.lazy() for d in origs]

    # Confirm that there's an orig dataset for each dataset the adv observed:
    if len(datasets) != len(origs):
        raise ValueError("Mismatch between ``datasets`` and ``origs`` lenghts")

    for d in datasets:
        _validate_dataset(d, count_col, sum_col, group_by_col, owner_col)

    for d in origs:
         _validate_orig(d, agg_col, group_by_col, owner_col)

    # Owners must be the same across all datasets:
    base_owners = _get_owners(datasets[0])
    owners = reduce(
        lambda acc, d: _get_owners(d, owner_col) & acc,
        datasets[1:] + origs,
        base_owners
    )

    if owners != base_owners:
        raise ValueError("``datasets + ``origs`` must have the same owners!")

    # And aggregated records in ``datasets`` and ``origs`` must be the same:
    sort_cols = _filter_optional([owner_col, group_by_col, count_col, sum_col])
    agg_entries_seq = list(
        d.sort(sort_cols).pipe(_mk_records, owner_col)
        for d in datasets
    )

    long_agg_dataset = (
        _mk_long_dataset((agg_entries_seq), owner_col)
        .rename({"record": "agg_record"})
    )

    long_agg_orig = _mk_long_dataset((
        _mk_agg_entries(d, agg_col, count_col, sum_col, group_by_col, owner_col)
        .sort(sort_cols)
        .pipe(_mk_records, owner_col)
        for d in origs
    ), owner_col).rename({"record": "agg_record"})

    is_eq = (
        long_agg_dataset
        .join(long_agg_orig, on=owner_col)
        .select((pl.col("agg_record") == pl.col("agg_record_right")).all())
        .collect()
        .item()
    )

    if not is_eq: raise ValueError("``datasets`` incompatible with ``origs``!")

    # =============================================================
    # End pre-conditions
    # =============================================================

    # We begin by constructing the baseline model. We also request
    # the map from hint labels to the columns in the baseline.
    # This gives us only the labels that are possible in practice,
    # which means that we do not need to construct the whole adv model.
    pi_orig, ch_orig, map_owners, map_labels = baseline(
        # Dataset with ``agg_col`` as the hints   
        origs, list(_filter_optional([agg_col, group_by_col])),
        owner_col=owner_col, return_owners=True, return_labels=True,
        # We disable opt_memory, so that we keep labels around for aligning.
        opt_memory=False
    )

    # Then we need to remap the prior and channel, so that the inputs
    # are aggregated records, not the detailed records from ``orig``.
    # 
    # This can be done by summing over records that map to same agg.
    # There's no need for normalisation, as the gain fn induces eq classes.
    pi_agg = ProbabDist(
        map_owners
        .sort("record")
        .with_columns(pl.lit(pi_orig.dist).alias("p"))
        .join(long_agg_orig, on=owner_col)
        .group_by("agg_record").agg(pl.col("p").sum())
        .sort("agg_record")
        .select("p")
        .collect()
        .to_numpy()
        .ravel()
    )

    # For the channel, we first construct the joint,
    # then remap to aggregated records and then back to channel.
    joint_dist_orig = qif.joint(pi_orig, ch_orig).dist.tocoo()
    data, rows, cols = joint_dist_orig.data, *joint_dist_orig.coords

    joint_agg_metadata = (
        pl.LazyFrame({ "record": rows, "hint": cols, "p": data })
        .join(map_owners, on="record")
        .join(long_agg_orig, on=owner_col)
        .group_by("agg_record", "hint")
        .agg(pl.col("p").sum())
        .collect()
    )

    n_rows = joint_agg_metadata["agg_record"].max() + 1
    n_cols = joint_agg_metadata["hint"].max() + 1

    data = joint_agg_metadata["p"].to_numpy()
    rows = joint_agg_metadata["agg_record"].to_numpy()
    cols = joint_agg_metadata["hint"].to_numpy()

    joint_dist = coo_array((data, (rows, cols)), shape=(n_rows, n_cols))
    baseline_ch_dist = joint_dist / pi_agg.dist[:, np.newaxis]
    baseline_ch = Channel(baseline_ch_dist.tocsr())

    # Now that we have the baseline channel, we can construct
    # the adversary's strategy but only for a subset of valid hints.
    #
    # We first collect the valid columns (non-zero cells)
    # for each row (aggregated record) in the baseline.
    indices = baseline_ch.dist.indices
    sections = baseline_ch.dist.indptr[1:-1]
    valid_cols = np.split(indices, sections)

    # Then we construct the metadata for the hint channnel:
    # for each aggregated record, we need the hint labels.
    # We standardise the hints as a list (in case this is not longitudinal).
    labels_schema = map_labels.collect_schema()
    as_list = lambda c: c if labels_schema[c] == pl.List else pl.concat_list(c)

    agg_entries = (
        pl.concat(agg_entries_seq)
        .group_by(owner_col)
        .agg(pl.col("record").alias("agg_entries"))
    )

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
        pred_owner = pl.col(owner_col) == pl.col(owner_col + "_right")
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
            .join_where(agg_entries, pred_owner, pred_group)
            .select("agg_record", "hint", "hint_label", agg_expr)
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
        rlen_expr = k.unique().sum().over("agg_record")

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
    remaining_p_expr = (1 - pl.col("p").sum()).alias("p")
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
        return pi_agg, baseline_ch, adv_st, map_owners, map_labels

    if return_owners: return pi_agg, baseline_ch, adv_st, map_owners
    if return_labels: return pi_agg, baseline_ch, adv_st, map_labels

    return pi_agg, baseline_ch, adv_st


@multimethod
def build(
    dataset: Dataset,
    orig: Dataset,
    agg_col: str,
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id",
    return_owners: bool = False,
    return_labels: bool = False
) -> ReturnModel:
    return build(
        [dataset], [orig],
        agg_col, count_col, sum_col, group_by_col, owner_col,
        return_owners, return_labels
    )
