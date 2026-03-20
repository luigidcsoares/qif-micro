from collections.abc import Iterable, Sequence
from functools import reduce

import numpy as np
import polars as pl

from multimethod import multimethod
from scipy.sparse import coo_array

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist

from qif_micro.model._internal import _mk_long_dataset, _mk_records
from qif_micro.typing import BaselineModel, Dataset
from qif_micro._utils import _valid_columns

def _mk_long_prior(long_dataset : Dataset) -> ProbabDist:
    p_expr = (pl.len() / long_dataset.height).alias("p")

    prior_dist = (
        long_dataset
        .group_by("record")
        .agg(p_expr)
        .sort("record")
        .select("p")
        .to_numpy()
        .ravel()
    )

    return ProbabDist(prior_dist)


@multimethod
def build(
    datasets: Iterable[Dataset],
    hint: str | Iterable[str],
    owner_col: str = "owner_id",
    n_partitions: int = 1,
    opt_memory: bool = True,
    return_owners: bool = False,
    return_labels: bool = False
) -> BaselineModel:
    """
    Build the adversary’s knowledge model from a dataset and auxiliary info.

    Parameters
    ----------
    This function is overloaded:

    - ``build(dataset, ...)``: accepts a single :class:`Dataset`
    - ``build([d0, d1, ...], ...): accepts a sequence of :class:`Dataset`
    
    dataset : Dataset
        A dataset containing owners, hints and sensitive attributes.
        (First overload)

    datasets : Iterable[Dataset]
        One or more datasets containing owners, hints and sensitive attributes.
        (Second overload)

    hint : str | iterable of str
        Column names that represent the adversary’s auxiliary information.

    owner_col : str, optional (default: ``"owner_id"``)
        Column name for the owner identifier.

    n_partitions : int, optional (default: ``1``)
        Controls the number of partitions used to split the channel column-wise.
        (Makes more sense for sparse channels, when memory is a concern.)

    opt_memory : bool, optional (default: ``True``)
        See the doc of ``qif.compose.parallel``

    return_owners : bool, optional (default: ``False``)
        If true, the result includes a map from owners to row_indices.

    return_labels : bool, optional (default: ``False``)
        If true, the result includes a map from hint labels to column indices.

    Returns
    -------
    Joint
        The adversary’s revised joint knowledge after observing the dataset.

    tuple (Joint, MapOwners | MapLabels)
        - The adversary’s revised joint knowledge after observing the dataset;
        - If ``map_owners`` enabled: map from owners to row indices OR
          If ``map_labels`` enabled: map from hint labels to indices.

    tuple (Joint, MapOwners, MapLabels)
        - The adversary’s revised joint knowledge after observing the dataset;
        - Map from owners to row indices;
        - Map from hint labels to indices.

    Examples
    --------
    >>> import polars as pl
    >>> from scipy.sparse import hstack
    >>> from qif_micro import model

    Consider the following dataset:

    >>> dataset = pl.DataFrame({
    ...     "owner_id":  [0, 1, 2, 2, 3, 3],
    ...     "hint":      [0, 0, 0, 1, 0, 1],
    ...     "sensitive": [0, 0, 0, 0, 1, 1]
    ... })

    The adversary's joint knowledge upon observing this dataset is:

    >>> joint = model.baseline(dataset, "hint")
    >>> joint.dist.toarray()
    array([[0.5  , 0.   ],
           [0.125, 0.125],
           [0.125, 0.125]])

    We can also construct a longitudinal model. Consider a second dataset:

    >>> dataset_rhs = pl.DataFrame({
    ...     "owner_id":  [0, 1, 2, 3],
    ...     "hint":      [0, 1, 0, 0],
    ...     "sensitive": [0, 0, 0, 1]
    ... })

    >>> datasets = [dataset, dataset_rhs]
    >>> joint = model.baseline(datasets, "hint")
    >>> hstack(joint.dist).toarray()
    array([[0.   , 0.25 , 0.   ],
           [0.25 , 0.   , 0.   ],
           [0.   , 0.125, 0.125],
           [0.   , 0.125, 0.125]])

    We can get the map from owners to record ids (rows):

    >>> m = model.baseline(datasets, "hint", return_owners=True)[1]
    >>> m.collect(streaming="engine")
    shape: (4, 2)
    ┌──────────┬────────┐
    │ owner_id ┆ record │
    │ ---      ┆ ---    │
    │ i64      ┆ u32    │
    ╞══════════╪════════╡
    │ 0        ┆ 0      │
    │ 1        ┆ 1      │
    │ 2        ┆ 2      │
    │ 3        ┆ 3      │
    └──────────┴────────┘

    And the map from hint labels to the corresponding cols in the channel:

    >>> m = model.baseline(datasets, "hint", return_labels=True)[1]
    >>> m.collect(streaming="engine")
    shape: (3, 2)
    ┌─────────────────┬──────┐
    │ hint_label      ┆ hint │
    │ ---             ┆ ---  │
    │ list[struct[1]] ┆ u32  │
    ╞═════════════════╪══════╡
    │ [null, {1}]     ┆ 0    │
    │ [{0}, {0}]      ┆ 1    │
    │ [{1}, {0}]      ┆ 2    │
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
    datasets = list(datasets)
    if len(datasets) == 0: raise ValueError("Empty sequence of datasets!")

    # If only one dataset, dispatch to build(dataset, ...):
    if len(datasets) == 1: return build(
        datasets[0],
        hint,
        owner_col=owner_col,
        n_partitions=n_partitions,
        opt_memory=opt_memory,
        return_owners=return_owners,
        return_labels=return_labels
    )

    # Standardise ``hint`` input
    hint = [hint] if isinstance(hint, str) else hint

    owners_expr = pl.col(owner_col).unique()
    owners = set(datasets[0].select(owners_expr).to_series())

    for i, dataset in enumerate(datasets):
        required = [owner_col, *hint]
        ok, missing = _valid_columns(dataset, required)

        if not ok:
            raise ValueError(f"{i}-th dataset missing attributes: {missing}")

        owners_i = set(dataset.select(owners_expr).to_series())
        if owners_i != owners:
            raise ValueError("All datasets must have the same owners!")

    # =============================================================
    # End pre-conditions
    # =============================================================
     
    # We begin by building the prior for the (possibly longitudinal) dataset:
    records_it = (_mk_records(d, owner_col) for d in datasets)
    long_dataset = _mk_long_dataset(records_it, owner_col)
    pi = _mk_long_prior(long_dataset.drop(owner_col))

    # Now, for each dataset we compute the channel and get the map_labels.
    # We also need to augment the individual datasets so that we get
    # the longitudinal records and get channel rows properly aligned.
    def _build_model(dataset, n_partitions):
        model = build(
            dataset,
            hint,
            owner_col=owner_col,
            n_partitions=n_partitions,
            return_owners=True,
            return_labels=return_labels
        ) 

        joint, map_owners = model[:2]
        reindex = (
            map_owners
            .rename({"record": "row"})
            .join(long_dataset.lazy(), on=owner_col)
            .drop(owner_col)
            .unique()
            .sort("record")
            .select("row")
            .collect(engine="streaming")
            .to_numpy()
            .ravel()
        )

        is_partitioned = isinstance(joint.dist, Sequence)
        joint_dist = joint.dist if is_partitioned else [joint.dist]

        prior_dist = [s.sum(axis=1)[:, np.newaxis] for s in joint_dist]
        prior_dist = np.hstack(prior_dist).sum(axis=1)

        ch_dist = [s / prior_dist[:, np.newaxis] for s in joint_dist]
        ch_dist = [s.tocsr()[reindex, :] for s in ch_dist]
        ch_dist = ch_dist if len(ch_dist) > 1 else ch_dist[0]
        ch = Channel(ch_dist)

        schema = {"hint_label": pl.Struct, "hint": pl.UInt64}
        map_labels = model[2] if return_labels else pl.LazyFrame(schema=schema)
        return ch, map_labels
        

    # We focus all the partitioning into the first dataset of the sequence,
    # so that each intermediate comp will have the desired num of partitions.
    models_it = (
        _build_model(d, n_partitions if i == 0 else 1)
        for i, d in enumerate(datasets)
    )

    ch_seq, map_labels_seq = zip(*models_it)
    
    # With the channel seq and the output labels, we proceed as follows.
    # 
    # For each pair in the sequence, we compute the parallel composition,
    # and we request the column pairs in the original composition.
    # 
    # Given the column pairs, we then need to find the corresponding labels.
    def _compose(model_lhs, next_idx):
        i = next_idx - 1
        j = next_idx
        
        ch_lhs, labels_lhs = model_lhs
        ch_rhs, labels_rhs = ch_seq[j], map_labels_seq[j]

        result = qif.compose.parallel(
            ch_lhs,
            ch_rhs,
            opt_memory=opt_memory,
            return_cols=return_labels
        )

        if not return_labels:
            schema = {"hint_label": pl.Struct, "hint": pl.UInt64}
            return result, pl.DataFrame(schema=schema)

        ch, cols = result
        
        with_suffix = lambda lf, col, s: lf.rename({col: f"{col}_{s}"})
        labels_lhs = with_suffix(labels_lhs, "hint_label", i)
        labels_rhs = with_suffix(labels_rhs, "hint_label", j)

        cols_lf = pl.LazyFrame({ str(i): cols[:, 0], str(j): cols[:, 1] })
        map_labels = (
            cols_lf
            .with_row_index()
            .join(labels_lhs, left_on=str(i), right_on="hint", how="left")
            .drop(str(i))
            .join(labels_rhs, left_on=str(j), right_on="hint", how="left")
            .drop(str(j))
            .rename({"index": "hint"})
        )

        return ch, map_labels


    ch, map_labels = reduce(
        _compose,
        range(1, len(ch_seq)),
        (ch_seq[0], map_labels_seq[0])
    )

    hint_label_expr = pl.concat_list(pl.exclude("hint")).alias("hint_label")
    map_labels = map_labels.select(hint_label_expr, "hint")

    joint = qif.joint(pi, ch)
    map_owners = long_dataset.lazy() # Map owners is just our long_dataset

    if return_owners and return_labels: return joint, map_owners, map_labels
    if return_owners: return joint, map_owners
    if return_labels: return joint, map_labels

    return joint


@multimethod
def build(
    dataset: Dataset,
    hint: Iterable[str],
    owner_col: str = "owner_id",
    n_partitions: int = 1,
    opt_memory: bool = True,
    return_owners: bool = False,
    return_labels: bool = False
) -> BaselineModel:
    # =============================================================
    # Pre-conditions: The dataset must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    # =============================================================
    # Standardise ``hint`` input
    hint = [hint] if isinstance(hint, str) else hint

    required = [owner_col, *hint]
    ok, missing = _valid_columns(dataset, required)

    if not ok: raise ValueError(f"Dataset missing attributes: {missing}")

    # =============================================================
    # End pre-conditions
    # =============================================================

    records = _mk_records(dataset, owner_col)
    long_dataset = _mk_long_dataset([records], owner_col)
    pi = _mk_long_prior(long_dataset.drop(owner_col))

    len_expr = pl.len().alias("len")
    hint_label_expr = pl.struct(hint).alias("hint_label")
    hint_expr = pl.col("hint_label").rank("dense").alias("hint") - 1
    p_expr = (pl.len() / pl.col("len").first()).alias("p")

    ch_metadata = (
        dataset.join(long_dataset, on=owner_col)
        .select(owner_col, "record", hint_label_expr)
        .group_by(owner_col)
        .agg(pl.col("record").first(), "hint_label", len_expr)
        .drop(owner_col)

        # Drop possible duplicate records from the dataset,
        # as in the case of the channel we count things within records
        .unique()
        .explode("hint_label")

        # Then, we compute the probability of each cell in the channel
        .group_by("record", "hint_label")
        .agg(p_expr)

        # and transform the hint labels into col indices:
        .with_columns(hint_expr)
    )

    n_rows = ch_metadata.select("record").max().item() + 1
    def _mk_ch_dist(ch_dist_df):
        # Make hint column compact again (as this is now a partition):
        hint_expr = pl.col("hint_label").rank("dense").alias("hint") - 1
        ch_dist_df = ch_dist_df.select("record", hint_expr, "p")

        # Here we recompute n_cols so we get the number of cols in the slice
        n_cols = ch_dist_df["hint"].max() + 1

        data = ch_dist_df["p"].to_numpy()
        rows = ch_dist_df["record"].to_numpy()
        cols = ch_dist_df["hint"].to_numpy()

        ch_dist = coo_array((data, (rows, cols)), shape=(n_rows, n_cols))
        return ch_dist.tocsr()


    n_cols = ch_metadata.select("hint").max().item() + 1
    n_partitions = max(0, min(n_partitions, n_cols))
    part_expr = (pl.col("hint") % n_partitions).alias("part")

    partitions =  ch_metadata.with_columns(part_expr).partition_by("part")

    ch_dist = [_mk_ch_dist(part_metadata) for part_metadata in partitions]
    ch_dist = ch_dist if len(ch_dist) > 1 else ch_dist[0]

    ch = Channel(ch_dist)
    
    joint = qif.joint(pi, ch)
    map_labels = ch_metadata.lazy().select("hint_label", "hint").unique()
    map_owners = long_dataset.lazy() # Map owners is just our long_dataset

    if return_owners and return_labels: return joint, map_owners, map_labels
    if return_owners: return joint, map_owners
    if return_labels: return joint, map_labels

    return joint
