from collections.abc import Iterable, Sequence
from functools import reduce

import polars as pl

from multimethod import multimethod
from scipy.sparse import coo_array

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, ProbabDist
from qif_micro._internal import _valid_columns

type Dataset = pl.DataFrame | pl.LazyFrame
type MapOwners = pl.DataFrame | pl.LazyFrame
type MapLabels = pl.DataFrame | pl.LazyFrame

def _mk_records(dataset: Dataset, owner_col: str = "owner_id") -> Dataset:
    record_entry_expr = pl.struct(pl.exclude(owner_col)).alias("record")
    return dataset.group_by(owner_col).agg(record_entry_expr)


def _mk_long_dataset(
    map_owners: Iterable[MapOwners],
    owner_col: str = "owner_id"
) -> Dataset:
    # First we construct the new the longitudinal records
    record_idx_expr = lambda i: pl.struct("record", pl.lit(i).alias("i")).alias("record")
    monthly_records = (m.select(owner_col, record_idx_expr(i)) for i, m in enumerate(map_owners))

    record_expr = pl.col("record").rank("dense") - 1
    return (
        pl.concat(monthly_records)
        # The longitudinal record will be a sequence of record ids
        .group_by(owner_col).agg("record")
        # Which we then transform in a single id (row)
        .with_columns(record_expr)
    )


def _mk_long_prior(long_dataset : Dataset) -> ProbabDist:
    n_records_expr = pl.len().alias("n_records")
    p_expr = (pl.len() / pl.col("n_records").first()).alias("p")

    prior_dist = (
        long_dataset
        .lazy()
        .with_columns(n_records_expr) # Should be the same as before
        .group_by("record")
        .agg(p_expr)
        .sort("record")
        .select("p")
        .collect()
        .to_numpy()
        .ravel()
    )

    return ProbabDist(prior_dist)


type ReturnModel = (
    tuple[ProbabDist, Channel]
    | tuple[ProbabDist, Channel, MapOwners | MapLabels]
    | tuple[ProbabDist, Channel, MapOwners, MapLabels]
)

@multimethod
def build(
    dataset: Dataset,
    hint_attrs: Iterable[str],
    owner_col: str = "owner_id",
    n_partitions: int | Iterable[int] = 1,
    return_owners: bool = False,
    return_labels: bool = False
) -> ReturnModel:
    """
    Build the adversary’s knowledge model from a dataset and auxiliary info.

    Parameters
    ----------
    dataset : Dataset
        A dataset containing owners, hints and sensitive attributes.

    hint_attrs : iterable of str
        Attributes that represent the adversary’s auxiliary information.

    owner_col : str, optional (default: ``"owner_id"``)
        Column name for the owner identifier.

    return_owners : bool, optional (default: ``False``)
        If true, the result includes a map from owners to row_indices.

    return_labels : bool, optional (default: ``False``)
        If true, the result includes a map from hint labels to column indices.

    Returns
    -------
    tuple (ProbabDist, Channel) 
        - The adversary’s revised knowledge after observing the dataset;
        - The slice of the hint channel that matches the adversary’s knowledge.

    tuple (ProbabDist, Channel, MapOwners | MapLabels)
        - The adversary’s revised knowledge after observing the dataset;
        - The slice of the hint channel that matches the adversary’s knowledge;
        - If ``map_owners`` enabled: map from owners to row indices OR
          If ``map_labels`` enabled: map from hint labels to indices.

    tuple (ProbabDist, Channel, MapOwners, MapLabels)
        - The adversary’s revised knowledge after observing the dataset;
        - The slice of the hint channel that matches the adversary’s knowledge;
        - Map from owners to row indices;
        - Map from hint labels to indices.

    Examples
    --------
    >>> import polars as pl
    >>> from qif_micro import model

    Consider the following data:
    >>> dataset = pl.DataFrame({
    ...     "owner_id":  [0, 1, 2, 2, 3, 3],
    ...     "hint":      [0, 0, 0, 1, 0, 1],
    ...     "sensitive": [0, 0, 0, 0, 1, 1]
    ... })

    The adversary's knowledge upon observing this dataset is:
    >>> pi, ch = model.baseline(dataset, ["hint"])
    >>> pi
    ProbabDist(dist=array([0.5 , 0.25, 0.25]))

    And the channel modelling the adversary's auxiliary info is:
    >>> ch.dist.toarray()
    array([[1. , 0. ],
           [0.5, 0.5],
           [0.5, 0.5]])

    We can also construct a longitudinal model. Consider a second dataset:
    >>> dataset_rhs = pl.DataFrame({
    ...     "owner_id":  [0, 1, 2, 3],
    ...     "hint":      [0, 1, 0, 0],
    ...     "sensitive": [0, 0, 0, 1]
    ... })

    >>> datasets = [dataset, dataset_rhs]
    >>> pi, ch = model.baseline(datasets, ["hint"])
    >>> pi
    ProbabDist(dist=array([0.25, 0.25, 0.25, 0.25]))

    >>> ch.dist.toarray()
    array([[0. , 1. , 0. ],
           [0. , 1. , 0. ],
           [0. , 0.5, 0.5],
           [1. , 0. , 0. ]])

    We can get the map from owners to record ids (rows):
    >>> m = model.baseline(datasets, ["hint"], return_owners=True)[2]
    >>> m.sort("owner_id").collect()
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
    >>> m = model.baseline(datasets, ["hint"], return_labels=True)[2]
    >>> m.sort("hint_label").collect()
    shape: (3, 2)
    ┌─────────────────┬──────┐
    │ hint_label      ┆ hint │
    │ ---             ┆ ---  │
    │ list[struct[1]] ┆ u32  │
    ╞═════════════════╪══════╡
    │ [{0}, {0}]      ┆ 1    │
    │ [{1}, null]     ┆ 0    │
    │ [{1}, {0}]      ┆ 2    │
    └─────────────────┴──────┘
    """
    # =============================================================
    # Pre-conditions: The dataset must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    # =============================================================
    dataset = dataset.lazy()
    schema = dataset.collect_schema()

    required_attrs = [owner_col, *hint_attrs]
    missing_attrs = set(required_attrs) - set(schema.keys())

    if len(missing_attrs) > 0:
        raise ValueError(f"Dataset missing the following attrs: {missing_attrs}")

    # =============================================================
    # End pre-conditions
    # =============================================================

    # We begin by building the prior for the (possibly longitudinal) dataset:
    records = _mk_records(dataset, owner_col)
    long_dataset = _mk_long_dataset([records], owner_col)
    pi = _mk_long_prior(long_dataset.drop(owner_col))

    # Then we build a map from owners to records to hints,
    # so that each record is identified as a row (of the prior and channel),
    # and each hint is identified as a column (of the channel).
    # 
    # We also add the record length as a metadata.
    len_expr = pl.len().alias("len")
    hint_label_expr = pl.struct(hint_attrs).alias("hint_label")
    record_entry_expr = pl.struct(pl.exclude(owner_col)).alias("record_entry")
    record_expr = pl.col("record_entry").rank("dense").alias("record") - 1

    record_attrs = [c for c in schema.keys() if c != owner_col]
    records_and_hints = (
        dataset
        .select(owner_col, record_entry_expr, hint_label_expr)
        .group_by(owner_col)
        .agg("record_entry", "hint_label", len_expr)
        .select(owner_col, record_expr, "hint_label", "len")
    )

    p_expr = (pl.len() / pl.col("len").first()).alias("p")
    hint_expr = pl.col("hint_label").rank("dense").alias("hint") - 1

    ch_metadata = (
        records_and_hints
        .drop(owner_col)
        # Drop possible duplicate records from the dataset,
        # as in the case of the channel we count things within records
        .unique()
        .explode("hint_label")

        # Then, we compute the probability of each cell in the channel,
        .group_by("record", "hint_label")
        .agg(p_expr)

        # Transform the hint labels into col indices,
        # and sort so we can construct the channel matrix:
        .with_columns(hint_expr)
        .sort("record", "hint")
    )

    ch_dist_df =  ch_metadata.select("record", "hint", "p").collect()

    n_rows = ch_dist_df["record"].max() + 1
    n_cols = ch_dist_df["hint"].max() + 1

    data = ch_dist_df["p"].to_numpy()
    rows = ch_dist_df["record"].to_numpy()
    cols = ch_dist_df["hint"].to_numpy()

    ch_dist = coo_array((data, (rows, cols)), shape=(n_rows, n_cols))

    ch = Channel(ch_dist.tocsr())
    
    map_owners = records_and_hints.select(owner_col, "record")
    map_labels = ch_metadata.select("hint_label", "hint").unique()

    if return_owners and return_labels: return pi, ch, map_owners, map_labels
    if return_owners: return pi, ch, map_owners
    if return_labels: return pi, ch, map_labels

    return pi, ch


@multimethod
def build(
    datasets: Sequence[Dataset],
    hint_attrs: Iterable[str],
    owner_col: str = "owner_id",
    n_partitions: int | Iterable[int] = 1,
    return_owners: bool = False,
    return_labels: bool = False
) -> ReturnModel:
    # =============================================================
    # Pre-conditions: The dataset must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    #
    # If more than one dataset, they must contain the same owners.
    # =============================================================
    if len(datasets) == 0: raise ValueError("Empty sequence of datasets!")

    # If only one dataset, dispatch to build(dataset, ...):
    if len(datasets) == 1: return build(
        datasets[0],
        hint_attrs,
        owner_col,
        n_partitions,
        return_owners,
        return_labels
    )

    # More than one dataset, so we must check owners:
    datasets = [d.lazy() for d in datasets]

    owners_expr = pl.col(owner_col).unique()
    owners = set(datasets[0].select(owners_expr).collect().to_series())

    for i, dataset in enumerate(datasets):
        schema = dataset.collect_schema()

        required_attrs = [owner_col, *hint_attrs]
        missing_attrs = set(required_attrs) - set(schema.keys())

        if len(missing_attrs) > 0:
            raise ValueError(f"{i}-th dataset missing the following attrs: {missing_attrs}")

        owners_i = set(datasets[i].select(owners_expr).collect().to_series())
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
    # For that, we just join the dataset with long_dataset to get the
    # id of the longitudinal record (row in the channel).
    build_model = lambda dataset: build(
        dataset.join(long_dataset, on=owner_col),
        hint_attrs, owner_col, n_partitions,
        return_labels=True, return_owners=True
    ) 

    models_it = (build_model(d) for d in datasets)
    _, ch_seq, map_owners_seq, map_labels_seq = zip(*models_it)
    
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

        with_suffix = lambda lf, col, s: lf.rename({col: f"{col}_{s}"})
        labels_lhs = with_suffix(labels_lhs, "hint_label", i)
        labels_rhs = with_suffix(labels_rhs, "hint_label", j)

        ch, cols = qif.compose.parallel(ch_lhs, ch_rhs, return_cols=True)
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
        lambda acc_model, next_idx : _compose(acc_model, next_idx),
        range(1, len(ch_seq)),
        (ch_seq[0], map_labels_seq[0])
    )

    # Map owners in this case is the same for any individual model:
    hint_label_expr = pl.concat_list(pl.exclude("hint")).alias("hint_label")
    map_owners = map_owners_seq[0]
    map_labels = map_labels.select(hint_label_expr, "hint")

    if return_owners and return_labels: return pi, ch, map_owners, map_labels
    if return_owners: return pi, ch, map_owners
    if return_labels: return pi, ch, map_labels

    return pi, ch
