from collections.abc import Iterable

import polars as pl

from scipy.sparse import coo_array

from qif_micro.qif.datatypes import Channel, ProbabDist
from qif_micro._internal import _valid_columns

type ReturnModel = (
    tuple[ProbabDist, Channel] |
    tuple[ProbabDist, Channel, pl.LazyFrame] |
    tuple[ProbabDist, Channel, pl.LazyFrame, pl.LazyFrame]
)

def build(
    dataset: pl.DataFrame | pl.LazyFrame,
    hint_attrs: Iterable[str],
    owner_col: str = "owner_id",
    return_owners: bool = False,
    return_labels: bool = False
) -> ReturnModel:
    """
    Build the adversary’s knowledge model from a dataset and auxiliary info.

    Parameters
    ----------
    dataset : pl.DataFrame or pl.LazyFrame
        The data containing owners, hints and sensitive attributes.

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
        - The adversary’s revised knowledge after observing the dataset.
        - The slice of the hint channel that matches the adversary’s knowledge.

    tuple (ProbabDist, Channel, pl.LazyFrame) |
        - The adversary’s revised knowledge after observing the dataset.
        - The slice of the hint channel that matches the adversary’s knowledge.
        - If ``map_owners`` enabled: map from owners to row indices;
          If ``map_labels`` enabled: map from hint labels to indices

    tuple (ProbabDist, Channel, pl.LazyFrame)
        - (Optional) A Polars ``LazyFrame`` that maps records to hints.
        - The adversary’s revised knowledge after observing the dataset.
        - The slice of the hint channel that matches the adversary’s knowledge.
        - Map from owners to row indices;
        - Map from hint labels to indices

    Examples
    --------
    >>> import polars as pl
    >>> from qif_micro import model

    Consider the following data:
    >>> dataset = pl.DataFrame({
    ...     "owner_id": [0, 1, 2, 2, 3, 3],
    ...     "hint": [0, 0, 0, 1, 0, 1],
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
        raise ValueError(f"Missing the following attrs: {missing_attrs}")

    # =============================================================
    # End pre-conditions
    # =============================================================

    # We begin by constructing a map from records to hints,
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

    n_records_expr = pl.len().alias("n_records")
    p_expr = (pl.len() / pl.col("n_records").first()).alias("p")

    prior_dist = (
        records_and_hints
        .drop(owner_col)
        .with_columns(n_records_expr)
        .group_by("record")
        .agg(p_expr)
        .sort("record")
        .select("p")
        .collect()
        .to_numpy()
        .ravel()
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

    pi = ProbabDist(prior_dist)
    ch = Channel(ch_dist.tocsr())
    
    map_owners = records_and_hints.select(owner_col, "record")
    map_labels = ch_metadata.select("hint_label", "hint").unique()

    if return_owners and return_labels: return pi, ch, map_owners, map_labels
    if return_owners: return pi, ch, map_owners
    if return_labels: return pi, ch, map_labels

    return pi, ch
