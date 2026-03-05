import polars as pl

from scipy.sparse import coo_array

from qif_micro.qif.datatypes import Channel, ProbabDist
from qif_micro._internal import _valid_columns

type ReturnModel = (
    tuple[ProbabDist, Channel]
    | tuple[ProbabDist, Channel, pl.LazyFrame]
    | tuple[ProbabDist, Channel, pl.LazyFrame, pl.LazyFrame]
)

def build(
    dataset: pl.DataFrame | pl.LazyFrame,
    count_col: str = "count",
    sum_col: str = "sum",
    group_by_col: str | None = None,
    owner_col: str = "owner_id",
    return_owners: bool = False,
    return_labels: bool = False
) -> ReturnModel:
    """
    TODO
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

    
    """
    # =============================================================
    # Pre-conditions: The dataset must be in "wide" format, where
    # each row corresponds to the entry of one record, each column
    # corresponds to one of the record's attributes, and there must
    # be a special column that identified the owner of that record.
    #
    # If ``group_by_col`` is not defined, records must have len one.
    #
    # The type of ``count_col`` and ``sum_col`` must be an integer.
    # =============================================================
    dataset = dataset.lazy()
    schema = dataset.collect_schema()

    filter_optional = lambda xs: [x for x in xs if x is not None]
    record_cols = filter_optional([count_col, sum_col, group_by_col])

    required_cols = [owner_col, *record_cols]
    missing_cols = set(required_cols) - set(schema.keys())

    if len(missing_cols) > 0:
        raise ValueError(f"Missing the following attrs: {missing_cols}")

    record_lens = dataset.group_by(owner_col).agg(pl.len())
    max_record_len = record_lens.select(pl.col("len").max()).collect().item()

    if (group_by_col is None) and (max_record_len > 1):
        raise ValueError("Record length must be 1, unless ``group_by_col`` is set`")

    if not schema[count_col].is_integer():
        raise ValueError(f"``count_col`` ({count_col}) must be integer")

    if not schema[sum_col].is_integer():
        raise ValueError(f"``sum_col`` ({sum_col}) must be integer")

    # =============================================================
    # End pre-conditions
    # =============================================================

    # We begin by constructing a map from records to hints,
    # so that each record is identified as a row (of the prior and channel),
    # and each hint is identified as a column (of the channel).
    # 
    # For each unique pair of (count, sum), generate all valid hint ranges:
    hint_start_expr = (
        pl.when(pl.col(count_col) > 1)
        .then(0)
        .otherwise(pl.col(sum_col))
        .alias("hint_start")
    )

    hint_end_expr = (pl.col(sum_col) + 1).alias("hint_end")
    hint_range_expr = pl.struct(hint_start_expr, hint_end_expr).alias("hint_range")

    record_entry_expr = pl.struct(record_cols).alias("record_entry")
    record_expr = pl.col("record_entry").rank("dense").alias("record") - 1

    records_and_hints = (
        dataset
        .with_columns(hint_range_expr)
        .group_by(owner_col)
        .agg(record_entry_expr, pl.all())
        .select(record_expr, pl.exclude("record_entry"))
    )

    # This assumes a uniform prior, so the adversary's revised knowledge
    # upon observing the dataset with histograms is just a frequentist dist:
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

    # We need the sum, count and hint value (which are the vals that were agg):
    n = pl.col(sum_col)
    k = pl.col(count_col)
    h = pl.col("hint")

    log_p_expr = (
        (k - 1).log() - (n + 1).log() 
        + gammaln(n + k - h - 1) - gammaln(n - h + 1)
        + gammaln(n + 2) - gammaln(n + k)
    )

    # The hint will either be a single value (the value that has been agg),
    # or a pair, in case there is a group_by column:
    has_group_by = group_by_col is not None
    hint_label_expr = (
        (pl.struct("hint", group_by_col) if has_group_by else pl.col("hint"))
        .alias("hint_label")
    )

    hint_expr = pl.col("hint_label").rank("dense").alias("hint") - 1

    # The probability of the hint is [hint == n] if the count is 1;
    # else, we follow the formula above using ln of gamma.
    # 
    # Given the way we have constructed the hints, there will
    # be no cell for the case (k == 1) ^ (h != n), so we just set 1 if k == 1.
    #
    # Then, we need to normalise according to the weight of each group_by val,
    # which is essentially the probability Pr(group | record).
    weight_expr = pl.len().over("record", group_by_col) / pl.len().over("record")
    p_expr = (
        pl
        .when(k == 1)
        .then(1)
        .otherwise(log_p_expr.exp())
        .mul(weight_expr if has_group_by else 1)
        .alias("p")
    )
        
    # Now we are ready to construct the actual channel:
    ch_metadata = (
        # We first construct all values that could have been aggregated:
        records_and_hints
        .drop(owner_col)
        .unique()
        .explode(pl.exclude("record"))
        .unnest("hint_range")
        .with_columns(pl.int_ranges("hint_start", "hint_end").alias("hint"))
        .drop("hint_start", "hint_end")
        .explode("hint")

        # Then, we compute the probability of each cell in the channel,
        # and combine the agg value with the group_by value to get the hint:
        .select("record", hint_label_expr, p_expr)

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
