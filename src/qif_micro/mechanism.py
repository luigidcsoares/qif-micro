from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import polars as pl
import scipy.sparse as sp

from qif_micro import qif
from qif_micro.qif.datatypes import Channel
from qif_micro.typing import DataFrame, is_dataframe, Record


@runtime_checkable
class Mechanism(Protocol):
    def __call__(
        self,
        input_domain: Iterable[Any],
        output_domain: Iterable[Any] | None = None,
    ) -> Channel: ...


def record(
    input_domain: DataFrame | Iterable[Record],
    output_domain: DataFrame | Iterable[Record] | None = None,
    record_col: str = "record_id",
    entry_col: str = "entry_id",
    **mechanisms: Mechanism,
) -> Channel:
    """
    Build a record-level mechanism from attribute-level mechanisms.

    Constructs a stochastic matrix representing how records map to records
    under the given privacy mechanisms applied to individual attributes.

    Parameters
    ----------
    input_domain : Iterable[Record] | DataFrame
        The domain of input records.
        - If ``Iterable[Record]``: a sequence of records (list of dicts).
        - If ``DataFrame``: each row is an entry in a record, with columns
          ``record_col`` and ``entry_col`` (defaults: "record_id", "entry_id")
          identifying records and entries.

    output_domain : Iterable[Record] or DataFrame, optional (default: None)
        The domain of output records (same structure as ``input_domain``).
        If omitted, defaults to ``input_domain``.

    record_col : str, optional (default: "record_id")
        Column name for the record identifier (DataFrame only).

    entry_col : str, optional (default: "entry_id")
        Column name for the entry identifier within each record
        (DataFrame only).

    **mechanisms : Mechanism
        Mapping from attribute names to privacy mechanisms. Each mechanism
        is a callable taking (input_domain, output_domain) and returning a
        Channel. If empty, applies identity to each attribute.

    Returns
    -------
    Channel
        A channel matrix where row i corresponds to input record i and
        column j corresponds to output record j, representing the
        probability of transforming one record to another.

    Pre-conditions
    --------------
    - For DataFrame inputs, ``record_col`` and ``entry_col`` columns must
      exist and identify records and entries.

    - Input domains must contain at least one record.

    - If ``output_domain`` is provided, it must have compatible structure
      (that is, same attributes as ``input_domain``).

    - All mechanism keys must correspond to attribute columns in the domain

    - Mechanisms must accept domains and return valid Channels.

    Examples
    --------
    >>> from functools import partial
    >>> import math
    >>> from qif_micro import mechanism
    >>> from qif_micro import qif

    Consider the following domain of records as a list of lists:
    
    >>> records = [
    ...     [{"q": 0, "s": 0}],
    ...     [{"q": 0, "s": 1}],
    ...     [{"q": 1, "s": 0}],
    ...     [{"q": 1, "s": 1}],
    ... ]

    We can apply the mechanism to a single attribute:
    
    >>> eps = math.log(2) # p = 2/3
    >>> rr_q = partial(qif.dp.random_response, eps=eps)
    >>> mechanism.record(records, q=rr_q).dist.toarray()
    array([[0.66666667, 0.        , 0.33333333, 0.        ],
           [0.        , 0.66666667, 0.        , 0.33333333],
           [0.33333333, 0.        , 0.66666667, 0.        ],
           [0.        , 0.33333333, 0.        , 0.66666667]])

    We can also apply to multiple attributes:
    
    >>> eps = math.log(3) # p = 3/4
    >>> rr_s = partial(qif.dp.random_response, eps=eps)
    >>> mechanism.record(records, q=rr_q, s=rr_s).dist.toarray()
    array([[0.5       , 0.16666667, 0.25      , 0.08333333],
           [0.16666667, 0.5       , 0.08333333, 0.25      ],
           [0.25      , 0.08333333, 0.5       , 0.16666667],
           [0.08333333, 0.25      , 0.16666667, 0.5       ]])

    The domain of records can also be a DataFrame, in which case each row
    represents an entry and must have ``record_id`` and ``entry_id`` columns:

    >>> records = pl.from_records([
    ...     {"record_id": 0, "entry_id": 0, "q": 0, "s": 0},
    ...     {"record_id": 1, "entry_id": 0, "q": 0, "s": 1},
    ...     {"record_id": 2, "entry_id": 0, "q": 1, "s": 0},
    ...     {"record_id": 3, "entry_id": 0, "q": 1, "s": 1},
    ... ])

    >>> mechanism.record(records, q=rr_q, s=rr_s).dist.toarray()
    array([[0.5       , 0.16666667, 0.25      , 0.08333333],
           [0.16666667, 0.5       , 0.08333333, 0.25      ],
           [0.25      , 0.08333333, 0.5       , 0.16666667],
           [0.08333333, 0.25      , 0.16666667, 0.5       ]])
    """ 
    # ========================================================================
    # Pre-processing inputs
    # ========================================================================
    def as_df(i, r): return pl.LazyFrame(r).with_columns(
        pl.lit(i).alias(record_col),
        pl.row_index(entry_col)
    )

    if output_domain is None:
        output_domain = input_domain

    if not is_dataframe(input_domain):
        input_domain = (as_df(i, r) for i, r in enumerate(input_domain))
        input_domain = pl.concat(input_domain, how="diagonal")

    if not is_dataframe(output_domain):
        output_domain = (as_df(i, r) for i, r in enumerate(output_domain))
        output_domain = pl.concat(output_domain, how="diagonal")

    # Group the DataFrames by record and entry
    input_domain = (
        input_domain
        .lazy()
        .sort(record_col, entry_col)
        .group_by(record_col)
        .agg(pl.all())
        .unique()
    )

    output_domain = (
        output_domain
        .lazy()
        .sort(record_col, entry_col)
        .group_by(record_col)
        .agg(pl.all())
        .unique()
    )

    # ========================================================================
    # Pre-conditions
    # ========================================================================
    id_cols = {record_col, entry_col}
    attrs = set(output_domain.collect_schema()) - id_cols
    for input_attr in set(input_domain.collect_schema()) - id_cols:
        if input_attr not in attrs:
            raise ValueError(f"Input and output incompatible: {input_attr}")

    # The attributes to be transformed must match with the records.
    for transform_attr in mechanisms.keys():
        if transform_attr not in attrs:
            raise ValueError(f"{transform_attr} is not a valid attribute!")

    n_input = (
        input_domain
        .select(pl.col(record_col).n_unique())
        .collect(engine="streaming")
        .item()  # ty:ignore[unresolved-attribute]
    )

    if n_input == 0:
        raise ValueError("Input domain cannot be empty!")

    n_output = (
        output_domain
        .select(pl.col(record_col).n_unique())
        .collect(engine="streaming")
        .item()  # ty:ignore[unresolved-attribute]
    )

    if n_output == 0:
        raise ValueError("Output domain cannot be empty!")

    # If there are no mechanisms, this is just the identity channel.
    transform_attrs = list(mechanisms.keys())
    if len(transform_attrs) == 0:
        return qif.channel.identity(n_input)

    # ========================================================================
    # We first get the record-level channel for each attr, taking into account
    # the attributes that must be preserved (no mechanism applied)
    n_input_records = (
        input_domain
        .select(pl.col(record_col).max() + 1)
        .collect(engine="streaming")
        .item()  # ty:ignore[unresolved-attribute]
    )

    n_output_records = (
        output_domain
        .select(pl.col(record_col).max() + 1)
        .collect(engine="streaming")
        .item()  # ty:ignore[unresolved-attribute]
    )

    preserve_attrs = attrs - set(transform_attrs)

    def _build_for(attr):
        attr_input_domain = (
            input_domain
            .select(pl.col(attr).alias("row_label"))
            .explode("row_label").unique().sort("row_label")
            .collect(engine="streaming")
        )

        attr_output_domain = (
            output_domain
            .select(pl.col(attr).alias("col_label"))
            .explode("col_label").unique().sort("col_label")
            .collect(engine="streaming")
        )

        ch = mechanisms[attr](
            input_domain=attr_input_domain.to_series().to_list(),  # ty:ignore[unresolved-attribute]
            output_domain=attr_output_domain.to_series().to_list(),  # ty:ignore[unresolved-attribute]
        )

        map_row_labels = attr_input_domain.lazy().with_row_index("row")  # ty:ignore[unresolved-attribute]
        map_col_labels = attr_output_domain.lazy().with_row_index("col")  # ty:ignore[unresolved-attribute]

        dist = sp.coo_array(ch.dist)
        data, rows, cols = dist.data, *dist.coords
        
        mechanism_df = (
            pl.LazyFrame({"row": rows, "col": cols, "p": data})
            .join(map_row_labels, on="row").drop("row")
            .join(map_col_labels, on="col").drop("col")
        )
        
        len_expr = pl.col(attr).list.len().alias("len")
        entry_expr = pl.row_index(entry_col).over(record_col)
        
        input_entries = (
            input_domain
            .with_columns(len_expr)
            .rename({attr: "row_label"})
            .explode("row_label")
            .with_columns(entry_expr)
            .collect(engine="streaming")
            .partition_by("len", as_dict=True)  # ty:ignore[unresolved-attribute]
        )

        output_entries = (
            output_domain
            .with_columns(len_expr)
            .rename({attr: "col_label"})
            .explode("col_label")
            .with_columns(entry_expr)
            .collect(engine="streaming")
            .partition_by("len", as_dict=True)  # ty:ignore[unresolved-attribute]
        )

        join_cols = list(preserve_attrs | {"col_label", entry_col})
        ch_metadata = []
        
        for n_entries in input_entries.keys():
            input_part = input_entries[n_entries].drop("len").lazy()
            output_part = output_entries[n_entries].drop("len").lazy()

            metadata_l = (
                # We first join with the mechanism to get the possible
                # sanitised values, and join with the output_part to
                # get records that are candidate to be compatible.
                input_part
                .join(mechanism_df, on="row_label").drop("row_label")
                .join(output_part, on=join_cols)

                # For records with length > 1, it may be that we get
                # a few entries compatible, but not all of them.
                # In this case we must discard such records.
                .group_by(record_col, f"{record_col}_right")
                .agg(pl.len(), pl.col("p").product())
                .filter(pl.col("len") == n_entries[0])
            )

            ch_metadata.append(metadata_l)

        ch_metadata = pl.concat(ch_metadata).collect(engine="streaming")
        
        data = ch_metadata["p"].to_numpy()
        rows = ch_metadata[record_col].to_numpy()
        cols = ch_metadata[record_col + "_right"].to_numpy()

        shape = (n_input_records, n_output_records)
        ch_dist = sp.coo_array((data, (rows, cols)), shape=shape)
        return ch_dist.tocsr()

    # Then we combine each individual channel, element-wise:
    ch_dist = _build_for(transform_attrs[0])

    for attr in transform_attrs[1:]:
        ch_dist *= _build_for(attr)

    return Channel(ch_dist)
