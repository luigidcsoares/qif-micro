from collections.abc import Iterable, Sequence
from functools import partial
from inspect import signature
from typing import Any

from multimethod import multimethod
import numpy as np
import polars as pl
import scipy.sparse as sp

from qif_micro import qif
from qif_micro.qif.datatypes import Channel
from qif_micro.typing import AttrMechanism, DataFrame, Record

@multimethod
def build(
    input_domain: DataFrame,
    output_domain: DataFrame | None = None,
    record_col: str = "record_id",
    entry_col: str = "entry_id",
    **mechanisms: AttrMechanism,
) -> Channel | tuple[Channel, Sequence[Any]]:
    """
    This function generates a mechanism from records to records,
    based on attribute-level mechanisms.

    This function is overloaded:

    - ``build(mechanisms, input_domain, ...)``: accepts a :class:`DataFrame`
    - ``build(mechanisms, input_domain, ...)``: accepts an iterable of records

    Parameters
    ----------
    mechanisms : dict[str, AttrMechanism]
        A mapping from attributes to mechanisms. If empty, this is
        equivalent to applying the identity mechanism onto each attribute.
        
    input_domain : Iterable[Record]
        An iterable of records (list of dicts).
        (First overload)

    input_domain : DataFrame
        A DataFrame where each row represents an entry in a record.
        The DataFrame must have columns ``record_col`` and ``entry_col``
        (by default ``"record_id"`` and ``"entry_id"``) identifying each
        record and entry. Other columns represent the attributes.
        (Second overload)

    output_domain : Iterable[Record], optional (default: None)
        An iterable of records with the same structure as ``input_domain``.
        If omitted, we assume it is the same as ``input_domain``.
        (First overload)

    output_domain: DataFrame, optional (default: None)
        A DataFrame with the same structure as ``input_domain``.
        If omitted, we assume it is the same as ``input_domain``.
        (Second overload)

    record_col : str, optional (default: ``"record_id"``)
        Column name for the record identifier.

    entry_col : str, optional (default: ``"entry_id"``)
        Column name for the entry identifier within each record.

    Returns
    -------
    Channel
        A channel matrix modelling the mechanism, where the i-th row
        corresponds to the i-th record in the input domain; similarly,
        the i-th column correspond to the i-th record in the output
        domain.

    Pre-conditions
    --------------
    - ``input_domain`` must have ``record_col`` and ``entry_col`` columns
      (defaults: "record_id", "entry_id") identifying records and entries.
    - ``input_domain`` must contain at least one record.
    - If ``output_domain`` is provided, it must have the same structure as
      ``input_domain`` and contain at least one record.
    - All ``mechanisms`` keys must correspond to columns in the input domain
      (excluding ID columns).
    - Input and output domains must have compatible attributes (same set
      of attributes except possibly different mechanisms applied).

    Examples
    --------
    >>> from functools import partial
    >>> from qif_micro import mechanism

    Consider the following domain of records as a list of lists:
    
    >>> records = [
    ...     [{"q": 0, "s": 0}],
    ...     [{"q": 0, "s": 1}],
    ...     [{"q": 1, "s": 0}],
    ...     [{"q": 1, "s": 1}],
    ... ]

    We can apply the mechanism to a single attribute:
    
    >>> rr_q = partial(mechanism.random_response, p=2/3)
    >>> mechanism.record(records, q=rr_q).dist.toarray()
    array([[0.66666667, 0.        , 0.33333333, 0.        ],
           [0.        , 0.66666667, 0.        , 0.33333333],
           [0.33333333, 0.        , 0.66666667, 0.        ],
           [0.        , 0.33333333, 0.        , 0.66666667]])

    We can also apply to multiple attributes:
    
    >>> rr_s = partial(mechanism.random_response, p=3/4)
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
    # Pre-conditions
    # ========================================================================
    if output_domain is None: output_domain = input_domain
        
    # Group the DataFrame by record and entry
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
        .item()
    )

    if n_input == 0: raise ValueError("Input domain cannot be empty!")

    n_output = (
        output_domain
        .select(pl.col(record_col).n_unique())
        .collect(engine="streaming")
        .item()
    )

    if n_output == 0: raise ValueError("Output domain cannot be empty!")

    # If there are no mechanisms, this is just the identity channel.
    transform_attrs = list(mechanisms.keys())
    if len(transform_attrs) == 0: return qif.channel.identity(n_input)

    # ========================================================================
    # We first get the record-level channel for each attr, taking into account
    # the attributes that must be preserved (no mechanism applied)
    n_input_records = (
        input_domain
        .select(pl.col(record_col).max() + 1)
        .collect(engine="streaming")
        .item()
    )

    n_output_records = (
        output_domain
        .select(pl.col(record_col).max() + 1)
        .collect(engine="streaming")
        .item()
    )

    preserve_attrs = attrs - set(transform_attrs)
    def _build_for(attr) -> Channel:
        # Min and max record length comes from the input domain
        min_record_len = (
            input_domain
            .select(pl.col(attr).list.len().min())
            .collect(engine="streaming")
            .item()
        )

        max_record_len = (
            input_domain
            .select(pl.col(attr).list.len().max())
            .collect(engine="streaming")
            .item()
        )

        # If the user has not provided the attr input or output domains,
        # we derive them from the domain of records.
        m = mechanisms[attr]
        param = signature(mechanisms[attr]).parameters["input_domain"]
        if param.default is param.empty:
            attr_domain = (
                input_domain
                .select(attr)
                .explode(attr)
                .unique()
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )
            m = partial(m, input_domain=attr_domain)
        
        param = signature(mechanisms[attr]).parameters["output_domain"]
        if param.default is param.empty:
            attr_domain = (
                output_domain
                .select(attr)
                .explode(attr)
                .unique()
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )
            m = partial(m, output_domain=attr_domain)


        ch, row_labels, col_labels = m(return_labels=True)

        n_rows_m = len(row_labels)
        map_row_labels = {"row_label": row_labels, "row": range(n_rows_m)}
        map_row_labels = pl.LazyFrame(map_row_labels)

        n_cols_m = len(col_labels)
        map_col_labels = {"col_label": col_labels, "col": range(n_cols_m)}
        map_col_labels = pl.LazyFrame(map_col_labels)
         
        ch_dist = sp.coo_array(ch.dist)
        data = ch_dist.data
        rows, cols = ch_dist.coords
        
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
            .partition_by("len", as_dict=True)
        )

        output_entries = (
            output_domain
            .with_columns(len_expr)
            .rename({attr: "col_label"})
            .explode("col_label")
            .with_columns(entry_expr)
            .collect(engine="streaming")
            .partition_by("len", as_dict=True)
        )

        join_cols = preserve_attrs | {"col_label", entry_col}
        ch_metadata = []
        
        for l in input_entries.keys():
            input_part = input_entries[l].drop("len").lazy()
            output_part = output_entries[l].drop("len").lazy()

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
                .group_by(record_col, f"{record_col}_right").agg("p")
                .filter(pl.col("p").list.len() == l[0])

                # Once we have only the compatible records, we can
                # compute the probability as a product for each entry.
                .with_columns(pl.col("p").list.agg(pl.element().product()))
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
    for attr in transform_attrs[1:]: ch_dist *= _build_for(attr)

    return Channel(ch_dist)


@multimethod
def build(
    input_domain: Iterable[Record],
    output_domain: Iterable[Record] | None = None,
    record_col: str = "record_id",
    entry_col: str = "entry_id",
    **mechanisms: AttrMechanism,
) -> Channel | tuple[Channel, Sequence[Any]]:
    as_df = lambda i, r:  pl.LazyFrame(r).with_columns(
        pl.lit(i).alias(record_col),
        pl.row_index(entry_col)
    )

    input_domain = (as_df(i, r) for i, r in enumerate(input_domain))
    input_domain = pl.concat(input_domain, how="diagonal")

    if output_domain is not None:
        output_domain = (as_df(i, r) for i, r in enumerate(output_domain))
        output_domain = pl.concat(output_domain, how="diagonal")

    return build(
        input_domain,
        output_domain,
        record_col=record_col,
        entry_col=entry_col,
        **mechanisms
    )
