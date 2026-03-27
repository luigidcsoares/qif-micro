from collections.abc import Iterable, Sequence
from functools import partial
from inspect import signature
from typing import Any

import polars as pl

from scipy.sparse import coo_array, csr_array

from qif_micro import qif
from qif_micro.qif.datatypes import Channel
from qif_micro.typing import AttrMechanism, Record

def build(
    mechanisms: dict[str, AttrMechanism],
    input_domain: Iterable[Record],
    output_domain: Iterable[Record] | None = None
) -> Channel | tuple[Channel, Sequence[Any]]:
    """
    This function generates a mechanism from records to records,
    based on attribute-level mechanisms.

    TODOS
    -----
    Add support to mechanisms with different output domain.
    
    Parameters
    ----------
    mechanisms : dict[str, AttrMechanism]
        A mapping from attributes to mechanisms. If empty, this is
        equivalent to applying the identity mechanism onto each attribute.
        
    input_domain : Iterable[Record]
        A (sub-)domain of records, where each record is a list of dicts.
        This need not be the entire domain; it may be a sub-domain with
        only the records that the adversary believes that are possible.

    output_domain: Iterable[Record], optional (default: None)
        A (sub-)domain of records, where each record is a list of dicts.
        This need not be the entire domain; it may be a sub-domain with
        only the records that the adversary believes that are possible.

        If omitted, we assume it is the same as ``input_domain``.

    Returns
    -------
    Channel
        A channel matrix modelling the mechanism, where the i-th row
        corresponds to the i-th record in ``records``; similarly, the
        i-th column correspond to the i-th record in ``records``.
    
    Examples
    --------
    >>> from functools import partial
    >>> from qif_micro import mechanism

    Consider the following domain of records:
    
    >>> records = [
    ...     [{"q": 0, "s": 0}],
    ...     [{"q": 0, "s": 1}],
    ...     [{"q": 1, "s": 0}],
    ...     [{"q": 1, "s": 1}],
    ... ]

    Then we can apply the mechanism to a single attribute:
    
    >>> rr_q = partial(mechanism.random_response, p=2/3)
    >>> mechanism.record({"q": rr_q}, records).dist.toarray()
    array([[0.66666667, 0.        , 0.33333333, 0.        ],
           [0.        , 0.66666667, 0.        , 0.33333333],
           [0.33333333, 0.        , 0.66666667, 0.        ],
           [0.        , 0.33333333, 0.        , 0.66666667]])

    We can also apply to multiple attributes:
    
    >>> rr_s = partial(mechanism.random_response, p=3/4)
    >>> mechanism.record({"q": rr_q, "s": rr_s}, records).dist.toarray()
    array([[0.5       , 0.16666667, 0.25      , 0.08333333],
           [0.16666667, 0.5       , 0.08333333, 0.25      ],
           [0.25      , 0.08333333, 0.5       , 0.16666667],
           [0.08333333, 0.25      , 0.16666667, 0.5       ]])
    """ 
    # ========================================================================
    # Pre-conditions
    # ========================================================================

    # Attributes must be consistent within records and across all records.
    # To deal with that, we consider missing attributes as null.
    as_df = lambda i, r: pl.DataFrame(r).with_columns(pl.lit(i).alias("rid"))

    output_domain = input_domain if output_domain is None else output_domain
    output_domain = (as_df(i, r) for i, r in enumerate(output_domain))
    output_domain = (
        pl.concat(output_domain, how="diagonal")
        .group_by("rid")
        .agg(pl.all())
        .unique()
    )

    input_domain = (as_df(i, r) for i, r in enumerate(input_domain))
    input_domain = (
        pl.concat(input_domain, how="diagonal")
        .group_by("rid")
        .agg(pl.all())
        .unique()
    )

    attrs = set(output_domain.schema.keys()) - {"rid"}
    for input_attr in set(input_domain.schema.keys()) - {"rid"}:
        if input_attr not in attrs:
            raise ValueError(f"Input and output incompatible: {input_attr}")

    # The attributes to be transformed must match with the records.
    for transform_attr in mechanisms.keys():
        if transform_attr not in attrs:
            raise ValueError(f"{transform_attr} is not a valid attribute!")

    n_input = input_domain.height
    if n_input == 0: raise ValueError("Input domain cannot be empty!")

    n_output = output_domain.height
    if n_output == 0: raise ValueError("Output domain cannot be empty!")

    # If there are no mechanisms, this is just the identity channel.
    transform_attrs = list(mechanisms.keys())
    if len(transform_attrs) == 0: return qif.channel.identity(n_input)

    # ========================================================================
    # We first get the record-level channel for each attr, taking into account
    # the attributes that must be preserved (no mechanism applied)
    preserve_attrs = attrs - set(transform_attrs)
    def _build_for(attr) -> Channel:
        # Min and max record length comes from the input domain
        min_record_len = input_domain[attr].list.len().min()
        max_record_len = input_domain[attr].list.len().max()

        # If the user has not provided the attr input or output domains,
        # we derive them from the domain of records.
        m = mechanisms[attr]
        param = signature(mechanisms[attr]).parameters["input_domain"]
        if param.default is param.empty:
            attr_domain = input_domain[attr].explode().unique().to_list()
            m = partial(m, input_domain=attr_domain)
        
        param = signature(mechanisms[attr]).parameters["output_domain"]
        if param.default is param.empty:
            attr_domain = output_domain[attr].explode().unique().to_list()
            m = partial(m, output_domain=attr_domain)

        ch, row_labels, col_labels = m(return_labels=True)

        domain_size = len(attr_domain)
        map_row_labels = {"row_label": row_labels, "row": range(domain_size)}
        map_row_labels = pl.LazyFrame(map_row_labels)

        map_col_labels = {"col_label": col_labels, "col": range(domain_size)}
        map_col_labels = pl.LazyFrame(map_col_labels)
         
        ch_dist = coo_array(ch.dist)
        data = ch_dist.data
        rows, cols = ch_dist.coords
        
        mechanism_df = (
            pl.LazyFrame({"row": rows, "col": cols, "p": data})
            .join(map_row_labels, on="row").drop("row")
            .join(map_col_labels, on="col").drop("col")
        )
        
        len_expr = pl.col(attr).list.len().alias("len")
        entry_expr = pl.row_index("entry_id").over("rid")
        
        input_entries = (
            input_domain
            .with_columns(len_expr, pl.col(attr).alias("row_label"))
            .explode("row_label")
            .with_columns(entry_expr)
            .partition_by("len", as_dict=True)
        )

        output_entries = (
            output_domain
            .with_columns(len_expr, pl.col(attr).alias("col_label"))
            .explode("col_label")
            .with_columns(entry_expr)
            .partition_by("len", as_dict=True)
        )

        join_cols = subrecord_cols = preserve_attrs | {"col_label", "entry_id"}
        ch_metadata = []
        
        for l in input_entries.keys():
            input_part = input_entries[l].lazy()
            output_part = output_entries[l].lazy()

            metadata_l = (
                input_part
                .join(mechanism_df, on="row_label")
                .join(output_part, on=join_cols)
                .group_by("rid", "rid_right")
                .agg(pl.col("p").product())
            )

            ch_metadata.append(metadata_l)

        ch_metadata = pl.concat(ch_metadata).collect(engine="streaming")
        
        data = ch_metadata["p"].to_numpy()
        rows = ch_metadata["rid"].to_numpy()
        cols = ch_metadata["rid_right"].to_numpy()

        shape = (n_input, n_output)
        ch_dist = coo_array((data, (rows, cols)), shape=shape)
        
        return ch_dist.tocsr()

    # Then we combine each individual channel, element-wise:
    ch_dist = _build_for(transform_attrs[0])
    for attr in transform_attrs[1:]: ch_dist *= _build_for(attr)

    return Channel(ch_dist, is_slice=True)
