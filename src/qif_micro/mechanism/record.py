from collections.abc import Iterable
from typing import Any, Protocol

import polars as pl

from scipy.sparse import coo_array, csr_array

from qif_micro import qif
from qif_micro.qif.datatypes import Channel

class Mechanism(Protocol):
    def __call__(self, domain: Iterable[Any], **kwargs: Any) -> Channel:
        ...
        
type RecordEntry = dict[str, Any]
type Record = list[RecodEntry]

def build(
    records: Iterable[Record],
    mechanisms: dict[str, Mechanism],
) -> Channel:
    """
    This function generates a mechanism from records to records,
    based on attribute-level mechanisms.
    
    Parameters
    ----------
    records : Iterable{record}
        The domain of records, where each record is a list of dicts.

    mechanisms : dict[str, Mechanism]
        A mapping from attributes to mechanisms. If empty, this is
        equivalent to applying the identity mechanism onto each attribute.

    Returns
    -------
    Channel
        A channel matrix modelling the mechanism, where the i-th row
        corresponds to the i-th record in ``records``; similarly, the
        i-th column correspond to the i-th record in ``records``.
    
    Examples
    --------
    >>> from qif_micro import mechanism

    Consider the following domain of records:
    
    >>> records = [
    ...     [{"q": 0, "s": 0}],
    ...     [{"q": 0, "s": 1}],
    ...     [{"q": 1, "s": 0}],
    ...     [{"q": 1, "s": 1}],
    ... ]

    We can apply a mechanism to a single attribute:
    
    >>> rr = mechanism.random_response
    >>> rr_q = lambda domain, **kwargs: rr(domain, domain, 2/3, **kwargs)
    >>> mechanism.record(records, {"q": rr_q}).dist.toarray()
    array([[0.66666667, 0.        , 0.33333333, 0.        ],
           [0.        , 0.66666667, 0.        , 0.33333333],
           [0.33333333, 0.        , 0.66666667, 0.        ],
           [0.        , 0.33333333, 0.        , 0.66666667]])

    And we can also apply to multiple attributes:
    
    >>> rr_s = lambda domain, **kwargs: rr(domain, domain, 3/4, **kwargs)
    >>> mechanism.record(records, {"q": rr_q, "s": rr_s}).dist.toarray()
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
    records = (as_df(i, r) for i, r in enumerate(records))

    records_df = (
        pl.concat(records, how="diagonal")
        .group_by("rid")
        .agg(pl.all())
        .unique()
    )

    n_records = records_df.height
    if n_records == 0: raise ValueError("Domain cannot be empty!")

    # If there are no mechanisms. this is just the identity channel.
    transform_attrs = list(mechanisms.keys())
    if len(transform_attrs) == 0: return qif.channel.identity(n_records)

    # The attributes to be transformed must match with the records.
    attrs = set(records_df.schema.keys()) - {"rid"}
    for transform_attr in mechanisms.keys():
        if transform_attr not in attrs:
            raise ValueError(f"{transform_attr} is not a valid attribute!")

    # ========================================================================

    # We first get the record-level channel for each attr, taking into account
    # the attributes that must be preserved (no mechanism applied)
    preserve_attrs = attrs - set(transform_attrs)
    def _build_for(attr) -> Channel:
        max_record_len = records_df[attr].list.len().max()
        attr_domain = records_df[attr].explode().unique().to_list()
        
        domain_singleton = {"col_label": [[a] for a in attr_domain]}
        seq_domain = [pl.DataFrame(domain_singleton)]

        for record_len in range(2, max_record_len + 1):
            curr_domain = seq_domain[-1]
            next_domain = curr_domain.join(seq_domain[0], how="cross")
            
            concat_expr = pl.concat_list("col_label", "col_label_right")
            concat_expr = concat_expr.alias("col_label")
            seq_domain.append(next_domain.select(concat_expr))

        seq_domain = pl.concat(seq_domain)

        # Get the attribute-level mechanism with the labels:
        result = mechanisms[attr](attr_domain, return_labels=True)
        ch, row_labels, col_labels = result

        domain_size = len(attr_domain)
        map_row_labels = {"row_label": row_labels, "row": range(domain_size)}
        map_row_labels = pl.DataFrame(map_row_labels)

        map_col_labels = {"col_label": col_labels, "col": range(domain_size)}
        map_col_labels = pl.DataFrame(map_col_labels)
         
        ch_dist = coo_array(ch.dist)
        data = ch_dist.data
        rows, cols = ch_dist.coords
        
        mechanism_df = (
            pl.DataFrame({"row": rows, "col": cols, "p": data})
            .join(map_row_labels, on="row").drop("row")
            .join(map_col_labels, on="col").drop("col")
        )
        
        pred_join = pl.col(attr).list.len() == pl.col("col_label").list.len()
        subrecord_cols = preserve_attrs | {"col_label"}
        ch_metadata = (
            records_df
            # First get all possible sequences of attr values for each record
            .join_where(seq_domain, pred_join).rename({attr: "row_label"})
            # Then get the record id that matches each sequence
            .join(records_df.rename({attr: "col_label"}), on=subrecord_cols)
            .select("rid", "rid_right", "row_label", "col_label")
            .explode("row_label", "col_label")
            # Now get the probability for each pair of input-output,
            # and aggregate for each pair of input-output records:
            .join(mechanism_df, on=["row_label", "col_label"])
            .group_by("rid", "rid_right")
            .agg(pl.col("p").product())
        )

        data = ch_metadata["p"].to_numpy()
        rows = ch_metadata["rid"].to_numpy()
        cols = ch_metadata["rid_right"].to_numpy()

        shape = (n_records, n_records)
        ch_dist = coo_array((data, (rows, cols)), shape=shape)

        return ch_dist.tocsr()

    # Then we combine each individual channel, element-wise:
    ch_dist = _build_for(transform_attrs[0])
    for attr in transform_attrs[1:]: ch_dist *= _build_for(attr)
    
    return Channel(ch_dist)
