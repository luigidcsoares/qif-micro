from collections.abc import Iterable

import polars as pl

from qif_micro.datatypes import channel, Channel, LazyChannel
from qif_micro._internal.dataset import _valid_columns

def build(
    domain_records: pl.DataFrame | pl.LazyFrame,
    mechanism: tuple[str, Iterable[str], Channel | LazyChannel],
    *mechanisms: tuple[str, Iterable[str], Channel | LazyChannel]
) -> LazyChannel:
    """
    This function produces a mechanism from records to records,
    given one or more attribut-level mechanisms.

    This expects `domain_records` to be a frame with a single column
    "record", which must be a list of structs (list of maps).

    The domain of records should contain all records that are possible
    according to the adversary's knowledge, but it need not
    contain all records in the actual domain. In the case that the
    adversary's knowledge excludes some records (assigns probability zero),
    this function returns a slice of the channel modelling the mechanism.

    ## Example
    >>> import polars as pl
    >>> from qif_micro import mechanism
    >>> from qif_micro.datatypes import channel

    We first construct an attribute-level mechanism:
    
    >>> input_domain = [0, 1, 2]
    >>> output_domain = [0, 1, 2]
    >>> tg = mechanism.geometric(input_domain, output_domain, 1/3)

    Consider the following (sub)domain of records:
    >>> domain_records = pl.LazyFrame({
    ...     "record_id": [0, 0, 1],
    ...     "q0": [0, 1, 1],
    ...     "q1": [0, 0, 1],
    ...     "s0": [1, 1, 0]
    ... })

    Suppose we are applying mechanism `tg` to both q0 and s0:
    >>> ch = mechanism.record(domain_records, ("q0", [0, 1, 2], tg), ("s0", [0, 1, 2], tg))
    >>> channel.collect(ch)
    shape: (90, 3)
    ┌────────────────────┬────────────────────┬───────────┐
    │ input              ┆ output             ┆ p         │
    │ ---                ┆ ---                ┆ ---       │
    │ list[struct[3]]    ┆ list[struct[3]]    ┆ f64       │
    ╞════════════════════╪════════════════════╪═══════════╡
    │ [{0,0,1}, {1,0,1}] ┆ [{0,0,0}, {0,0,0}] ┆ 0.011719  │
    │ [{0,0,1}, {1,0,1}] ┆ [{0,0,0}, {0,0,1}] ┆ 0.0234375 │
    │ [{0,0,1}, {1,0,1}] ┆ [{0,0,0}, {0,0,2}] ┆ 0.011719  │
    │ [{0,0,1}, {1,0,1}] ┆ [{0,0,0}, {1,0,0}] ┆ 0.0234375 │
    │ [{0,0,1}, {1,0,1}] ┆ [{0,0,0}, {1,0,1}] ┆ 0.046875  │
    │ …                  ┆ …                  ┆ …         │
    │ [{1,1,0}]          ┆ [{1,1,1}]          ┆ 0.083333  │
    │ [{1,1,0}]          ┆ [{1,1,2}]          ┆ 0.041667  │
    │ [{1,1,0}]          ┆ [{2,1,0}]          ┆ 0.1875    │
    │ [{1,1,0}]          ┆ [{2,1,1}]          ┆ 0.041667  │
    │ [{1,1,0}]          ┆ [{2,1,2}]          ┆ 0.020833  │
    └────────────────────┴────────────────────┴───────────┘
    """
    # ==================================================
    # Pre-conditions: records must either be in "long"
    #   format with rows tagged with a record_id,
    #   or must have a single record column.
    # ==================================================
    domain_records = domain_records.lazy()
    
    diff_record, ok_record = _valid_columns(domain_records, ["record"])
    diff_id, ok_id = _valid_columns(domain_records, ["record_id"])
    if not (ok_record or ok_id):
        msg_record = "Domain of records must either have a single `record` column"
        msg_id = "have a column `record_id` tagged to every entry"
        raise ValueError(f"{msg_record} or {msg_id}")

    schema = domain_records.collect_schema()
    if ok_record:
        ok_record_type = schema["record"] == pl.List
        if not ok_record_type:
            raise ValueError("`record` dtype must be list")

        ok_record_inner = schema["record"].inner == pl.Struct
        if not ok_record_inner:
            raise ValueError("`record` inner dtype must be struct")

    else: # Transform domain of records into long form
        record_attrs = [c for c in schema.keys() if c != "record_id"]
        record_expr = pl.struct(record_attrs).alias("record")
        _ = domain_records.select("record_id", record_expr)
        _ = _.group_by("record_id").agg(pl.col("record"))
        domain_records = _.drop("record_id")

    # ==================================================
    # Finished pre-conditions
    # ==================================================

    mechanisms = [mechanism, *mechanisms]
    attrs, attr_domains, mechanisms = list(zip(*mechanisms))
    input_domain = domain_records.rename({"record": "input"})

    # Now we construct the output domain,
    # varying attributes according to each mechanism
    domain_records = domain_records.with_row_index("rid")

    _ = domain_records.select(pl.col("record").list.len().max())
    max_record_len = _.collect().item()

    fields = [f"{i}" for i in range(max_record_len)]
    def _expand_attr(lf, attr):
        repeat_domain_expr = pl.col("record").list.eval(pl.lit(domain))
        to_struct_expr = repeat_domain_expr.list.to_struct(fields=fields)
        unnest_expr = to_struct_expr.struct.unnest()

        lf = lf.with_columns(unnest_expr)
        for f in fields: lf = lf.explode(f, keep_nulls=True)

        zip_expr = pl.concat_list(fields).list.drop_nulls().alias("var")
        lf = lf.with_columns(zip_expr).drop("rid", *fields)
        return lf.with_row_index("rid")
    
    _ = domain_records.collect_schema()["record"]
    all_attrs = _.inner.to_schema().keys()

    _ = domain_records
    for attr, domain in zip(attrs, attr_domains): 
        _ = _expand_attr(_, domain)
        _ = _.explode(pl.exclude("rid"))
        
        _ = _.select("rid", pl.col("record").struct.unnest(), "var")
        _ = _.drop(attr).rename({"var": attr})
        
        _ = _.select("rid", pl.struct(all_attrs).alias("record"))
        _ = _.group_by("rid").agg(pl.col("record"))
        _ = _.unique(pl.exclude("rid"))
    output_domain = _.drop("rid").rename({"record": "output"})

    # Next step is to obtain the possible input, output pairs,
    # noting that records can only be remapped to records of same length
    def _extract_attr_from(col, attrs):
        extract_expr = pl.element().struct.field(attrs)
        return pl.col(col).list.eval(extract_expr)
        
    keep_attrs = [attr for attr in all_attrs if attr not in attrs]
    if len(keep_attrs) > 0:
        keep_input_expr = _extract_attr_from("input", keep_attrs).alias("keep_input")
        keep_output_expr = _extract_attr_from("output", keep_attrs).alias("keep_output")
        input_domain = input_domain.with_columns(keep_input_expr)
        output_domain = output_domain.with_columns(keep_output_expr)
        pred_join_keep = pl.col("keep_input") == pl.col("keep_output")
    else:
        pred_join_keep = pl.lit(True)
        
    pred_join_len = pl.col("input").list.len() == pl.col("output").list.len()
    pred_join = pred_join_keep & pred_join_len
    cross_domain = input_domain.join_where(output_domain, pred_join)

    def _step_p(lf, attr, mechanism):
        lf = lf.with_columns(
            _extract_attr_from("input", attr).alias("input_attr"),
            _extract_attr_from("output", attr).alias("output_attr"),
        )

        m_dist = mechanism.dist.select(
            pl.col(mechanism.secret).alias("input_attr"),
            pl.col(mechanism.output).alias("output_attr"),
            "p"
        )

        p_expr = pl.col("p").product()
        p_step_expr = pl.col("p_step").first()
        combine_p_expr = (p_step_expr * p_expr).alias("p_step")

        lf = lf.explode("input_attr", "output_attr")
        lf = lf.join(m_dist, on=["input_attr", "output_attr"])
        return lf.group_by("input", "output").agg(combine_p_expr)
    
    _ = cross_domain.with_columns(pl.lit(1).alias("p_step"))
    for attr, m in zip(attrs, mechanisms): _ = _step_p(_, attr, m)

    m_dist = _.rename({"p_step": "p"})
    return channel.make_lazy(m_dist, ["input"], ["output"])
