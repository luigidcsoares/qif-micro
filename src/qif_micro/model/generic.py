from collections.abc import Iterable

import polars as pl

from qif_micro import qif
from qif_micro.datatypes import Channel, Joint, ProbabDist
from qif_micro._internal.dataset import _valid_columns

def _hint_ch(
    domain_records: pl.DataFrame | pl.LazyFrame,
    domain_hints: pl.DataFrame | pl.LazyFrame,
) -> Channel:
    """
    This function expects the following shapes for each domain:

    Domain of records:
    - A single column named `record`, with type List
    - The inner type of `record` must be a struct

    Domain of hints:
    - A single column named `hint`, with type Struct
    - All fields of `hint` must be fields of the inner type of `record`

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model

    Consider the following example with attributes `q0`, `q1`, `s0`:

    >>> domain_q0 = pl.LazyFrame({ "q0": [0, 1, 2] })
    >>> domain_q1 = pl.LazyFrame({ "q1": [0, 1, 2] })
    >>> domain_s0 = pl.LazyFrame({ "s0": [0, 1, 2] })

    Let us assume that records may have one or two entries:
    
    >>> _ = domain_q0.join(domain_q1, how="cross")
    >>> _ = _.join(domain_s0, how="cross")
    >>> domain_hints = _.select(pl.struct("q0", "q1").alias("hint").unique())
    >>> domain_records1 = _.select(pl.concat_list(pl.struct("q0", "q1", "s0").alias("record").unique()))
    >>> domain_records2 = domain_records1.join(domain_records1, how="cross").select(pl.concat_list(pl.all()))
    >>> domain_records = pl.concat([domain_records1, domain_records2])

    Then the hint channel is
    
    >>> model.generic._hint_ch(domain_records, domain_hints)
    shape: (1_404, 3)
    ┌────────────────────┬───────────┬─────┐
    │ record             ┆ hint      ┆ p   │
    │ ---                ┆ ---       ┆ --- │
    │ list[struct[3]]    ┆ struct[2] ┆ f64 │
    ╞════════════════════╪═══════════╪═════╡
    │ [{0,0,0}]          ┆ {0,0}     ┆ 1.0 │
    │ [{0,0,0}, {0,0,0}] ┆ {0,0}     ┆ 1.0 │
    │ [{0,0,0}, {0,0,1}] ┆ {0,0}     ┆ 1.0 │
    │ [{0,0,0}, {0,0,2}] ┆ {0,0}     ┆ 1.0 │
    │ [{0,0,0}, {0,1,0}] ┆ {0,0}     ┆ 0.5 │
    │ …                  ┆ …         ┆ …   │
    │ [{2,1,2}, {2,2,2}] ┆ {2,1}     ┆ 0.5 │
    │ [{2,1,2}, {2,2,2}] ┆ {2,2}     ┆ 0.5 │
    │ [{2,2,0}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
    │ [{2,2,1}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
    │ [{2,2,2}, {2,2,2}] ┆ {2,2}     ┆ 1.0 │
    └────────────────────┴───────────┴─────┘
    """
    # Valid both domains first. We use assert bc this is an internal function.
    expected_cols = ["record"]
    diff, ok = _valid_columns(domain_records, expected_cols)
    assert ok, "Record domain missing `record`"

    record_schema = domain_records.collect_schema()["record"]
    assert record_schema == pl.List, "`record` dtype must be list"

    entry_schema = record_schema.inner
    assert entry_schema == pl.Struct, "`record` inner dtype must be struct"

    expected_cols = ["hint"]
    diff, ok = _valid_columns(domain_hints, expected_cols)
    assert ok, "Record domain missing `hint`"

    hint_schema = domain_hints.collect_schema()["hint"]
    assert hint_schema == pl.Struct, "`hint` dtype must be a struct"

    record_attrs = list(entry_schema.to_schema().keys())
    hint_attrs = list(hint_schema.to_schema().keys())

    diff = set(hint_attrs) - set(record_attrs)
    assert len(diff) == 0, "Mismatch between hint and record attributes"

    extract_single_expr = pl.struct(pl.element().struct.field(hint_attrs))
    extract_all_expr = pl.col("record").list.eval(extract_single_expr)

    _ = domain_records.with_columns(extract_all_expr.alias("hint"))
    _ = _.explode("hint").join(domain_hints, on="hint")
    _ = _.group_by("record", "hint").agg(pl.len().alias("p"))

    ch_dist = _.with_columns(pl.col("p") / pl.col("record").list.len())
    return Channel.from_polars(ch_dist, ["record"], ["hint"])


def _hyper_mechanism(
    prior: ProbabDist,
    mechanism: Channel
) -> [ProbabDist, Channel]:
    """
    This function takes a prior and a mechanism (e.g., geometric noise),
    and it returns a hyper-distribution as a pair:
    - The first component is the outer distribution on the outputs
    - The second component is the set of posterior distributions,
      which (mathematically) is essentially a channel

    ## Example
    >>> import polars as pl
    >>> from qif_micro import mechanism
    >>> from qif_micro import model

    >>> input_domain = [0, 1, 2]
    >>> output_domain = [0, 1, 2]
    >>> tg = mechanism.geometric.build(input_domain, output_domain, 1/3)

    >>> prior_dist = pl.LazyFrame({
    ...     "outcome": [0, 1, 2],
    ...     "p": [1/4, 1/2, 1/4],
    ... })
    >>> prior = ProbabDist.from_polars(prior_dist, ["outcome"])

    >>> outer, posteriors = model.generic._hyper_mechanism(prior, tg)
    >>> outer
    shape: (3, 2)
    ┌────────┬──────────┐
    │ output ┆ p        │
    │ ---    ┆ ---      │
    │ i64    ┆ f64      │
    ╞════════╪══════════╡
    │ 0      ┆ 0.333333 │
    │ 1      ┆ 0.333333 │
    │ 2      ┆ 0.333333 │
    └────────┴──────────┘
    >>> posteriors
    shape: (9, 3)
    ┌────────┬───────┬────────┐
    │ output ┆ input ┆ p      │
    │ ---    ┆ ---   ┆ ---    │
    │ i64    ┆ i64   ┆ f64    │
    ╞════════╪═══════╪════════╡
    │ 0      ┆ 0     ┆ 0.5625 │
    │ 0      ┆ 1     ┆ 0.375  │
    │ 0      ┆ 2     ┆ 0.0625 │
    │ 1      ┆ 0     ┆ 0.125  │
    │ 1      ┆ 1     ┆ 0.75   │
    │ 1      ┆ 2     ┆ 0.125  │
    │ 2      ┆ 0     ┆ 0.0625 │
    │ 2      ┆ 1     ┆ 0.375  │
    │ 2      ┆ 2     ┆ 0.5625 │
    └────────┴───────┴────────┘
    """
    joint = qif.push(prior, mechanism)

    p_expr = pl.col("p").sum()
    outer_dist = joint.dist.group_by(joint.output).agg(p_expr)
    outer = ProbabDist.from_polars(outer_dist, joint.output)

    p_expr = (pl.col("p") / pl.col("p_right")).alias("p")
    post_dists = joint.dist.join(outer_dist, on=joint.output).select(
        *joint.output, *joint.input, p_expr
    )

    posts = Channel.from_polars(post_dists, joint.output, joint.input)
    return outer, posts


def _strategies(
    prior: ProbabDist,
    mechanism: Channel,
    dataset: pl.DataFrame | pl.LazyFrame,
    gain_fn: pl.DataFrame | pl.LazyFrame,
    hint_attrs: Iterable[str]
) -> pl.LazyFrame:
    """
    TODO: Assume gain fn has "record" as column? Create a wrapper for gain fn?
    """
    # This assumes that dataset is in the required format:
    # a single column "record" as a list of structs.
    expected_cols = ["record"]
    diff, ok = _valid_columns(dataset, expected_cols)
    assert ok, f"Missing columns {diff}"

    # We first compute the sum of the posterior probabilities
    # given each record in the sanitised dataset.
    dataset = dataset.rename({"record": mechanism.output[0]})

    _, posts_mechanism = _hyper_mechanism(prior, mechanism)
    _ = posts_mechanism.dist.join(dataset, on=mechanism.output)
    _ = _.group_by(mechanism.input).agg(pl.col("p").sum())
    agg_posteriors = _.rename({mechanism.input[0]: hint_ch.input[0]})

    # Then we get the (sub)domain of records from the prior,
    # and also the (sub)domain of hints, to build the hint channel:
    record_expr = pl.col(prior.outcome).unique().alias("record")
    domain_records = prior.dist.select(record_expr)

    extract_expr = pl.col("record").struct.field(hint_attrs)
    _ = domain_records.with_row_index("rid")
    _ = _.explode("record")
    _ = _.select(pl.struct(extract_expr).alias("hint"))
    domain_hints = _.unique()

    # We then join each record (input) on the hint channel, gain fn an agg posteriors,
    # and sum over records, to compute the expected gain for each hint and action:
    combine_p_expr = (pl.col("p") * pl.col("p_right")).alias("p")
    _ = hint_ch.dist.join(agg_posteriors, on=hint_ch.input)
    _ = _.with_columns(combine_p_expr).drop("p_right")

    # TODO: `record` and `gain` are hardcoded for now, we should change this
    combine_gain_expr = (pl.col("p") * pl.col("gain")).alias("gain")
    _ = _.join(gain_fn, left_on=hint_ch.input, right_on="record")
    _ = _.with_columns(combine_gain_expr).drop("p")
    _ = _.group_by(*hint_ch.output, "action")
    expected_gain = _.agg(pl.col("gain").sum())

    # Finally, we compute the argmax (as a set) for each hint:
    max_expr = pl.col("gain").max().over(hint_ch.output).alias("max")
    filter_expr = pl.col("gain") == pl.col("max")
    argmax = expected_gain.with_columns(max_expr).filter(filter_expr)

    return argmax.select(*hint_ch.output, "action")
