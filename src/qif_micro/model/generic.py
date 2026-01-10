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
    - Each row of the frame corresponds to one entry of a record
    - Each row must be tagged with a record id
    - Each row contains multiple columns, one for each attribute

    Domain of hints:
    - Each row of the frame corresponds to a single hint
    - Each row contains multiple columns, one for each attribute
    - A hint is a tuple (a conjunction of all attributes in a row)

    ## Example
    >>> import polars as pl
    >>> from qif_micro import model
    >>> domain_records = pl.LazyFrame({
    ...     "record_id": [0, 0, 1],
    ...     "q0": [0, 1, 1],
    ...     "q1": [0, 0, 1],
    ...     "s0": [1, 1, 0]
    ... })
    >>> domain_hints = pl.LazyFrame({
    ...     "q0": [0, 0, 1, 1],
    ...     "q1": [0, 1, 0, 1],
    ... })
    >>> model.generic._hint_ch(domain_records, domain_hints)
    shape: (3, 3)
    ┌────────────────────┬───────────┬─────┐
    │ record             ┆ hint      ┆ p   │
    │ ---                ┆ ---       ┆ --- │
    │ list[struct[3]]    ┆ struct[2] ┆ f64 │
    ╞════════════════════╪═══════════╪═════╡
    │ [{0,0,1}, {1,0,1}] ┆ {0,0}     ┆ 0.5 │
    │ [{0,0,1}, {1,0,1}] ┆ {1,0}     ┆ 0.5 │
    │ [{1,1,0}]          ┆ {1,1}     ┆ 1.0 │
    └────────────────────┴───────────┴─────┘
    """
    domain_records = domain_records.lazy().unique()
    domain_hints = domain_hints.lazy().unique()

    # Valid both domains first. We use assert bc this is an internal function.
    expected_cols = ["record_id"]
    diff, ok = _valid_columns(domain_records, expected_cols)
    assert ok, "Record domain missing `record_id`"

    hint_schema = domain_hints.collect_schema()
    expected_cols = hint_schema.names() 
    diff, ok = _valid_columns(domain_records, expected_cols)
    assert ok, "Mismatch between hint and record attributes"

    record_schema = domain_records.collect_schema()
    record_attrs = [c for c in record_schema.names() if c != "record_id"]
    hint_attrs = hint_schema.names()

    _ = domain_records.join(domain_hints, on=hint_attrs)
    _ = _.with_columns(
       pl.len().over("record_id").alias("record_len"),
       pl.len().over("record_id", *hint_attrs).alias("freq")
    )
    _ = _.with_columns((pl.col("freq") / pl.col("record_len")).alias("p"))

    ch_dist = _.select(
        pl.struct(record_attrs).implode().over("record_id").alias("record"),
        pl.struct(hint_attrs).alias("hint"),
        "p"
    )

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
    dataset: pl.DataFrame | pl.LazyFrame,
    gain_fn: pl.DataFrame | pl.LazyFrame,
    prior_records: ProbabDist,
    mechanism: Channel,
) -> Channel:
    """
    TODO
    """
    # This assumes that dataset is in the required format:
    # a single column "record" as a list of structs.
    expected_cols = ["record"]
    diff, ok = _valid_columns(dataset, expected_cols)
    assert ok, f"Missing columns {diff}"

    _, posts_mechanism = _hyper_mechanism(prior_records, mechanism)

    # We first compute the sum of the posterior probabilities
    # given each record in the sanitised dataset.
    _ = posts_mechanism.join(
        dataset, left_on="record", right_on=mechanism.input
    )

    agg_posteriors = _.group_by(mechanism.input).agg(pl.col("p").sum())
