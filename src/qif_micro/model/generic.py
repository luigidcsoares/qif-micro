# from collections.abc import Iterable

import polars as pl

from qif_micro import qif
from qif_micro.datatypes import channel, Channel, LazyChannel
from qif_micro.datatypes import probab_dist, ProbabDist, LazyProbabDist
from qif_micro._internal.dataset import _valid_columns

def _hyper_mechanism(
    prior: ProbabDist | LazyProbabDist,
    mechanism: Channel | LazyChannel
) -> [LazyProbabDist, LazyChannel]:
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
    >>> from qif_micro.datatypes import channel
    >>> from qif_micro.datatypes import probab_dist

    >>> input_domain = [0, 1, 2]
    >>> output_domain = [0, 1, 2]
    >>> tg = mechanism.geometric(input_domain, output_domain, 1/3)

    >>> prior_dist = pl.LazyFrame({
    ...     "secret": [0, 1, 2],
    ...     "p": [1/4, 1/2, 1/4],
    ... })
    >>> prior = probab_dist.make_lazy(prior_dist, ["secret"])

    >>> outer, posteriors = model.generic._hyper_mechanism(prior, tg)
    >>> probab_dist.collect(outer)
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
    >>> channel.collect(posteriors)
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
    outer = probab_dist.make_lazy(outer_dist, joint.output)

    p_expr = (pl.col("p") / pl.col("p_right")).alias("p")
    _ = joint.dist.join(outer_dist, on=joint.output)
    post_dists = _.select(*joint.output, *joint.secret, p_expr)

    posts = channel.make_lazy(post_dists, joint.output, joint.secret)
    return outer, posts


def from_dataset(
    dataset: pl.DataFrame | pl.LazyFrame,
    prior: ProbabDist | LazyProbabDist,
    mechanism: Channel | LazyChannel
) -> LazyProbabDist:
    """
    This functioni takes a (possibly sanitised) dataset,
    a prior knowledge on records and the mechanism from records to records
    that was used to generate the dataset (it could be an identity mechanism).

    It then returns the adversary's intermediate knowledge on records,
    conditioned on the observation of the dataset.

    ## Example
    
    >>> import polars as pl
    >>> from qif_micro import mechanism
    >>> from qif_micro import model
    >>> from qif_micro.datatypes import channel
    >>> from qif_micro.datatypes import probab_dist

    Consider the following sanitised dataset:
    >>> dataset = pl.LazyFrame({
    ...     "record_id": [0, 0, 1, 1, 2, 2],
    ...     "qid": [0, 0, 0, 1, 0, 0],
    ...     "sens": [0, 0, 0, 0, 1, 1]
    ... })

    Now, suppose that the domain of qid and sens is the same: {0, 1, 2},
    and suppose that it was applied geometric noise to each attribute:

    >>> input_domain = [0, 1, 2]
    >>> output_domain = [0, 1, 2]
    >>> tg = mechanism.geometric(input_domain, output_domain, 1/3)
    
    The domain of records that could have been mapped to the records
    observed in the sanitised dataset can be computed as follows:

    >>> domain_q = pl.LazyFrame({ "qid": [0, 1, 2] })
    >>> domain_s = pl.LazyFrame({ "sens": [0, 1, 2] })
    >>> unique_record_expr = pl.struct("qid", "sens").alias("record").unique()
    >>> _ = domain_q.join(domain_q, how="cross")
    >>> _ = _.join(domain_s, how="cross")
    >>> _ = _.select(pl.concat_list(unique_record_expr))
    >>> domain_records = _.join(_, how="cross").select(pl.concat_list(pl.all()))

    Thus, the mechanism from records to records is
    >>> m_records = mechanism.record(
    ...     domain_records,
    ...     ("qid", [0, 1, 2], tg),
    ...     ("sens", [0, 1, 2], tg)
    ... )

    Consider a uniform prior over the records:

    >>> p_expr = (1 / pl.len()).alias("p")
    >>> prior_dist = domain_records.with_columns(p_expr)
    >>> prior = probab_dist.make_lazy(prior_dist, ["record"])

    Then, upon observing the sanitised dataset, the adversary's knowledge is

    >>> mid_knowledge = model.generic.from_dataset(dataset, prior, m_records)
    >>> probab_dist.collect(mid_knowledge)
    shape: (81, 2)
    ┌─────────────────┬──────────┐
    │ input           ┆ p        │
    │ ---             ┆ ---      │
    │ list[struct[2]] ┆ f64      │
    ╞═════════════════╪══════════╡
    │ [{0,0}, {0,0}]  ┆ 0.105085 │
    │ [{0,0}, {0,1}]  ┆ 0.05207  │
    │ [{0,0}, {0,2}]  ┆ 0.017357 │
    │ [{0,0}, {1,0}]  ┆ 0.094018 │
    │ [{0,0}, {1,1}]  ┆ 0.03702  │
    │ …               ┆ …        │
    │ [{1,1}, {2,2}]  ┆ 0.001088 │
    │ [{1,2}, {2,2}]  ┆ 0.000363 │
    │ [{2,0}, {2,2}]  ┆ 0.000457 │
    │ [{2,1}, {2,2}]  ┆ 0.000363 │
    │ [{2,2}, {2,2}]  ┆ 0.000121 │
    └─────────────────┴──────────┘
    """
    # ==================================================
    # Pre-conditions: dataset must either be in "long"
    #   format with rows tagged with a record_id,
    #   or must have a single record column.
    # ==================================================
     
    dataset = dataset.lazy()
    diff_record, ok_record = _valid_columns(dataset, ["record"])
    diff_id, ok_id = _valid_columns(dataset, ["record_id"])
    if not (ok_record or ok_id):
        msg_record = "Dataset must either have a single `record` column"
        msg_id = "have a column `record_id` tagged to every entry"
        raise ValueError(f"{msg_record} or {msg_id}")

    schema = dataset.collect_schema()
    if ok_record:
        ok_record_type = schema["record"] == pl.List
        if not ok_record_type:
            raise ValueError("`record` dtype must be list")

        ok_record_inner = schema["record"].inner == pl.Struct
        if not ok_record_inner:
            raise ValueError("`record` inner dtype must be struct")

    else: # Transform dataset into long form
        record_attrs = [c for c in schema.keys() if c != "record_id"]
        record_expr = pl.struct(record_attrs).alias("record")
        _ = dataset.select("record_id", record_expr)
        _= _.group_by("record_id")
        dataset = _.agg(pl.col("record"))

    # ==================================================
    # Finished pre-conditions
    # ==================================================
    
    n_records_expr = pl.len().alias("n_records")
    record_expr = pl.col("record").alias(mechanism.output[0])
    dataset = dataset.select(record_expr, n_records_expr)
    
    p_expr = (pl.col("p").sum() / pl.col("n_records").first()).alias("p")
    _, posts_mechanism = _hyper_mechanism(prior, mechanism)
    _ = posts_mechanism.dist.join(dataset, on=mechanism.output)
    intermediate_dist = _.group_by(mechanism.secret).agg(p_expr)

    return probab_dist.make_lazy(intermediate_dist, mechanism.secret)
