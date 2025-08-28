import math

import polars

from qif_micro.typing import Channel, ProbabDist
from qif_micro.typing import FieldName, FieldValue

def build(
    dataset: polars.DataFrame,
    owner_field: FieldName,
    qid_field: FieldName,
    qid_domain: list[FieldValue],
    sensitive_field: FieldName,
    p_keep: float,
) -> tuple[ProbabDist, Channel]:
    """
    The input to this function is a dataset that is result of
    applying random response to the qid_field of the original data. 

    We assume that random response is implemented as follows:
    - The probability of preserving the qid value is p_keep
    - The probability of remapping the qid value to another value in
      qid_domain is (1 - p_keep)/(|qid_domain| - 1)

    This function then returns the adversary's intermediate knowledge
    (i.e., after the dataset has been observed) and the channel that
    models the relation between the sensitive_field and the qid_field.

    That is, this function models an attribute-inference attack, in
    which the adversary's goal is to guess the sensitive value of a target.

    Limitations:
    - We require that the the length of records is one; that is, each
      individual owns a single row of the dataset
    - We accept only a single field as QID and a single field as sensitive

    ## Example

    >>> import polars
    >>> from qif_micro import model
    >>> dataset = polars.DataFrame({
    ...     "uid": [0, 1, 2, 3, 4],
    ...     "grade": ["A", "B", "B", "B", "A"],
    ...     "disability": ["yes", "no", "yes", "yes", "no"]
    ... })
    >>> owner_field = "uid"
    >>> qid_field = "grade"
    >>> qid_domain = ["A", "B", "C"]
    >>> sensitive_field = "disability"
    >>> p_keep = 3/4
    >>> prior, channel = model.random_response.build(
    ...     dataset,
    ...     owner_field,
    ...     qid_field,
    ...     qid_domain,
    ...     sensitive_field,
    ...     p_keep
    ... )
    >>> prior.sort(by=sensitive_field).collect()
    shape: (2, 2)
    ┌────────────┬─────┐
    │ disability ┆ p   │
    │ ---        ┆ --- │
    │ str        ┆ f64 │
    ╞════════════╪═════╡
    │ no         ┆ 0.4 │
    │ yes        ┆ 0.6 │
    └────────────┴─────┘
    >>> channel.sort(by=[sensitive_field, "qid"]).collect()
    shape: (6, 3)
    ┌────────────┬─────┬──────────┐
    │ disability ┆ qid ┆ p        │
    │ ---        ┆ --- ┆ ---      │
    │ str        ┆ str ┆ f64      │
    ╞════════════╪═════╪══════════╡
    │ no         ┆ A   ┆ 0.4375   │
    │ no         ┆ B   ┆ 0.4375   │
    │ no         ┆ C   ┆ 0.125    │
    │ yes        ┆ A   ┆ 0.333333 │
    │ yes        ┆ B   ┆ 0.541667 │
    │ yes        ┆ C   ┆ 0.125    │
    └────────────┴─────┴──────────┘
    """
    assert owner_field in dataset.columns

    n_records = dataset.get_column(owner_field).n_unique()
    assert n_records == dataset.height

    expr_match = p_keep / n_records
    expr_conflict = (1 - p_keep) / (n_records *(len(qid_domain) - 1))

    # We assume the dataset fits in memory, but the prior and channel
    # could be really large, so we from this point we rely on laziness
    joint = dataset.lazy().with_columns(qid=qid_domain).explode("qid")
    joint = joint.select(
        sensitive_field, "qid",
        p=polars
        .when(polars.col(qid_field) == polars.col("qid"))
        .then(expr_match)
        .otherwise(expr_conflict)
    )

    joint_sensitive = joint.group_by(sensitive_field, "qid").sum()

    prior = (
        joint_sensitive
        .group_by(sensitive_field)
        .agg(p=polars.col("p").sum())
    )

    prior_sum = prior.select(polars.col("p").sum()).collect()
    assert math.isclose(prior_sum.item(), 1.0)

    channel = (
        joint_sensitive
        .join(prior, on=sensitive_field, how="inner")
        .with_columns(p=polars.col("p") / polars.col("p_right"))
        .drop("p_right")
    )

    # ---
    # To transform the lazy channel into matrix-like format, do:
    # ---
    # _channel = channel.drop(count_field, sum_field).collect()
    # _channel.pivot(on="qid", index=[count_field, sum_field], values="p")

    channel_sum = channel.select(polars.col("p").sum()).collect()
    expected_sum = prior.select(polars.col("p").len()).collect()
    assert math.isclose(channel_sum.item(), expected_sum.item())

    return prior, channel
