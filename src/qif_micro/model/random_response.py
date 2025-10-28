from typing import Any

import polars

from qif_micro import qif
from qif_micro.datatypes import Channel, Joint, ProbabDist

def build(
    dataset: polars.DataFrame,
    owner_attr: str,
    qid_attr: str,
    qid_domain: list[Any],
    sensitive_attr: str,
    p_keep: float,
) -> tuple[ProbabDist, Channel]:
    """
    The input to this function is a dataset that is result of
    applying random response to the qid_attr of the original data. 

    We assume that random response is implemented as follows:
    - The probability of preserving the qid value is p_keep
    - The probability of remapping the qid value to another value in
      qid_domain is (1 - p_keep)/(|qid_domain| - 1)

    This function then returns the adversary's intermediate knowledge
    (i.e., after the dataset has been observed) and the channel that
    models the relation between the sensitive_attr and the qid_attr.

    That is, this function models an attribute-inference attack, in
    which the adversary's goal is to guess the sensitive value of a target.

    Limitations:
    - We require that the the length of records is one; that is, each
      individual owns a single row of the dataset
    - We accept only a single attr as QID and a single attr as sensitive

    ## Example

    >>> import polars
    >>> from qif_micro import model
    >>> dataset = polars.DataFrame({
    ...     "uid": [0, 1, 2, 3, 4],
    ...     "grade": ["A", "B", "B", "B", "A"],
    ...     "disability": ["yes", "no", "yes", "yes", "no"]
    ... })
    >>> owner_attr = "uid"
    >>> qid_attr = "grade"
    >>> qid_domain = ["A", "B", "C"]
    >>> sensitive_attr = "disability"
    >>> p_keep = 3/4
    >>> prior, channel = model.random_response.build(
    ...     dataset,
    ...     owner_attr,
    ...     qid_attr,
    ...     qid_domain,
    ...     sensitive_attr,
    ...     p_keep
    ... )
    >>> prior.dist.sort(by=sensitive_attr).collect()
    shape: (2, 2)
    ┌────────────┬─────┐
    │ disability ┆ p   │
    │ ---        ┆ --- │
    │ str        ┆ f64 │
    ╞════════════╪═════╡
    │ no         ┆ 0.4 │
    │ yes        ┆ 0.6 │
    └────────────┴─────┘
    >>> channel.dist.sort(by=[sensitive_attr, "qid"]).collect()
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
    assert owner_attr in dataset.columns

    n_records = dataset.get_column(owner_attr).n_unique()
    assert n_records == dataset.height

    expr_match = p_keep / n_records
    expr_conflict = (1 - p_keep) / (n_records *(len(qid_domain) - 1))

    # We assume the dataset fits in memory, but the prior and channel
    # could be really large, so from this point on we rely on laziness
    joint_dist_records = (
        dataset.lazy()
        .with_columns(qid=qid_domain)
        .explode("qid")
        .select(
            sensitive_attr, "qid",
            p=polars
            .when(polars.col(qid_attr) == polars.col("qid"))
            .then(expr_match)
            .otherwise(expr_conflict)
        )
    )

    joint_dist_sens = (
        joint_dist_records
        .group_by(sensitive_attr, "qid")
        .sum()
    )

    joint = Joint.from_polars(joint_dist_sens, [sensitive_attr], ["qid"])
    prior, ch = qif.push_back(joint)

    return prior, ch
