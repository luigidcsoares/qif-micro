from typing import Any

import polars as pl

from qif_micro import qif
from qif_micro.datatypes import Channel, Joint, ProbabDist

def build(
    dataset: pl.DataFrame,
    owner_attr: str,
    hint_attr: str,
    hint_domain: list[Any],
    sens_attr: str,
    p_keep: float,
) -> tuple[ProbabDist, Channel]:
    """
    The input to this function is a dataset that is result of
    applying random response to the hint_attr of the original data. 

    We assume that random response is implemented as follows:
    - The probability of preserving the hint value is p_keep
    - The probability of remapping the hint value to another value in
      hint_domain is (1 - p_keep)/(|hint_domain| - 1)

    This function then returns the adversary's intermediate knowledge
    (i.e., after the dataset has been observed) and the channel that
    models the relation between the sens_attr and the hint_attr.

    That is, this function models an attribute-inference attack, in
    which the adversary's goal is to guess a target's sensitive value.

    Limitations:
    - We require that the the length of records is one; that is, each
      individual owns a single row of the dataset
    - We accept only a single attr as QID and a single attr as sensitive

    ## Example

    >>> import polars as pl
    >>> from qif_micro import model
    >>> dataset = pl.DataFrame({
    ...     "uid": [0, 1, 2, 3, 4],
    ...     "grade": ["A", "B", "B", "B", "A"],
    ...     "disability": ["yes", "no", "yes", "yes", "no"]
    ... })
    >>> owner_attr = "uid"
    >>> hint_attr = "grade"
    >>> hint_domain = ["A", "B", "C"]
    >>> sens_attr = "disability"
    >>> p_keep = 3/4
    >>> prior, ch = model.random_response.build(
    ...     dataset,
    ...     owner_attr,
    ...     hint_attr,
    ...     hint_domain,
    ...     sens_attr,
    ...     p_keep
    ... )
    >>> prior
    shape: (2, 2)
    ┌────────────┬─────┐
    │ disability ┆ p   │
    │ ---        ┆ --- │
    │ str        ┆ f64 │
    ╞════════════╪═════╡
    │ no         ┆ 0.4 │
    │ yes        ┆ 0.6 │
    └────────────┴─────┘
    >>> ch
    shape: (6, 3)
    ┌────────────┬──────┬──────────┐
    │ disability ┆ hint ┆ p        │
    │ ---        ┆ ---  ┆ ---      │
    │ str        ┆ str  ┆ f64      │
    ╞════════════╪══════╪══════════╡
    │ no         ┆ A    ┆ 0.4375   │
    │ no         ┆ B    ┆ 0.4375   │
    │ no         ┆ C    ┆ 0.125    │
    │ yes        ┆ A    ┆ 0.333333 │
    │ yes        ┆ B    ┆ 0.541667 │
    │ yes        ┆ C    ┆ 0.125    │
    └────────────┴──────┴──────────┘
    """
    assert owner_attr in dataset.columns

    n_records = dataset.get_column(owner_attr).n_unique()
    assert n_records == dataset.height

    expr_match = p_keep / n_records
    expr_conflict = (1 - p_keep) / (n_records *(len(hint_domain) - 1))

    # We assume the dataset fits in memory, but the prior and channel
    # could be really large, so from this point on we rely on laziness
    joint_dist_records = (
        dataset
        .lazy()
        .with_columns(pl.lit(hint_domain).alias("hint"))
        .explode("hint")
        .select(
            sens_attr, "hint",
            pl
            .when(pl.col(hint_attr) == pl.col("hint"))
            .then(expr_match)
            .otherwise(expr_conflict)
            .alias("p")
        )
    )

    joint_dist_sens = joint_dist_records.group_by(sens_attr, "hint").sum()
    joint = Joint.from_polars(joint_dist_sens, [sens_attr], ["hint"])

    return qif.push_back(joint)
