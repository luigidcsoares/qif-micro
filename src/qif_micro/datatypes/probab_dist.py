from __future__ import annotations
from dataclasses import dataclass

import polars as pl

from qif_micro._internal.dataset import _valid_columns

def _is_valid(dist: pl.LazyDataFrame) -> bool:
    expr_check_sum = pl.col("p").sum().is_close(1, abs_tol=0.005)
    return dist.select(expr_check_sum).collect().item()


@dataclass(frozen=True)
class ProbabDist:
    """
    A ProbabDist is an ensemble (x, Ax, Px), 
    where x is the outcome of a random variable X,
    Ax is the alphabet, with x in Ax, and Px = {p_x0, p_x1, ...}.
    """
    dist: pl.LazyDataFrame
    outcome: list[str]

    def __post_init__(self):
        expected_cols = ["p", *self.outcome]
        _, ok = _valid_columns(self.dist, expected_cols)
        assert ok

        if not _is_valid(self.dist):
            raise ValueError("Not a valid probability distribution!")


    def __repr__(self):
        """ Warning: might be expensive!"""
        schema = self.dist.collect_schema()
        sorted_dist = self.dist.sort(self.outcome)

        for col in self.outcome:
            dtype = schema[col]
            if isinstance(dtype, pl.List):
                sort_within_expr = pl.col(col).list.sort()
                sorted_dist = sorted_dist.with_columns(sort_within_expr)

        return sorted_dist.collect().__repr__()

    
    @classmethod
    def from_polars(
        cls,
        dist: pl.LazyDataFrame,
        outcome_cols: list[str],
        probab_col: str = "p"
    ) -> ProbabDist:
        """
        Factory method: expects a LazyDataFrame with two columns:
        - `outcome_cols`: names of columns with the possible outcomes 
        - `probab_col`: name of column containing p(outcome)
        """
        dist = dist.rename({probab_col: "p"})
        return cls(dist, outcome_cols)
