from __future__ import annotations
from dataclasses import dataclass

import polars as pl

def _is_valid(dist: pl.LazyDataFrame) -> bool:
    all_cols = set(dist.collect_schema().names())
    assert "p" in all_cols

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
        columns = set(self.dist.collect_schema().names())

        assert len(set(self.outcome) - columns) == 0
        assert "p" in columns

        if not _is_valid(self.dist):
            raise ValueError("Not a valid probability distribution!")


    def __repr__(self):
        """ Warning: might be expensive!"""
        sorted_dist = self.dist.sort(self.outcome)
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
