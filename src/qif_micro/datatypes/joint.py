from __future__ import annotations
from dataclasses import dataclass

import polars as pl

from qif_micro._internal.dataset import _valid_columns

def _is_valid(dist: pl.LazyDataFrame) -> bool:
    expr_check_sum = pl.col("p").sum().is_close(1, abs_tol=0.005)
    return dist.select(expr_check_sum).collect().item()


@dataclass(frozen=True)
class Joint:
    """
    A joint models the correlation between the input and a output
    of a probabilistic function. That is, given an input x and an 
    output y, the entry x,y of a joint is the joint probability p(x, y).
    """
    dist: pl.LazyDataFrame
    input: list[str]
    output: list[str]

    def __post_init__(self):
        expected_cols = ["p", *self.input, *self.output]
        _, ok = _valid_columns(self.dist, expected_cols)
        assert ok

        if not _is_valid(self.dist):
            raise ValueError("Not a valid joint distribution!")


    def __repr__(self):
        """ Warning: might be expensive!"""
        schema = self.dist.collect_schema()
        sorted_dist = self.dist.sort(self.input + self.output)

        for col in self.input + self.output:
            dtype = schema[col]
            if isinstance(dtype, pl.List):
                sort_within_expr = pl.col(col).list.sort()
                sorted_dist = sorted_dist.with_columns(sort_within_expr)

        return sorted_dist.collect().__repr__()

    
    @classmethod
    def from_polars(
        cls,
        dist: pl.LazyDataFrame,
        input_cols: list[str],
        output_cols: list[str],
        probab_col: str = "p"
    ) -> Joint:
        """
        Factory method: expects a LazyDataFrame with three columns:
        - `input_cols`: name of column containing the possible inputs 
        - `output_cols`: name of column containing the possible outputs 
        - `probab_col`: name of column containing p(input, output)
        """
        dist = dist.rename({probab_col: "p"})
        return cls(dist, input_cols, output_cols)
