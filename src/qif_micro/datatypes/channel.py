from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import polars as pl

def _is_valid(dist: pl.LazyDataFrame, cols: Iterable[str]) -> bool:
    all_cols = set(dist.collect_schema().names())

    assert len(set(cols) - all_cols) == 0
    assert "p" in all_cols

    expr_check_sum = pl.col("p").sum().is_close(1, abs_tol=0.005)
    row_check_sum = dist.group_by(cols).agg(expr_check_sum.alias("is_one"))

    expr_check_all = pl.col("is_one").all()
    return True
    return row_check_sum.select(expr_check_all).collect().item()


@dataclass(frozen=True)
class Channel:
    """
    A channel models probabilistic functions from inputs to outputs.
    That is, given an input x and an output y, the entry x,y of the
    channel corresponds to the conditional probability p(y | x). In other 
    words, a channel is a mapping from inputs to distribution on outputs.
    """
    dist: pl.LazyDataFrame
    input: list[str]
    output: list[str]

    def __post_init__(self):
        columns = set(self.dist.collect_schema().names())

        assert len(set(self.input) - columns) == 0
        assert len(set(self.output) - columns) == 0
        assert "p" in columns

        if not _is_valid(self.dist, self.input):
            raise ValueError("Not a valid channel!")


    def __repr__(self):
        """ Warning: might be expensive!"""
        sorted_dist = self.dist.sort(self.input + self.output)
        return sorted_dist.collect().__repr__()

    
    @classmethod
    def from_polars(
        cls,
        dist: pl.LazyDataFrame,
        input_cols: list[str],
        output_cols: list[str],
        probab_col: str = "p"
    ) -> Channel:
        """
        Factory method: expects a LazyDataFrame with three columns:
        - `input_cols`: names of columns containing the possible inputs 
        - `output_cols`: names of columns containing the possible outputs 
        - `probab_col`: name of column containing p(output | input)
        """
        dist = dist.rename({probab_col: "p"})
        return cls(dist, input_cols, output_cols)
