from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import polars

def _is_valid(dist: polars.LazyDataFrame, cols: Iterable[str]) -> bool:
    columns = set(dist.collect_schema().names())

    assert len(set(cols) - columns) == 0
    assert "p" in columns

    expr_check_sum = polars.col("p").sum().is_close(1, abs_tol=1e-3)
    row_check_sum = dist.group_by(cols).agg(is_one=expr_check_sum)

    expr_check_all = polars.col("is_one").all()
    return row_check_sum.select(expr_check_all).collect().item()


@dataclass(frozen=True)
class Channel:
    """
    A channel models probabilistic functions from inputs to outputs.
    That is, given an input x and an output y, the entry x,y of the
    channel corresponds to the conditional probability p(y | x). In other 
    words, a channel is a mapping from inputs to distribution on outputs.
    """
    dist: polars.LazyDataFrame
    input_names: list[str]
    output_names: list[str]

    def __post_init__(self):
        columns = set(self.dist.collect_schema().names())

        assert len(set(self.input_names) - columns) == 0
        assert len(set(self.output_names) - columns) == 0
        assert "p" in columns

        if not _is_valid(self.dist, self.input_names):
            raise ValueError("Not a valid channel!")

    
    @classmethod
    def from_polars(
        cls,
        dist: polars.LazyDataFrame,
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
