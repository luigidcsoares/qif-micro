from __future__ import annotations
from dataclasses import dataclass

import polars

def _is_valid(dist: polars.LazyDataFrame) -> bool:
    columns = set(dist.collect_schema().names())
    assert "p" in columns

    expr_check_sum = polars.col("p").sum().is_close(1, abs_tol=1e-3)
    return dist.select(is_one=expr_check_sum).collect().item()


@dataclass(frozen=True)
class Joint:
    """
    A joint models the correlation between the input and a output
    of a probabilistic function. That is, given an input x and an 
    output y, the entry x,y of a joint is the joint probability p(x, y).
    """
    dist: polars.LazyDataFrame
    input_names: list[str]
    output_names: list[str]

    def __post_init__(self):
        columns = set(self.dist.collect_schema().names())

        assert len(set(self.input_names) - columns) == 0
        assert len(set(self.output_names) - columns) == 0
        assert "p" in columns

        if not _is_valid(self.dist):
            raise ValueError("Not a valid joint distribution!")

    
    @classmethod
    def from_polars(
        cls,
        dist: polars.LazyDataFrame,
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
