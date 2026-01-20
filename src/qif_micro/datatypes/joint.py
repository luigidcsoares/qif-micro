from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from qif_micro._internal.dataset import _valid_columns

def _is_valid(dist: pl.LazyFrame) -> bool:
    expr_check_sum = pl.col("p").sum().is_close(1, abs_tol=0.005)
    return dist.select(expr_check_sum).collect().item()


@dataclass(frozen=True)
class Joint:
    """
    A joint models the correlation between the input and a output
    of a probabilistic function. That is, given an input x and an 
    output y, the entry x,y of a joint is the joint probability p(x, y).
    """
    dist: pl.LazyFrame
    secret: list[str]
    output: list[str]

    def __post_init__(self):
        expected_cols = ["p", *self.secret, *self.output]
        _, ok = _valid_columns(self.dist, expected_cols)
        assert ok

        if not _is_valid(self.dist):
            raise ValueError("Not a valid joint distribution!")


    def __repr__(self):
        """ Warning: might be expensive!"""
        schema = self.dist.collect_schema()
        sorted_dist = self.dist.sort(self.secret + self.output)

        for col in self.secret + self.output:
            dtype = schema[col]
            if isinstance(dtype, pl.List):
                sort_within_expr = pl.col(col).list.sort()
                sorted_dist = sorted_dist.with_columns(sort_within_expr)

        return sorted_dist.collect().__repr__()


@dataclass(frozen=True)
class LazyJoint:
    """
    Lazy version of joint, which defers validation until is has been
    materialised into a proper joint. This is mainly to improve
    performance, but at the trade-off of late errors...
    """
    dist: pl.LazyFrame
    secret: list[str]
    output: list[str]

    def __post_init__(self):
        expected_cols = ["p", *self.secret, *self.output]
        _, ok = _valid_columns(self.dist, expected_cols)
        assert ok

    
def make(
    dist: pl.DataFrame | pl.LazyFrame,
    secret_cols: Iterable[str],
    output_cols: Iterable[str],
    probab_col: str = "p"
) -> Joint:
    """
    Factory method: expects a LazyFrame with three columns:
    - `secret_cols`: names of columns containing the possible secrets
    - `output_cols`: names of columns containing the possible outputs 
    - `probab_col`: name of column containing p(output | input)
    """
    dist = dist.lazy().rename({probab_col: "p"})
    return Joint(dist, secret_cols, output_cols)


def make_lazy(
    dist: pl.DataFrame | pl.LazyFrame,
    secret_cols: Iterable[str],
    output_cols: Iterable[str],
    probab_col: str = "p"
) -> Joint:
    """
    Factory method: expects a LazyFrame with three columns:
    - `secret_cols`: names of columns containing the possible secrets
    - `output_cols`: names of columns containing the possible outputs 
    - `probab_col`: name of column containing p(output | input)
    """
    dist = dist.lazy().rename({probab_col: "p"})
    return LazyJoint(dist, secret_cols, output_cols)


def collect(joint: Joint | LazyJoint) -> Joint:
    return Joint(joint.dist, joint.secret, joint.output)
