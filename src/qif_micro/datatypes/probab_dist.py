from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from qif_micro._internal import _valid_columns

def _is_valid(dist: pl.LazyFrame) -> bool:
    expr_check_sum = pl.col("p").sum().is_close(1, abs_tol=0.005)
    return dist.select(expr_check_sum).collect().item()


@dataclass(frozen=True)
class ProbabDist:
    """
    A ProbabDist is an ensemble (x, Ax, Px), 
    where x is the outcome of a random variable X,
    Ax is the alphabet, with x in Ax, and Px = {p_x0, p_x1, ...}.
    """
    dist: pl.LazyFrame
    secret: list[str]

    def __post_init__(self):
        expected_cols = ["p", *self.secret]
        _, ok = _valid_columns(self.dist, expected_cols)
        assert ok

        if not _is_valid(self.dist):
            raise ValueError("Not a valid probability distribution!")


    def __repr__(self):
        """ Warning: might be expensive!"""
        schema = self.dist.collect_schema()
        sorted_dist = self.dist.sort(self.secret)

        for col in self.secret:
            dtype = schema[col]
            if isinstance(dtype, pl.List):
                sort_within_expr = pl.col(col).list.sort()
                sorted_dist = sorted_dist.with_columns(sort_within_expr)

        return sorted_dist.collect().__repr__()


@dataclass(frozen=True)
class LazyProbabDist:
    """
    Lazy version of probab dist, which defers validation until is has been
    materialised into a proper distribution. This is mainly to improve
    performance, but at the trade-off of late errors...
    """
    dist: pl.LazyFrame
    secret: list[str]

    def __post_init__(self):
        expected_cols = ["p", *self.secret]
        _, ok = _valid_columns(self.dist, expected_cols)
        assert ok

   
def make(
    dist: pl.DataFrame | pl.LazyFrame,
    secret_cols: Iterable[str],
    probab_col: str = "p"
) -> ProbabDist:
    """
    Factory method: expects a data frame with two columns:
    - `secret_cols`: names of columns with the possible secrets
    - `probab_col`: name of column containing p(outcome)
    """
    dist = dist.lazy().rename({probab_col: "p"})
    return ProbabDist(dist, secret_cols)


def make_lazy(
    dist: pl.DataFrame | pl.LazyFrame,
    secret_cols: Iterable[str],
    probab_col: str = "p"
) -> LazyProbabDist:
    """
    Factory method: expects a data frame with two columns:
    - `secret_cols`: names of columns with the possible secrets
    - `probab_col`: name of column containing p(outcome)
    """
    dist = dist.lazy().rename({probab_col: "p"})
    return LazyProbabDist(dist, secret_cols)


def collect(p: ProbabDist | LazyProbabDist) -> ProbabDist:
    return ProbabDist(p.dist, p.secret)
