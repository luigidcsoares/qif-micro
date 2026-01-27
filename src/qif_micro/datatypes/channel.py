from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from qif_micro._internal import _valid_columns

def _is_valid(dist: pl.LazyFrame, cols: Iterable[str]) -> bool:
    expr_check_sum = pl.col("p").sum().is_close(1, abs_tol=0.005)
    row_check_sum = dist.group_by(cols).agg(expr_check_sum.alias("is_one"))

    expr_check_all = pl.col("is_one").all()
    return row_check_sum.select(expr_check_all).collect().item()


@dataclass(frozen=True)
class Channel:
    """
    A channel models probabilistic functions from inputs to outputs.
    That is, given an input x and an output y, the entry x,y of the
    channel corresponds to the conditional probability p(y | x). In other 
    words, a channel is a mapping from inputs to distribution on outputs.
    """
    dist: pl.LazyFrame
    secret: list[str]
    output: list[str]

    def __post_init__(self):
        expected_cols = ["p", *self.secret, *self.output]
        _, ok = _valid_columns(self.dist, expected_cols)
        assert ok

        if not _is_valid(self.dist, self.secret):
            raise ValueError("Not a valid channel!")


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
class LazyChannel:
    """
    Lazy version of channel, which defers validation until is has been
    materialised into a proper channel. This is mainly to improve
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
) -> Channel:
    """
    Factory method: expects a data frame with three columns:
    - `secret_cols`: names of columns containing the possible secrets
    - `output_cols`: names of columns containing the possible outputs 
    - `probab_col`: name of column containing p(output | input)
    """
    dist = dist.lazy().rename({probab_col: "p"})
    return Channel(dist, secret_cols, output_cols)


def make_lazy(
    dist: pl.DataFrame | pl.LazyFrame,
    secret_cols: Iterable[str],
    output_cols: Iterable[str],
    probab_col: str = "p"
) -> LazyChannel:
    """
    Factory method: expects a data frame with three columns:
    - `secret_cols`: names of columns containing the possible secrets
    - `output_cols`: names of columns containing the possible outputs 
    - `probab_col`: name of column containing p(output | input)
    """
    dist = dist.lazy().rename({probab_col: "p"})
    return LazyChannel(dist, secret_cols, output_cols)


def collect(ch: Channel | LazyChannel) -> Channel:
    return Channel(ch.dist, ch.secret, ch.output)
