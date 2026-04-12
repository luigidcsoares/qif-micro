from dataclasses import dataclass, field

import numpy as np

from qif_micro.qif.datatypes.probab_dist import ProbabDist
from qif_micro.qif.datatypes.stoch_matrix import StochMatrix


@dataclass(frozen=True)
class Hyper:
    """
    Represents the result of pushing a prior through a channel.

    A Hyper object stores both the marginal distribution (outer) over outputs
    and the posterior distributions (conditional probabilities of inputs given
    each output observation).

    Attributes
    ----------
    outer : ProbabDist
        The marginal distribution over outputs (1D probability distribution).
        Shape is (n_outputs,). Access completeness via outer.is_complete.

    posteriors : StochMatrix
        The posterior distributions (inputs conditioned on outputs).
        Must have dist_orient=0 (column-oriented, P(input|output)).
        Shape is (n_inputs, n_outputs) or partitioned by columns.
        Access completeness via posteriors.is_complete.

    Pre-conditions
    ---------------
    - ``outer`` must be a ProbabDist
    - ``posteriors`` must be a StochMatrix with dist_orient=0
    - All posterior entries must be non-negative
    - Each posterior column must sum to at most 1.0
    - If posteriors are partitioned, all partitions must have same row count

    Examples
    --------
    >>> import numpy as np
    >>> from qif_micro.qif.datatypes import Hyper, ProbabDist
    >>> from qif_micro.qif.datatypes import StochMatrix

    >>> outer = ProbabDist(np.array([0.25, 0.75]))
    >>> posteriors = StochMatrix(np.array([
    ...     [2/3, 1/3],
    ...     [1/3, 2/3]
    ... ]), dist_orient=0)

    >>> h = Hyper(outer, posteriors)
    >>> h.outer
    ProbabDist(dist=array([0.25, 0.75]), is_complete=True)

    >>> h.posteriors
    StochMatrix(dist=array([[0.66666667, 0.33333333],
           [0.33333333, 0.66666667]]), dist_orient=0, is_complete=True)
    """
    outer: ProbabDist
    posteriors: StochMatrix

    def __post_init__(self):
        # ====================================================================
        # Pre-conditions
        # ====================================================================
        if not isinstance(self.outer, ProbabDist):
            raise TypeError("``outer`` must be a ProbabDist!")

        if not isinstance(self.posteriors, StochMatrix):
            raise TypeError("``posteriors`` must be a StochMatrix!")

        if self.posteriors.dist_orient != 0:
            raise ValueError(
                "``posteriors`` must have dist_orient=0 "
                "(column-oriented, P(input|output))!"
            )
        # ===================================================================
