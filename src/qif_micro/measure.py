from collections.abc import Sequence

import numpy as np

from multimethod import multimethod

from qif_micro import qif
from qif_micro.qif.datatypes import Joint

from qif_micro.typing import BaselineModel, Model

@multimethod
def linkage_risk(adv_model: BaselineModel) -> np.floating:
    """
    Measures the risk with respect to a linkage attack where an adversary
    combines some auxiliary information (obtained via external sources)
    with the dataset released via a privacy-preserving pipeline,
    and tries to infer some sensitive information about a target.
    
    Parameters
    ----------
    This function is overloaded:

    adv_model : BaselineModel
        The result of :func:`qif_micro.model.baseline.build`, which assumes
        an adversary who observed the real (de-identified) dataset.

    adv_model : Model
        All other models produced with :mod:`qif_micro.model`, which assumes
        an adversary who observed the result of post-processing a dataset.
    
    Returns
    -------
    np.floating
        Probability of successful record inference by the adversary (0 to 1).
    
    Pre-conditions
    --------------
    - The ``adv_model`` must be a valid result from a model builder function.
    - For Model type: baseline and strategy must have the same partition structure.
    
    Examples
    --------
    >>> import polars as pl
    >>> from qif_micro import measure
    >>> from qif_micro import model

    Consider the following dataset:

    >>> dataset = pl.DataFrame({
    ...     "owner_id": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ...     "entry_id": [0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 3],
    ...     "agg":      [0, 2, 1, 1, 0, 2, 0, 2, 1, 0, 1],
    ...     "group":    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ... })

    We can compute the threat of an adversary who observes the real data.
    Assuming that the goal of the adversary is to recover the entire record
    of a target (chosen at random), we get that the adversary's probability
    of making a correct inference is approximately 60%:

    >>> adv_model = model.baseline(dataset, ["agg", "group"])
    >>> measure.linkage_risk(adv_model)
    np.float64(0.6041666666666666)

    Now, consider an adversary who observes the result of the a query

    .. code-block:: sql
        SELECT count(*) as count, sum(agg_col) as sum
        FROM dataset
        GROUP_BY owner_id, group

    and whose goal is to infer the target's aggregated record.
    The chance that this adversary makes a correct inference is approx 56%:

    >>> adv_model = model.count_sum(dataset, "agg", group_by_col="group")
    >>> measure.linkage_risk(adv_model)
    np.float64(0.5625)
    """
    # TODO: Consider other gain functions.
    #       This requires support for gain fn in the qif lib.
    joint = adv_model if isinstance(adv_model, Joint) else adv_model[:1]
    return qif.measure.bayes.posterior(joint)


@multimethod
def linkage_risk(adv_model: Model) -> np.floating:
    baseline, adv_st = adv_model[:2]

    is_partitioned = isinstance(baseline.dist, Sequence)
    baseline_dist = baseline.dist if is_partitioned else [baseline.dist]

    is_partitioned = isinstance(adv_st.dist, Sequence)
    adv_st_dist = adv_st.dist if is_partitioned else [adv_st.dist]

    # Pre-condition: baseline and st must be partitioned in the same way:
    if len(baseline_dist) != len(adv_st_dist):
        raise ValueError("Baseline and Adv Strategy must have same partitions!")
   
    # TODO: Consider other gain functions.
    #       This requires support for gain fn in the qif lib.
    expected_gain = baseline_dist
    return sum(
        (s_gain * s_st.T).sum()
        for s_gain, s_st in zip(expected_gain, adv_st_dist)
    )
