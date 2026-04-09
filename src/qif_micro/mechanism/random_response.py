from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from qif_micro.qif.datatypes import Channel

def build(
    p: np.floating,
    input_domain: Iterable[Any],
    output_domain: Iterable[Any] | None = None,
    domain_size: int | None = None,
    return_labels: bool = False
) -> Channel | tuple[Channel, Sequence[Any]]:
    """
    Examples
    --------
    >>> from qif_micro import mechanism
    >>> input_domain = [0, 1, 2]
    >>> mechanism.random_response(1/2, input_domain).dist
    array([[0.5 , 0.25, 0.25],
           [0.25, 0.5 , 0.25],
           [0.25, 0.25, 0.5 ]])

    We can also construct a RR mechanism where the output domain has outliers:
    
    >>> input_domain = ["d", "a", "b"]
    >>> output_domain = ["a", "b", "c", "d", "e"]
    >>> mechanism.random_response(1/2, input_domain, output_domain).dist
    array([[0.5  , 0.125, 0.125, 0.125, 0.125],
           [0.125, 0.5  , 0.125, 0.125, 0.125],
           [0.125, 0.125, 0.5  , 0.125, 0.125]])

    We can get the labels for each row and column:

    >>> row_labels, col_labels = mechanism.random_response(
    ...     1/2,
    ...     input_domain,
    ...     output_domain,
    ...     return_labels=True
    ... )[1:]

    >>> row_labels
    array(['a', 'b', 'd'], dtype='<U1')

    >>> col_labels
    array(['a', 'b', 'd', 'c', 'e'], dtype='<U1')

    And we can construct a slice of the mechanism:
    
    >>> mechanism.random_response(1/2, [0, 1, 2], [0, 2], domain_size=3).dist
    array([[0.5 , 0.25],
           [0.25, 0.5 ],
           [0.25, 0.25]])
    """
    # ========================================================================
    # Pre-conditions
    # ========================================================================

    input_domain = np.unique(input_domain)

    if output_domain is None: output_domain = input_domain
    else: output_domain = np.unique(output_domain)

    n_rows = input_domain.shape[0]
    n_cols = output_domain.shape[0]

    if domain_size is not None:
        if n_rows > domain_size:
            raise ValueError("Input domain has more values than expected ``domain_size``")

        if n_cols > domain_size:
            raise ValueError("Output domain has more values than expected ``domain_size``")
    else:
        domain_size = n_cols

    is_complete= n_cols == domain_size
    diff_inp = np.setdiff1d(input_domain, output_domain, assume_unique=True)

    if is_complete and (diff_inp.shape[0] > 0):
        raise ValueError("Full channel: output must be a superset of input!")
    
    # ========================================================================

    diff_out = np.setdiff1d(output_domain, input_domain, assume_unique=True)
    shared = np.intersect1d(input_domain, output_domain, assume_unique=True)

    # We reorder to keep shared values at the front:
    input_domain = np.hstack([shared, diff_inp])
    output_domain = np.hstack([shared, diff_out])

    n_shared = shared.shape[0]
    n_outliers = diff_out.shape[0]
    assert (n_shared + n_outliers) == n_cols
    
    # We first construct the slice of the channel without outliers:
    p_replace = (1 - p) / (domain_size - 1)
    dist_matching = np.full((n_rows, n_shared), p_replace)
    np.fill_diagonal(dist_matching, p)

    # Then we construct a second partition for outliers
    dist_outliers = np.full((n_rows, n_outliers), p_replace)
    
    ch = Channel(np.hstack([dist_matching, dist_outliers]))
    return (ch, input_domain, output_domain) if return_labels else ch
