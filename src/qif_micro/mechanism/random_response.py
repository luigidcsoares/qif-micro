from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from qif_micro.qif.datatypes import Channel

def build(
    p: np.floating,
    input_domain: Iterable[Any],
    output_domain: Iterable[Any] | None = None,
    return_labels: bool = False
) -> Channel | tuple[Channel, Sequence[Any]]:
    """
    Examples
    --------
    >>> from qif_micro import mechanism
    >>> input_domain = [0, 1, 2]
    >>> output_domain = [0, 1, 2]
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

    And we can get the labels for each row and column:

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
    """
    # ========================================================================
    # Pre-conditions
    # ========================================================================

    # The output domain must be a superset of the the input:
    input_domain = set(input_domain)
    output_domain = input_domain if output_domain is None else set(output_domain)

    if len(input_domain - output_domain) > 0:
        raise ValueError("Output must be a superset of input!")
    
    # ========================================================================

    outliers = np.array(list(output_domain - input_domain))

    input_domain = np.array(list(input_domain))
    output_domain = np.array(list(output_domain))

    n_rows = len(input_domain)
    n_cols = len(output_domain)

    # Use np.unique to map domains to rows and columns:
    input_labels, rows = np.unique(input_domain, return_inverse=True)
    
    # Use the same indices for the output, and set any outputs outside
    # of the input domain to the right-hand side of the channel.
    output_labels, cols = np.unique(outliers, return_inverse=True)
    output_labels = [input_labels, output_labels]
    output_labels = np.hstack([a for a in output_labels if a.shape[0] > 0])
    cols = np.hstack([rows, cols + n_rows])

    # We first construct the slice of the channel without outliers:
    p_replace = (1 - p) / (n_cols - 1)
    dist_matching = np.full((n_rows, n_rows), p_replace)
    np.fill_diagonal(dist_matching, p)

    # Then we construct a second partition for outliers
    dist_outliers = np.full((n_rows, n_cols - n_rows), p_replace)
    
    ch = Channel(np.hstack([dist_matching, dist_outliers]))
    return (ch, input_labels, output_labels) if return_labels else ch
