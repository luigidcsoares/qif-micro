from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from qif_micro.qif.datatypes import Channel

def build(
    alpha: np.floating,
    input_domain: Iterable[Any],
    output_domain: Iterable[Any] | None = None,
    return_labels: bool = False
) -> Channel | tuple[Channel, Sequence[Any]]:
    """
    Constructs a truncated geometric mechanism, which adds noise to an input
    value by sampling from a geometric distribution truncated to the output.

    Parameters
    ----------
    alpha : np.floating
        The privacy parameter. Must satisfy 0 < alpha <= 1.

    input_domain : Iterable[Any]
        The domain of possible input values. Must contain integers.

    output_domain : Iterable[Any] | None
        The domain of possible output values. If None, defaults to input_domain.
        Must be a superset of input_domain and contain integers.

    return_labels : bool
        If True, returns the channel and the sorted labels for rows and columns.

    Returns
    -------
    Channel | tuple[Channel, Sequence[Any]]
        The constructed channel matrix. If return_labels is True, also returns
        the row and column labels.

    Examples
    --------
    >>> from qif_micro import mechanism
    >>> input_domain = [0, 1, 2]
    >>> output_domain = [0, 1, 2]
    >>> mechanism.geometric(0.5, input_domain, output_domain).dist
    array([[0.66666667, 0.16666667, 0.16666667],
           [0.33333333, 0.33333333, 0.33333333],
           [0.16666667, 0.16666667, 0.66666667]])
    """
    # ========================================================================
    # Pre-conditions
    # ========================================================================

    if not (0 < alpha <= 1):
        raise ValueError("Alpha must satisfy 0 < ``alpha`` <= 1")

    # The output domain must be a superset of the the input:
    input_domain = set(input_domain)
    output_domain = input_domain if output_domain is None else set(output_domain)

    if len(input_domain - output_domain) > 0:
        raise ValueError("Output must be a superset of input!")

    # Convert to numpy arrays early for type validation and sorting
    input_domain = np.sort(list(input_domain))
    output_domain = np.sort(list(output_domain))
    
    if not np.issubdtype(input_domain.dtype, np.integer):
        raise ValueError("Input domain must contain only integers.")

    if not np.issubdtype(output_domain.dtype, np.integer):
        raise ValueError("Output domain must contain only integers.")

    # ========================================================================

    n_rows = input_domain.shape[0]
    n_cols = output_domain.shape[0]

    min_val = output_domain[0]
    max_val = output_domain[-1]

    # Create a grid of distances |a - b|
    # Broadcasting: input_domain (n_rows, 1) - output_domain (1, n_cols)
    distances = input_domain[:, np.newaxis] - output_domain[np.newaxis, :]
    distances = np.abs(distances)

    # Calculate the base weights: alpha^|a - b| / (1 + alpha)
    base_weights = np.power(alpha, distances) / (1 + alpha)

    # Apply the boundary condition:
    # Interior points (min < b < max): multiply by (1 - alpha)
    # Boundary points (b = min or b = max): multiply by 1
    is_interior = (output_domain > min_val) & (output_domain < max_val)
    boundary_factor = np.where(is_interior, (1 - alpha), 1.0)

    ch = Channel(base_weights * boundary_factor)
    return (ch, input_domain, output_domain) if return_labels else ch
