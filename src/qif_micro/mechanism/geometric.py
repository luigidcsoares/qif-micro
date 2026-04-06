from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from qif_micro.qif.datatypes import Channel

def build(
    alpha: np.floating,
    input_domain: Iterable[Any],
    output_domain: Iterable[Any] | None = None,
    domain_min: int | None = None,
    domain_max: int | None = None,
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

    domain_min : int | None
        The minimum value of the full domain. If None, defaults to the minimum
        of output_domain. Used for boundary condition calculations.

    domain_max : int | None
        The maximum value of the full domain. If None, defaults to the maximum
        of output_domain. Used for boundary condition calculations.

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

    We can also construct a geometric mechanism as a channel slice by specifying
    the full domain boundaries. The boundary conditions are applied relative to
    the full domain, not just the output domain:

    >>> mechanism.geometric(
    ...     0.5, [1, 2, 3], [1, 2, 3],
    ...     domain_min=0, domain_max=4
    ... ).dist
    array([[0.33333333, 0.16666667, 0.08333333],
           [0.16666667, 0.33333333, 0.16666667],
           [0.08333333, 0.16666667, 0.33333333]])
    """
    # ========================================================================
    # Pre-conditions
    # ========================================================================

    if not (0 < alpha <= 1):
        raise ValueError("Alpha must satisfy 0 < ``alpha`` <= 1")

    input_domain = set(input_domain)
    if output_domain is None: output_domain = input_domain
    else: output_domain = set(output_domain)

    # Convert to numpy arrays early for type validation and sorting
    input_domain = np.sort(list(input_domain))
    output_domain = np.sort(list(output_domain))
    
    if not np.issubdtype(input_domain.dtype, np.integer):
        raise ValueError("Input domain must contain only integers.")

    if not np.issubdtype(output_domain.dtype, np.integer):
        raise ValueError("Output domain must contain only integers.")

    # Set domain boundaries
    if domain_min is None: domain_min = output_domain[0]
    if domain_max is None: domain_max = output_domain[-1]

    is_slice = (
        (domain_min < output_domain[0]) or
        (domain_max > output_domain[-1])
    )

    diff_inp = np.setdiff1d(input_domain, output_domain, assume_unique=True)

    if (not is_slice) and (diff_inp.shape[0] > 0):
        raise ValueError("Full channel: output must be a superset of input!")

    # ========================================================================

    # Create a grid of distances |a - b|
    # Broadcasting: input_domain (n rows, 1) - output_domain (1, n cols)
    distances = input_domain[:, np.newaxis] - output_domain[np.newaxis, :]
    distances = np.abs(distances)

    # Calculate the base weights: alpha^|a - b| / (1 + alpha)
    base_weights = np.power(alpha, distances) / (1 + alpha)

    # Apply the boundary condition using the full domain boundaries:
    # Interior points (min < b < max): multiply by (1 - alpha)
    # Boundary points (b = min or b = max): multiply by 1
    is_interior = (output_domain > domain_min) & (output_domain < domain_max)
    boundary_factor = np.where(is_interior, (1 - alpha), 1.0)

    ch = Channel(base_weights * boundary_factor, is_slice)
    return (ch, input_domain, output_domain) if return_labels else ch
