from collections.abc import Iterable
from typing import Any

import numpy as np

from qif_micro.qif.datatypes import Channel

def geometric(
    eps: float,
    input_domain: Iterable[Any],
    output_domain: Iterable[Any] | None = None,
    domain_min: int | None = None,
    domain_max: int | None = None
) -> Channel:
    """
    Constructs a truncated geometric mechanism, which adds noise to an input
    value by sampling from a geometric distribution truncated to the output.

    The probablity of remapping a value a to another value b is given by

        (1 - alpha) alpha^|a - b| / (1 + alpha) if min B < b max < B
                    alpha^|a - b| / (1 + alpha) otherwise

    where the parameter alpha = e^-eps

    Parameters
    ----------
    eps : float
        The privacy parameter epsilon.

    input_domain : Iterable[Any]
        The domain of input values.

    output_domain : Iterable[Any], optional (default: None)
        The domain of output values.  By default, it is assumed to
        be the same as the input domain.

    domain_min : int, optional (default: None)
        The lower bound of the output domain. By default, it is derived
        from ``output_domain``. This can be used to construct
        slices of the mechanism, by passing a subdomain to ``output_domain``.
        
    domain_min : int, optional (default: None)
        The upper bound of the output domain. By default, it is derived
        from ``output_domain``. This can be used to construct
        slices of the mechanism, by passing a subdomain to ``output_domain``.

    Returns
    -------
    Channel
        Stochastic channel (matrix) mapping input to output values.

    Pre-conditions
    --------------
    - The privacy parameter ``eps`` must be >= 0 and finite
      (otherwise the resulting channel is the identity, and thus not DP)

    - The values in ``input_domain`` and ``output_domain`` must be integers
    
    - ``domain_min`` must be <= than the smallest value in ``output_domain``
    - ``domain_max`` must be >= than the largest value in ``output_domain``
    - ``domain_min`` must be strictly smaller than ``domain_max``

    - If ``domain_min`` or ``domain_max`` are not informed, it is assumed that
      the resulting mechanism will be complete, in which case ``output_domain``
      must be a superset of ``input_domain``. The resulting mechanism may still
      be a slice, if ``output_domain`` is a superset, but is not contiguous.

    Post-conditions
    ---------------
    - The indices of the rows in the channel matrix correspond to the values
      in ``input_domain`` sorted in ascending order

    - The indices of the cols in the channel matrix correspond to the values
      in ``input_domain`` sorted in ascending order

    Examples
    --------
    >>> import math
    >>> from functools import partial
    >>> from qif_micro import qif
    
    By default it is assumed that input_domain = output_domain:

    >>> input_domain = [0, 1, 2]
    >>> tg = partial(qif.dp.geometric, input_domain=input_domain)

    >>> eps = - math.log(1/2) # alpha = 1/2
    >>> tg(eps).dist
    array([[0.66666667, 0.16666667, 0.16666667],
           [0.33333333, 0.33333333, 0.33333333],
           [0.16666667, 0.16666667, 0.66666667]])

    We can also construct a mechanism where the output domain has outliers:

    >>> domain = [-1, 0, 1, 2, 3]
    >>> tg(eps, output_domain=domain, domain_min=-1, domain_max=3).dist
    array([[0.33333333, 0.33333333, 0.16666667, 0.08333333, 0.08333333],
           [0.16666667, 0.16666667, 0.33333333, 0.16666667, 0.16666667],
           [0.08333333, 0.08333333, 0.16666667, 0.33333333, 0.33333333]])

    And we can construct a slice of the mechanism:
    >>> tg(eps, output_domain=[0, 2], domain_min=-1, domain_max=3).dist
    array([[0.33333333, 0.08333333],
           [0.16666667, 0.16666667],
           [0.08333333, 0.33333333]])
    """
    # ========================================================================
    # Pre-conditions
    # ========================================================================
    if eps < 0: raise ValueError("Privacy param ``eps`` must be >= 0!")
    if np.isinf(eps): raise ValueError("Privacy param ``eps`` must be finite!")
    
    input_domain = np.unique(input_domain)
    if output_domain is None: output_domain = input_domain
    else: output_domain = np.unique(output_domain)

    n_rows = input_domain.shape[0]
    n_cols = output_domain.shape[0]

    err_msg = "cannot be empty!"
    if n_rows <= 0: raise ValueError(f"``input_domain`` {err_msg}")
    if n_cols <= 0: raise ValueError(f"``output_domain`` {err_msg}")

    if not np.issubdtype(input_domain.dtype, np.integer):
        raise ValueError("``input_domain`` must contain only integers!")

    if not np.issubdtype(output_domain.dtype, np.integer):
        raise ValueError("``output_domain`` must contain only integers!")

    is_complete = (domain_min is None) and (domain_max is None)

    # Set domain boundaries, in case of slice:
    if domain_min is None: domain_min = output_domain[0]
    if domain_max is None: domain_max = output_domain[-1]

    if domain_min >= domain_max:
        domain_min_msg = f"``domain_min`` ({domain_min})"
        domain_max_msg = f"``domain_max`` ({domain_max})"
        raise ValueError(f"{domain_max_msg} must be > {domain_min_msg}")

    if domain_min > output_domain[0]:
        domain_min_msg = f"``domain_min`` ({domain_min})"
        output_min_msg = f"min ``output_domain`` ({output_domain[0]})"
        raise ValueError(f"{domain_min_msg} must be <= {output_min_msg}")

    if domain_max < output_domain[-1]:
        domain_max_msg = f"``domain_max`` ({domain_max})"
        output_max_msg = f"max ``output_domain`` ({output_domain[-1]})"
        raise ValueError(f"{domain_max_msg} must be >= {output_max_msg}")

    diff_inp = np.setdiff1d(input_domain, output_domain, assume_unique=True)
    if is_complete and (diff_inp.shape[0] > 0):
        raise ValueError("Full channel: output must be a superset of input!")
    # ========================================================================

    # Derive alpha from privacy parameter eps
    alpha = np.exp(-eps)

    # Create a grid of distances |a - b|
    # Broadcasting: input_domain (n rows, 1) - output_domain (1, n cols)
    distances = np.abs(
        input_domain[:, np.newaxis] - output_domain[np.newaxis, :]
    )

    # Calculate the base weights: alpha^|a - b| / (1 + alpha)
    base_weights = np.power(alpha, distances) / (1 + alpha)

    # Apply the boundary condition using the full domain boundaries:
    # Interior points (min < b < max): multiply by (1 - alpha)
    # Boundary points (b = min or b = max): multiply by 1
    is_interior = (output_domain > domain_min) & (output_domain < domain_max)
    boundary_factor = np.where(is_interior, (1 - alpha), 1.0)

    return Channel(base_weights * boundary_factor)


def random_response(
    eps: float,
    input_domain: Iterable[Any],
    output_domain: Iterable[Any] | None = None,
    domain_size: int | None = None
) -> Channel:
    """
    Constructs a random-response mechanism with a privacy parameter ``eps``.
    Equivalently, a mechanism that keeps the input value with probability

        p = e^eps / (e^eps + domain_size - 1)

    and replaces the input value with some other value with probability

        (1 - p) / (domain_size - 1) = 1 / (e^eps + domain_size - 1)

    Parameters
    ----------
    eps : float
        The privacy parameter epsilon.

    input_domain : Iterable[Any]
        The domain of input values.

    output_domain : Iterable[Any], optional (default: None)
        The domain of output values. Must be a superset of the input domain.
        By default, it is assumed to be the same as the input domain.

    domain_size: int, optional (default: None)
        The size of the output domain. By default, it is assumed to be the
        number of elements in ``output_domain``. This can be used to construct
        slices of the mechanism, by passing a subdomain to ``output_domain``.

    Returns
    -------
    Channel
        Stochastic channel (matrix) mapping input to output values.
        
    Pre-conditions
    --------------
    - The privacy parameter ``eps`` must be >= 0 and finite
      (otherwise the resulting channel is the identity, and thus not DP)

    - The size of ``input_domain`` must be >= 1 and <= ``domain_size``

    - ``domain_size`` must be >= 2 in all cases, including when not specified,
      in which case it is derived from ``output_domain``

    - A consequence of the above is that the size of ``output_domain`` must be
      >= 2 when ``domain_size`` is not specified. If ``domain_size`` is set,
      then the size of ``output_domain`` must be >= 1 and <= ``domain_size``
    
    - If ``domain_size`` is not informed, it is assumed that the resulting
      mechanism will be complete, in which case ``output_domain`` must be
      a superset of ``input_domain``

    Post-conditions
    ---------------
    - The indices of the rows in the channel matrix correspond to the values
      in ``input_domain`` sorted in ascending order

    - The indices of the cols in the channel matrix correspond to the values
      in ``input_domain`` sorted in ascending order
    
    Examples
    --------
    >>> import math
    >>> from functools import partial
    >>> from qif_micro import qif

    By default it is assumed that input_domain = output_domain:
    
    >>> input_domain = ["a", "b", "c"]
    >>> rr = partial(qif.dp.random_response, input_domain=input_domain)

    >>> eps = math.log(2) # p = 1/2
    >>> rr(eps).dist
    array([[0.5 , 0.25, 0.25],
           [0.25, 0.5 , 0.25],
           [0.25, 0.25, 0.5 ]])

    We can also construct a mechanism where the output domain has outliers:
    
    >>> eps = math.log(4) # p = 1/2
    >>> rr(eps, output_domain=["a", "b", "c", "d", "e"]).dist
    array([[0.5  , 0.125, 0.125, 0.125, 0.125],
           [0.125, 0.5  , 0.125, 0.125, 0.125],
           [0.125, 0.125, 0.5  , 0.125, 0.125]])

    And we can construct a slice of the mechanism:
    
    >>> rr(eps, output_domain=["a", "c"], domain_size=5).dist
    array([[0.5  , 0.125],
           [0.125, 0.125],
           [0.125, 0.5  ]])
    """
    # ========================================================================
    # Pre-conditions
    # ========================================================================
    if eps < 0: raise ValueError("Privacy param ``eps`` must be >= 0!")
    if np.isinf(eps): raise ValueError("Privacy param ``eps`` must be finite!")

    input_domain = np.unique(input_domain)
    if output_domain is None: output_domain = input_domain
    else: output_domain = np.unique(output_domain)

    n_rows = input_domain.shape[0]
    n_cols = output_domain.shape[0]

    err_msg = "cannot be empty!"
    if n_rows <= 0: raise ValueError(f"``input_domain`` {err_msg}")
    if n_cols <= 0: raise ValueError(f"``output_domain`` {err_msg}")

    if domain_size is None: domain_size = n_cols
    if domain_size < 2: raise ValueError("``domain_size`` must be >= 2")

    err_msg = "has more values than expected ``domain_size``!"
    if n_rows > domain_size: raise ValueError(f"``input_domain`` {err_msg}")
    if n_cols > domain_size: raise ValueError(f"``output_domain`` {err_msg}")

    is_complete = n_cols == domain_size
    diff_inp = np.setdiff1d(input_domain, output_domain, assume_unique=True)

    if is_complete and (diff_inp.shape[0] > 0):
        raise ValueError("Full channel: output must be a superset of input!")
    # ========================================================================
     
    # Derive probabilty of preserving or replacing input value from epsilon:
    exp_eps = np.exp(eps)
    p_keep = exp_eps / (exp_eps + domain_size - 1)
    p_replace = 1 / (exp_eps + domain_size - 1)

    # Initialize with p_replace, then set matching positions to p_keep
    # This avoids creating a full boolean matrix for the comparison
    dist = np.full((n_rows, n_cols), p_replace, dtype=np.float64)
    
    # Find positions where input values match output values
    # Both domains are sorted (from np.unique), so binary search is efficient
    indices = np.searchsorted(input_domain, output_domain, side="left")

    in_bounds = np.nonzero(indices < n_rows)[0]
    in_bounds_input = input_domain[indices[in_bounds]]
    in_bounds_output = output_domain[in_bounds]

    matches = np.zeros(n_cols, dtype=bool)
    matches[in_bounds] = in_bounds_input == in_bounds_output

    dist[indices[matches], np.nonzero(matches)[0]] = p_keep
    
    return Channel(dist)
