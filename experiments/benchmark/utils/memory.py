"""Memory profiling utilities for benchmarking"""

import resource


def current() -> float:
    """
    Get current process memory usage in GiB.

    Returns
    -------
    float
        Memory usage in GiB.
    """
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    rusage_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    mem_kib = max(rusage.ru_maxrss, rusage_child.ru_maxrss)
    return mem_kib / 2**20
