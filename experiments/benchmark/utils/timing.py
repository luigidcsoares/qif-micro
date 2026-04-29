"""Timing utilities for benchmarking"""

import gc
import timeit
from typing import Any, Callable


def measure(fn: Callable[[], Any], iterations=1) -> tuple[Any, list[float]]:
    if iterations < 1: raise ValueError("Number of iterations must be >= 1")
    
    elapsed = []
    for _ in range(iterations):
        gc.disable()

        start = timeit.default_timer()
        result = fn()
        end = timeit.default_timer()
        
        elapsed.append(end - start)
        gc.enable()
        gc.collect()
    
    return result, elapsed
