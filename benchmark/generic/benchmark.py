"""Benchmark logic for qif_micro.model.generic and qif_micro.mechanism

Orchestrates dataset generation, execution, and measurement for
generic model and mechanism benchmarks.
"""
from functools import partial
from multiprocessing import get_context

import numpy as np
import polars as pl
from loky import ProcessPoolExecutor

from benchmark.generic.experiment import Experiment
from benchmark.utils import memory, timing
from qif_micro import measure, mechanism, model, qif


def _benchmark_experiment(
    e: Experiment,
    iterations: int = 1
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if iterations < 1: raise ValueError("Number of iterations must be >= 1")
        
    def _wrap():
        # Warm-up build mechanism and construct model.
        hint = ["num"] # Set the hint to be the attr with largest domain
        args = [
            e.pi, 
            e.subdomain_records,
            # We have already computed the time to run the mechanism,
            # so we do not want to run it again.
            partial(mechanism.record, **e.mechanisms),
            e.baseline, 
            e.observed_dataset, 
            hint
        ]

        def fn(): return model.generic(*args)  # ty:ignore[invalid-argument-type]
        _ = timing.measure(fn, iterations=1)

        # We start by benchmarking the process of constructing
        # the record-level mechanism:
        args = [e.subdomain_records, e.observed_records]
        kwargs = e.mechanisms
        def fn(): return mechanism.record(*args, **kwargs)  # ty:ignore[invalid-argument-type]
        result_mechanism = timing.measure(fn, iterations=iterations)
    
        # The second step is to construct the baseline and adv models:
        args = [
            e.pi, 
            e.subdomain_records, 
            # We have already computed the time to run the mechanism,
            # so we do not want to run it again.
            lambda input_domain, output_domain: result_mechanism[0],
            e.baseline, 
            e.observed_dataset, 
            hint
        ]
        
        def fn(): return model.generic(*args)  # ty:ignore[invalid-argument-type]
        result_model = timing.measure(fn, iterations=iterations)
    
        # And finally linkage risk:
        def fn(): return measure.linkage_risk(result_model[0])
        result_risk = timing.measure(fn, iterations=iterations)
    
        times = [r[1] for r in [result_mechanism, result_model, result_risk]]
        total_time = list(map(sum, zip(*times)))
        
        result_time = pl.DataFrame({
            "step": ["mechanism", "model", "risk", "all"],
            "time": [*times, total_time],
        })

        result_peak = pl.DataFrame({"peak" : [memory.current()]})
        return result_time, result_peak


    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, context=ctx) as pool:
        return pool.submit(_wrap).result()


def run(
    n_entries: int,
    n_cat: int,
    n_num: int,
    sanitise_cat: bool = False,
    sanitise_num: bool = False,    
    iterations: int = 1
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rr = partial(qif.dp.random_response, eps=1)
    tg = partial(qif.dp.geometric, eps=1)

    mechanisms = {}

    if sanitise_cat:
        m_cat = partial(rr, eps=1, domain_size=n_cat)
        mechanisms |= {"cat": m_cat}

    if sanitise_num:
        m_num = partial(tg, eps=1, domain_min=0, domain_max=n_num)
        mechanisms |= {"num": m_num}

    e = Experiment(n_entries, n_cat, n_num, **mechanisms)
    return _benchmark_experiment(e, iterations=iterations)
