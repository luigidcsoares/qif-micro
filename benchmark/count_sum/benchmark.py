"""Benchmark logic for count-sum model

Orchestrates dataset generation, execution, and measurement for
count-sum model benchmarks.
"""
from multiprocessing import get_context

import polars as pl
from loky import ProcessPoolExecutor
from tqdm import tqdm

from benchmark.count_sum.config import ExperimentConfig
from benchmark.count_sum.experiment import Experiment
from benchmark.utils import memory, timing
from qif_micro import model


def _benchmark_experiment(
    e: Experiment,
    iterations: int = 1
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if iterations < 1: raise ValueError("Number of iterations must be >= 1")
        
    def _wrap():
        def fn(): return model.count_sum(
            e.baseline, agg_col="num", group_by_col="cat"
        )

        result_time = timing.measure(fn, iterations=iterations + 1)
        result_time = result_time[1:] # Discard warm-up iteration
            
        result_time = pl.DataFrame({"step": ["all"], "time": result_time })
        result_peak = pl.DataFrame({"peak" : [memory.current()]})

        return result_time, result_peak


    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, context=ctx) as pool:
        return pool.submit(_wrap).result()


def _run(cfg: ExperimentConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    e = Experiment(cfg.n_entries, cfg.n_cat, cfg.n_num)
    return _benchmark_experiment(e, iterations=cfg.iterations)


def run_many(
    cfg: ExperimentConfig,
    experiments: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ctx = get_context("spawn")

    result_time = pl.DataFrame()
    result_peak = pl.DataFrame()

    length_expr = pl.lit(cfg.n_entries).alias("length")
    domain_cat_expr = pl.lit(cfg.n_cat).alias("n_cat")
    domain_num_expr = pl.lit(cfg.n_num).alias("n_num")

    for _ in tqdm(range(experiments), leave=False):
        with ProcessPoolExecutor(max_workers=1, context=ctx) as pool:
            rt, rp = pool.submit(_run, cfg).result()
            rt = rt.with_columns(length_expr, domain_cat_expr, domain_num_expr)
            rp = rp.with_columns(length_expr, domain_cat_expr, domain_num_expr)

        result_time = pl.concat([result_time, rt])
        result_peak = pl.concat([result_peak, rp])

    return result_time, result_peak
