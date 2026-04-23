"""QIF-Micro Benchmarking Suite

This package contains performance benchmarks (runtime and memory usage)
for the QIF-Micro library, organized by target modules:

- generic: Benchmarks for qif_micro.model.generic and qif_micro.mechanism
- count_sum: Benchmarks for qif_micro.model.count_sum

Run benchmarks:
- python -m benchmark.generic.run    # Generic benchmarks only
- python -m benchmark.count_sum.run  # Count-sum benchmarks only
- python -m benchmark.all            # All benchmarks end-to-end
"""
