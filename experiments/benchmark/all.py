"""End-to-end benchmark orchestrator

Runs all benchmarks (generic and count_sum) and aggregates results.
"""
import sys

from benchmark.generic import benchmark as generic_benchmark
from benchmark.count_sum import benchmark as count_sum_benchmark


def main() -> int:
    """
    Run all benchmarks end-to-end.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        print("Starting benchmarks...\n")

        print("Running generic benchmarks...")
        generic_results = generic_benchmark.run_generic_benchmarks()
        print(f"  Completed: {len(generic_results)} scenarios\n")

        print("Running count_sum benchmarks...")
        count_sum_results = count_sum_benchmark.run_count_sum_benchmarks()
        print(f"  Completed: {len(count_sum_results)} scenarios\n")

        total = len(generic_results) + len(count_sum_results)
        print(f"All benchmarks completed: {total} scenarios")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
