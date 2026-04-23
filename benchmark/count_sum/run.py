"""Entry point for count_sum benchmarks

Runs count_sum benchmarks and generates results/plots.
"""
import sys

from benchmark.count_sum import benchmark


def main() -> int:
    """
    Run count_sum benchmarks.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        results = benchmark.run_count_sum_benchmarks()
        print(f"Count_sum benchmarks completed: {len(results)} scenarios")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
