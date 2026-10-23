"""Reproducible evaluation of Lazy-Commit's local model output.

Run with ``lazycommit-bench`` (or ``python -m lazycommit.benchmark``). Fixture
repositories are generated in fresh temporary directories and deleted
afterwards; the benchmark never touches the repository you run it from.
"""

BENCHMARK_NAME = "lazycommit-model-benchmark"
BENCHMARK_VERSION = "1"
