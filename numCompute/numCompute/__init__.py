"""NumCompute: a vectorised, NumPy-only scientific computing toolkit.

Submodules: io, preprocessing, sort_search, rank, stats, metrics, optim,
pipeline, utils, benchmarking.
"""

from . import (io, metrics, optim, pipeline, preprocessing,
               rank, sort_search, stats, utils)

__version__ = "0.1.0"
__all__ = ["io", "metrics", "optim", "pipeline",
           "preprocessing", "rank", "sort_search", "stats", "utils"]