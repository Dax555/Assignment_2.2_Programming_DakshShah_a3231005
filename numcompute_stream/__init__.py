"""NumCompute-Stream: a NumPy-only, decision-tree-based streaming ML toolkit.

Extends the Assignment 2.1 NumCompute toolkit so that every component can be
updated chunk by chunk (``partial_fit`` / ``update``), and adds a buffered
decision tree, a stream trainer and matplotlib plotting helpers.

Submodules
----------
io, preprocessing, sort_search, rank, stats, metrics, optim, pipeline, utils,
benchmarking (carried over from 2.1 and made stream-compatible), plus
tree (DecisionTreeClassifier), stream (StreamTrainer) and visualise.

``visualise`` is not imported here, so ``import numcompute_stream`` works on a
machine without matplotlib; use ``from numcompute_stream import visualise``.
"""

from . import (benchmarking, io, metrics, optim, pipeline, preprocessing,
               rank, sort_search, stats, stream, tree, utils)
from .pipeline import Pipeline
from .stream import StreamTrainer
from .tree import DecisionTree, DecisionTreeClassifier

__version__ = "0.2.0"
__all__ = ["benchmarking", "io", "metrics", "optim", "pipeline",
           "preprocessing", "rank", "sort_search", "stats", "stream", "tree",
           "utils", "Pipeline", "StreamTrainer", "DecisionTreeClassifier",
           "DecisionTree"]
