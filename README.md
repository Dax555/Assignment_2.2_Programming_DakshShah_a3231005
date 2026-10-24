# NumCompute

**NumCompute** is a modular, vectorised scientific-computing toolkit written with **plain Python and NumPy only** (no scikit-learn, pandas, SciPy or deep-learning libraries). It provides reusable building blocks for data input and preprocessing, sorting and top-k, ranking, descriptive statistics, evaluation metrics, finite-difference gradients, a lightweight pipeline abstraction and benchmark tools.

The emphasis is on numerical stability (max-shifted softmax and `logsumexp`, Welford-style running moments, NaN and tie handling), consistent `axis` and shape semantics, informative error messages, and docstrings that state parameters, return shapes, exceptions and time/space complexity.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [API overview](#api-overview)
- [Conventions](#conventions)
- [Running the demo notebook](#running-the-demo-notebook)
- [Running the tests](#running-the-tests)
- [Running the benchmarks](#running-the-benchmarks)
- [Performance summary](#performance-summary)
- [Troubleshooting](#troubleshooting)
- [Team contributions](#team-contributions)

## Requirements

| Requirement | Version | Needed for |
|---|---|---|
| Python | 3.9 or newer | everything |
| NumPy | 1.22 or newer (developed and tested with NumPy 2.x) | everything (the only runtime dependency) |
| pytest | 7 or newer | running the tests (optional) |
| Jupyter or Google Colab | any recent version | the demo notebook (optional) |
| git | any | cloning the repository (optional, a zip download also works) |

External ML, DL and data libraries are deliberately not used. Besides NumPy, only the Python standard library is used (for example `os`, `sys`, `csv`, `gc`, `time`, `platform`, `argparse`, `warnings`, `math`; `heapq` and `bisect` appear only in the benchmark baselines).

## Installation

Pick whichever option fits. All three give you `import numcompute`.

### Option A: clone and install in editable mode (recommended)

```bash
git clone https://github.com/<your-username>/Assignment_2.1_Programming_Task_1_-RedHats.git
cd Assignment_2.1_Programming_Task_1_-RedHats
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # installs numpy (+ pytest and jupyter with [dev])
python -c "import numcompute; print('ok')"
```

### Option B: install directly from GitHub

```bash
pip install "git+https://github.com/<your-username>/Assignment_2.1_Programming_Task_1_-RedHats.git"
```

Inside a notebook use `%pip install "git+https://github.com/<your-username>/Assignment_2.1_Programming_Task_1_-RedHats.git"`. To pin a branch or tag append `@branch-name`.

### Option C: no installation, run from the repository root

```bash
git clone https://github.com/<your-username>/Assignment_2.1_Programming_Task_1_-RedHats.git
cd Assignment_2.1_Programming_Task_1_-RedHats
python -c "from numcompute import stats; print(stats.mean([1, 2, 3]))"   # the repo root is on sys.path
```

From another folder add the repository to the path: `export PYTHONPATH=/path/to/Assignment_2.1_Programming_Task_1_-RedHats` (Windows PowerShell: `$env:PYTHONPATH="C:\path\to\Assignment_2.1_Programming_Task_1_-RedHats"`) or, in Python, `sys.path.insert(0, "/path/to/Assignment_2.1_Programming_Task_1_-RedHats")`.

## Quick start

This example is self-contained: it writes a small CSV with gaps, loads it, runs a pipeline and reads the outputs.

```python
import numpy as np
from numcompute import io, preprocessing, pipeline, metrics, rank, optim, stats, utils, sort_search

# 1. Read a CSV: blanks, NA, N/A, NaN and ? become NaN
with open("sample.csv", "w") as f:
    f.write("id,x1,x2,x3,label\n"
            "1,5.1,3.5,1.4,a\n2,4.9,,1.4,a\n3,7.0,3.2,4.7,b\n"
            "4,6.4,3.2,NA,b\n5,5.0,3.6,1.4,a\n6,6.9,3.1,4.9,b\n")

X, names = io.load_csv("sample.csv", header=True, usecols=(1, 2, 3))
y = io.load_csv("sample.csv", header=True, dtype=str, usecols=(4,))[0][:, 0]

# 2. A pipeline: fill gaps, standardise, then any object with fit/predict
class MajorityModel(pipeline.Estimator):
    """Toy model: predicts the most common training label."""
    def fit(self, X, y=None):
        labels, counts = np.unique(y, return_counts=True)
        self.label_ = labels[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.label_)

pipe = pipeline.Pipeline([
    ("impute", preprocessing.Imputer(strategy="mean")),
    ("scale", preprocessing.StandardScaler()),
    ("model", MajorityModel()),
])
pipe.fit(X, y)
pred = pipe.predict(X)

# 3. Read the outputs
metrics.accuracy(y, pred)                           # 0.5
metrics.f1(y, pred, average="macro")
metrics.confusion_matrix(y, pred)
rank.rank([10, 20, 20, 30], method="average")       # [1., 2.5, 2.5, 4.]
sort_search.topk([5, 1, 9, 3], k=2)                 # indices of the 2 largest: [2, 0]
optim.grad(lambda x: x[0] ** 2 + x[1], [1.0, 2.0])  # approx [2., 1.]
stats.describe(X, names=names)                      # count, missing, mean, std, quartiles
utils.softmax(np.array([1000.0, 1001.0]))           # stable, no overflow
```

A complete end-to-end version is in `demo/quickstart.ipynb`.

## Project structure

```text
<repository root>/
├── numcompute/
│   ├── __init__.py
│   ├── io.py              CSV reader: delimiter, header, dtype and missing-value handling
│   ├── preprocessing.py   StandardScaler, MinMaxScaler, Imputer, OneHotEncoder, ColumnTransformer
│   ├── sort_search.py     stable and multi-key sort, top-k, quickselect, binary search
│   ├── rank.py            ranking with tie rules, percentile rank, percentiles
│   ├── stats.py           descriptive statistics, histogram, quantiles, describe
│   ├── metrics.py         accuracy, precision, recall, F1, confusion matrix, MSE, ROC/AUC
│   ├── optim.py           finite-difference gradient and Jacobian
│   ├── pipeline.py        Transformer/Estimator API, Pipeline, FeatureUnion
│   ├── utils.py           distances, activations, logsumexp, top-k along an axis, batching
│   └── benchmarking.py    micro-benchmark harness (timing, correctness gate, tables)
├── benchmarks/            run_benchmarks.py, baselines.py (pure-Python loop versions), results/
├── tests/                 unit tests (pytest)
├── demo/
│   └── quickstart.ipynb   end-to-end demo
├── README.md
└── pyproject.toml
```

## API overview

Signatures below are the public API. Every function documents its parameters, shapes, exceptions and complexity in its docstring (for example `help(numcompute.stats.quantile)`).

| Module | Public API |
|---|---|
| `io` | `load_csv(path, delimiter=",", header=False, missing="fill", fill_value=None, missing_values=None, dtype=float, usecols=None)` returns a 2-D array (and the column names when `header=True`). `missing` is `"fill"` or `"skip"`. |
| `preprocessing` | `StandardScaler(with_mean=True, with_std=True)`, `MinMaxScaler(feature_range=(0.0, 1.0), clip=False)`, `Imputer(strategy="mean", fill_value=0.0)` with strategies `mean`, `constant`, `most_frequent`, `OneHotEncoder(categories=None, handle_unknown="ignore")`, `ColumnTransformer(numeric, categorical, numeric_cols, categorical_cols)`. All have `fit`, `partial_fit` (alias `update`), `transform`, `fit_transform`, `reset`. `OneHotEncoder` also has `freeze()`. |
| `sort_search` | `stable_sort(values, ascending=True, return_indices=False)`, `multi_key_sort(keys, ascending=True)`, `topk(values, k, largest=True, return_indices=True)`, `quickselect(values, k)`, `binary_search(values, x, return_index=True)` |
| `rank` | `rank(data, method="average", axis=-1, descending=False)` with methods `average`, `min`, `max`, `dense`, `ordinal`; `percentile_rank(data, axis=-1, descending=False)`; `percentile(data, q, interpolation="linear", axis=-1)` with `linear`, `lower`, `higher`, `midpoint` |
| `stats` | `count`, `mean`, `var`, `std`, `min`, `max`, `median`, `quantile(data, q, axis=None, interpolation="linear")`, `mode`, `histogram(data, bins=10, range=None, density=False)`, `describe(data, names=None)`, `describe_categorical`, `format_summary`, `format_histogram` |
| `metrics` | `accuracy`, `precision`, `recall`, `f1` (with `average` of `binary`, `macro`, `micro`, `weighted` or `None`), `confusion_matrix`, `mse`, `roc_curve`, `auc`, `roc_auc_score` |
| `optim` | `grad(f, x, h=1e-5, method="central")` and `jacobian(F, x, h=1e-5, method="central")` with `method` of `central` (error O(h^2)) or `forward` (error O(h)) |
| `pipeline` | `Transformer` and `Estimator` base classes, `Pipeline(steps)` (`fit`, `transform`, `fit_transform`, `predict`, `named_steps`), `FeatureUnion(transformer_list)` |
| `utils` | `pairwise_distances(X, Y=None, metric="euclidean", batch_size=None)` with `euclidean`, `sqeuclidean`, `manhattan`, `cosine`; `relu`, `leaky_relu`, `sigmoid`, `softplus`, `logsumexp`, `softmax`, `log_softmax`; `topk_along_axis`; `batch_slices`, `iter_batches`; validation helpers `check_axis`, `as_float_array` |

## Conventions

- **Shapes.** Feature matrices are `(n_samples, n_features)`. 1-D input is accepted where a single vector makes sense and documented per function.
- **`axis`.** Follows NumPy: an integer (negative counts from the end), or `None` to use all elements where stated.
- **NaN.** NaN means "missing". `stats` and `preprocessing` ignore it when fitting; `sort_search` ignores it in top-k, quickselect and binary search with a `UserWarning` and places it last when sorting; `utils.logsumexp/softmax/log_softmax` take `nan_policy` (`propagate` or `omit`); `rank.rank` and the `metrics` reject NaN with a `ValueError`. Infinity is rejected where it would make a result meaningless.
- **Ties.** Sorting is stable. `topk` returns tied values in original order and prefers earlier positions at the cut-off. `rank` offers five tie rules.
- **Percentile scales.** `stats.quantile` takes `q` in `[0, 1]`; `rank.percentile` takes `q` in `[0, 100]`.
- **Errors.** Wrong types raise `TypeError`, wrong shapes or values raise `ValueError`, and using an estimator before fitting raises `RuntimeError`. Messages say what was expected and what was received.
- **Streaming.** Preprocessing classes learn chunk by chunk with `partial_fit`; the result equals fitting on all the data at once.
- **Reproducibility.** Anything random takes a `seed`; the benchmarks and the notebook are seeded.

## Running the demo notebook

1. Install the package (see [Installation](#installation)) and Jupyter: `pip install jupyter`.
2. Start Jupyter with `jupyter notebook` and open `demo/quickstart.ipynb` (VS Code and Google Colab also work).
3. Choose `Kernel > Restart & Run All`.

The notebook reads a CSV that contains missing values with `io`, preprocesses it with `StandardScaler`, `Imputer` and `OneHotEncoder`, demonstrates `sort_search` (top-k, quickselect, binary search) and `rank` (tie handling, percentiles), computes `stats` (mean, variance, histogram, quantiles), estimates gradients and Jacobians with `optim`, and ends with a loop-versus-vectorised benchmark table.

## Running the tests

```bash
pip install pytest
python -m pytest tests -v
```

The suite has **105 unit tests**:

| Test file | Module | Tests |
|---|---|---:|
| `tests/optim_tests.py` | `optim` | 10 |
| `tests/sort_search_tests.py` | `sort_search` | 18 |
| `tests/stats_tests.py` | `stats` | 15 |
| `tests/test_preprocessing.py` | `preprocessing` | 36 |
| `tests/test_rank.py` | `rank` | 26 |

Both file-name patterns (`test_*.py` and `*_tests.py`) are collected through the `python_files` setting in `pyproject.toml`. The tests cover empty arrays, all-equal values, duplicates and ties, extreme `k`, NaN, infinity, non-contiguous (strided) arrays, integer and text input, streaming-equals-batch checks, comparison against NumPy reference results, and invalid inputs.

## Running the benchmarks

```bash
python benchmarks/run_benchmarks.py --quick                  # small sizes, a few seconds
python benchmarks/run_benchmarks.py                          # full sizes
python benchmarks/run_benchmarks.py --only rank --repeat 7 --seed 1
```

Options: `--quick`, `--repeat N` (timed repeats, default 5), `--seed S` (data seed, default 0), `--only TEXT` (cases whose name contains TEXT), `--out DIR` (default `benchmarks/results`).

For every case the pure-Python baseline (`benchmarks/baselines.py`) and the NumCompute function run on identical seeded data. The script first checks that both give the same answer, then reports the median time of each and the speed-up, together with environment notes (Python, NumPy, platform, CPU count, seed). Results are written to `benchmarks/results/benchmark_results.csv` and `.md`. The script exits with status 0 only if every result matched.

## Performance summary

Median time over 5 repeats, seed 0, largest size run for each case (the full table with every size is in `benchmarks/results/benchmark_results.md`). Environment: Python 3.12.3, NumPy 2.4.4, Linux x86_64, 1 CPU core.

| Case | Baseline | n | Loop (ms) | Vectorised (ms) | Speed-up |
|---|---|---:|---:|---:|---:|
| StandardScaler | python loops | 100,000 | 200.35 | 17.59 | 11.4x |
| MinMaxScaler | python loops | 100,000 | 150.81 | 19.69 | 7.7x |
| OneHotEncoder | dict + loop | 100,000 | 30.71 | 34.85 | 0.9x |
| rank (average, ties) | sort + tie loop | 100,000 | 43.23 | 11.68 | 3.7x |
| percentile (5 levels) | sorted list | 100,000 | 15.92 | 0.82 | 19.4x |
| topk (k=10) | heapq.nlargest | 1,000,000 | 9.79 | 9.17 | 1.1x |
| confusion_matrix (k=5) | dict counting | 1,000,000 | 61.33 | 45.87 | 1.3x |
| roc_auc_score (ties) | pairwise O(P*N) | 3,000 | 56.39 | 0.45 | 124.6x |
| mean + var (5% NaN) | python loops | 100,000 | 149.55 | 17.66 | 8.5x |
| histogram (20 bins) | bisect per element | 1,000,000 | 229.74 | 34.49 | 6.7x |
| grad (central) | python-list f | 2,000 | 221.01 | 26.28 | 8.4x |
| binary_search (100 queries) | linear scan | 100,000 | 151.67 | 81.46 | 1.9x |

Timings depend on your machine; the data does not. Very small inputs can show a speed-up below 1x because each NumPy call has a fixed overhead, and the advantage grows with size. Where the baseline is already efficient (`heapq.nlargest`, dictionary counting), the gain is modest. `OneHotEncoder` also performs input validation, category tracking and streaming support that the simple baseline does not.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'numcompute'` | The package is not on the import path. Install it with `pip install -e .` from the repository root, or run from the repository root, or set `PYTHONPATH` (see [Installation](#installation)). Also check that the right virtual environment is active. |
| `ImportError` for `numpy` | `pip install "numpy>=1.22"` in the active environment. |
| `pytest: command not found` | `pip install pytest`, then run `python -m pytest tests -v`. |
| `source .venv/bin/activate` fails on Windows | Use `.venv\Scripts\activate` (Command Prompt) or `.venv\Scripts\Activate.ps1` (PowerShell). |
| `from numcompute.stats import *` hides `min`/`max` | `stats.min` and `stats.max` shadow the Python built-ins. Import the module (`from numcompute import stats`) instead of star-importing. |
| A benchmark speed-up is below 1x | Expected for very small inputs (fixed NumPy call overhead). Compare the larger sizes, or run without `--quick`. |

