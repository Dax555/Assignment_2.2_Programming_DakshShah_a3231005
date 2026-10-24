"""
run_benchmarks.py
=================
Reproducible loop-vs-vectorised benchmarks for NumCompute.

Run from the project root (the folder that contains the package)::

    python benchmarks/run_benchmarks.py            # full run
    python benchmarks/run_benchmarks.py --quick    # small sizes, a few seconds
    python benchmarks/run_benchmarks.py --only rank --repeat 7 --seed 1

For every case the pure-Python baseline (benchmarks/baselines.py) and the
NumCompute function run on identical data. The script first checks that the
two results agree, then reports the median time of each and the speed-up.
It prints the environment notes and a Markdown table, and writes
``results/benchmark_results.csv`` and ``results/benchmark_results.md``.

Exit status is 0 if every result matched and no case failed, otherwise 1.

Conventions
-----------
- Data is generated with ``numpy.random.default_rng(seed)``; the same seed
  gives the same data on every machine (timings will of course differ).
- Conversions between lists and arrays happen when the data is built, not
  inside the timed region.
- n is the problem size (rows or elements) shown in the table.

Only the standard library and NumPy are used.
"""

import argparse
import inspect
import os
import sys
from collections import namedtuple
from importlib import import_module

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baselines as bl  # noqa: E402  (needs the sys.path edit above)
import harness  # noqa: E402  (local copy of the timing harness)

Case = namedtuple(
    "Case", "name baseline sizes quick_sizes make loop vec same"
)


def _import_package():
    """
    Import the NumCompute package under whichever name the repo uses.

    Parameters
    ----------
    None

    Returns
    -------
    module
        The imported package ('numcompute', else 'numCompute').

    Raises
    ------
    ImportError
        If neither name can be imported from the repository root.

    Notes
    -----
    Time: O(1) after the first import. Space: O(1).
    """
    for name in ("numcompute", "numCompute"):
        try:
            return import_module(name)
        except ImportError:
            continue
    raise ImportError(
        f"could not import 'numcompute' or 'numCompute' from {_ROOT}; "
        "run this script from the repository root"
    )


def _load_harness(pkg):
    """
    Return the timing harness: the package's own, else the local copy.

    Parameters
    ----------
    pkg : module
        The imported NumCompute package.

    Returns
    -------
    module
        An object providing ``compare``, ``environment_info``,
        ``format_table`` and ``write_csv``. ``<pkg>.benchmarking`` is used if
        it defines them, otherwise ``benchmarks/harness.py``.

    Raises
    ------
    None

    Notes
    -----
    This lets the benchmarks run before ``benchmarking.py`` is filled in.

    Time: O(1). Space: O(1).
    """
    try:
        mod = import_module(pkg.__name__ + ".benchmarking")
    except ImportError:
        return harness
    needed = ("compare", "environment_info", "format_table", "write_csv")
    return mod if all(hasattr(mod, name) for name in needed) else harness


def _close(a, b):
    """
    NaN-aware closeness test between two array-likes.

    Parameters
    ----------
    a, b : array_like
        Results to compare; they are converted with ``np.asarray``.

    Returns
    -------
    bool
        True if shapes match and all values agree to rtol=1e-6, atol=1e-9.

    Raises
    ------
    None

    Notes
    -----
    Time: O(N) for N elements. Space: O(N).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(
        np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
    )


def _equal(a, b):
    """
    Exact equality test between two array-likes.

    Parameters
    ----------
    a, b : array_like
        Results to compare.

    Returns
    -------
    bool
        True if ``np.asarray(a)`` and ``np.asarray(b)`` are identical.

    Raises
    ------
    None

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _same_mean_var(a, b):
    """
    Compare (means, variances) pairs from the loop and vectorised versions.

    Parameters
    ----------
    a, b : tuple of array_like
        Each is ``(means, variances)``.

    Returns
    -------
    bool
        True if both the means and the variances agree (NaN-aware).

    Raises
    ------
    None

    Notes
    -----
    Time: O(d). Space: O(d).
    """
    return _close(a[0], b[0]) and _close(a[1], b[1])


def _make_matrix(n, rng):
    """
    Random (n, 8) float matrix, as a list of rows and as an array.

    Parameters
    ----------
    n : int
        Number of rows.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    tuple (list of list of float, numpy.ndarray of shape (n, 8))

    Raises
    ------
    None

    Notes
    -----
    Time: O(n). Space: O(n).
    """
    X = rng.normal(5.0, 3.0, size=(n, 8))
    return X.tolist(), X


def _make_matrix_nan(n, rng):
    """
    Random (n, 8) float matrix with about 5 % NaN.

    Parameters
    ----------
    n : int
        Number of rows.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    tuple (list of list of float, numpy.ndarray of shape (n, 8))

    Raises
    ------
    None

    Notes
    -----
    Time: O(n). Space: O(n).
    """
    X = rng.normal(5.0, 3.0, size=(n, 8))
    X[rng.random(X.shape) < 0.05] = np.nan
    return X.tolist(), X


def _make_vector(n, rng):
    """
    Random standard-normal vector of length n.

    Parameters
    ----------
    n : int
        Length.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    tuple (list of float, numpy.ndarray of shape (n,))

    Raises
    ------
    None

    Notes
    -----
    Time: O(n). Space: O(n).
    """
    x = rng.normal(size=n)
    return x.tolist(), x


def build_cases(pkg):
    """
    Build the list of benchmark cases for the installed package.

    Parameters
    ----------
    pkg : module
        The imported NumCompute package.

    Returns
    -------
    list of Case
        Each Case holds a name, baseline description, full and quick problem
        sizes, a data factory ``make(n, rng) -> tuple``, the loop and
        vectorised callables (both take that tuple unpacked) and a result
        comparison function.

    Raises
    ------
    ImportError
        If a NumCompute submodule cannot be imported.

    Notes
    -----
    The small lambdas only adapt the shared argument tuple to each side; the
    real work is in baselines.py and in the package.

    Time: O(1) to build. Space: O(1).
    """
    pre = import_module(pkg.__name__ + ".preprocessing")
    rk = import_module(pkg.__name__ + ".rank")
    ss = import_module(pkg.__name__ + ".sort_search")
    mt = import_module(pkg.__name__ + ".metrics")
    st = import_module(pkg.__name__ + ".stats")
    op = import_module(pkg.__name__ + ".optim")

    def make_onehot(n, rng):
        cats = rng.choice(["red", "green", "blue", "cyan", "pink"], size=n)
        return cats.tolist(), cats.reshape(-1, 1)

    def make_ranks(n, rng):
        x = rng.integers(0, max(n // 4, 2), size=n).astype(float)  # many ties
        return x.tolist(), x

    def make_search(n, rng):
        a = np.sort(rng.normal(size=n))
        q = rng.normal(size=100).tolist()
        return a.tolist(), a, q

    def make_labels(n, rng):
        yt = rng.integers(0, 5, size=n)
        flip = rng.random(n) < 0.3
        yp = np.where(flip, rng.integers(0, 5, size=n), yt)
        return yt.tolist(), yp.tolist(), yt, yp

    def make_scores(n, rng):
        y = rng.integers(0, 2, size=n)
        s = np.round(rng.normal(size=n) + 0.8 * y, 1)  # rounding creates ties
        return y.tolist(), s.tolist(), y, s

    def make_hist(n, rng):
        x = rng.normal(size=n)
        edges = np.linspace(x.min(), x.max(), 21)
        return x.tolist(), x, edges.tolist()

    def f_list(v):
        return sum(t * t for t in v)

    def f_array(v):
        return float(np.dot(v, v))

    qs = [5, 25, 50, 75, 95]
    cases = [
        Case("StandardScaler", "python loops", (1_000, 10_000, 100_000), (500, 2_000),
             _make_matrix,
             lambda lst, arr: bl.loop_standard_scale(lst),
             lambda lst, arr: pre.StandardScaler().fit_transform(arr), _close),
        Case("MinMaxScaler", "python loops", (1_000, 10_000, 100_000), (500, 2_000),
             _make_matrix,
             lambda lst, arr: bl.loop_minmax_scale(lst),
             lambda lst, arr: pre.MinMaxScaler().fit_transform(arr), _close),
        Case("OneHotEncoder", "dict + loop", (1_000, 10_000, 100_000), (500, 2_000),
             make_onehot,
             lambda lst, arr: bl.loop_one_hot(lst),
             lambda lst, arr: pre.OneHotEncoder().fit_transform(arr), _close),
        Case("rank (average, ties)", "sort + tie loop", (1_000, 10_000, 100_000), (500, 2_000),
             make_ranks,
             lambda lst, arr: bl.loop_rank_average(lst),
             lambda lst, arr: rk.rank(arr, "average"), _close),
        Case("percentile (5 levels)", "sorted list", (1_000, 10_000, 100_000), (500, 2_000),
             _make_vector,
             lambda lst, arr: bl.loop_percentile(lst, qs),
             lambda lst, arr: rk.percentile(arr, qs), _close),
        Case("topk (k=10)", "heapq.nlargest", (1_000, 100_000, 1_000_000), (1_000, 20_000),
             _make_vector,
             lambda lst, arr: bl.loop_topk_values(lst, 10),
             lambda lst, arr: ss.topk(arr, 10, return_indices=False), _close),
        Case("confusion_matrix (k=5)", "dict counting", (1_000, 100_000, 1_000_000), (1_000, 20_000),
             make_labels,
             lambda yt, yp, at, ap: bl.loop_confusion_matrix(yt, yp, [0, 1, 2, 3, 4]),
             lambda yt, yp, at, ap: mt.confusion_matrix(at, ap, labels=[0, 1, 2, 3, 4]), _equal),
        Case("roc_auc_score (ties)", "pairwise O(P*N)", (200, 1_000, 3_000), (200, 500),
             make_scores,
             lambda yl, sl, ya, sa: bl.loop_roc_auc(yl, sl),
             lambda yl, sl, ya, sa: mt.roc_auc_score(ya, sa), _close),
        Case("mean+var (5% NaN)", "python loops", (1_000, 10_000, 100_000), (500, 2_000),
             _make_matrix_nan,
             lambda lst, arr: bl.loop_mean_var_nan(lst),
             lambda lst, arr: (st.mean(arr, axis=0), st.var(arr, axis=0)), _same_mean_var),
        Case("histogram (20 bins)", "bisect per element", (1_000, 100_000, 1_000_000), (1_000, 20_000),
             make_hist,
             lambda lst, arr, ed: bl.loop_histogram(lst, ed),
             lambda lst, arr, ed: st.histogram(arr, bins=20)[0], _equal),
        Case("grad (central, n vars)", "python-list f", (100, 500, 2_000), (50, 200),
             _make_vector,
             lambda lst, arr: bl.loop_grad_central(f_list, lst),
             lambda lst, arr: op.grad(f_array, arr),
             lambda a, b: bool(np.allclose(a, b, rtol=1e-4, atol=1e-4))),
    ]

    # Works with both signatures: binary_search(values, x) (sorts internally)
    # and binary_search(sorted_array, x, method=...).
    has_method = "method" in inspect.signature(ss.binary_search).parameters

    def lib_search(arr, x):
        if has_method:
            return ss.binary_search(arr, x, method="searchsorted")
        return ss.binary_search(arr, x)

    cases.append(
        Case("binary_search (100 queries)", "linear scan",
             (1_000, 10_000, 100_000), (500, 2_000), make_search,
             lambda lst, arr, q: bl.loop_linear_search(lst, q),
             lambda lst, arr, q: [lib_search(arr, x) for x in q], _equal)
    )
    if has_method:
        cases.append(
            Case("binary_search method=loop vs searchsorted", "library loop",
                 (10_000, 1_000_000), (1_000, 10_000), make_search,
                 lambda lst, arr, q: [ss.binary_search(arr, x, method="loop") for x in q],
                 lambda lst, arr, q: [ss.binary_search(arr, x, method="searchsorted") for x in q],
                 _equal)
        )
    return cases


def _parse_args(argv):
    """
    Parse command-line options.

    Parameters
    ----------
    argv : list of str or None
        Arguments without the program name; None uses ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Fields: quick (bool), repeat (int), seed (int), only (str or None),
        out (str).

    Raises
    ------
    SystemExit
        On invalid options (argparse behaviour).

    Notes
    -----
    Time: O(len(argv)). Space: O(1).
    """
    p = argparse.ArgumentParser(description="NumCompute loop-vs-vectorised benchmarks")
    p.add_argument("--quick", action="store_true", help="use small sizes")
    p.add_argument("--repeat", type=int, default=5, help="timed batches per side (default 5)")
    p.add_argument("--seed", type=int, default=0, help="data seed (default 0)")
    p.add_argument("--only", default=None, help="run cases whose name contains this text")
    p.add_argument("--out", default=os.path.join(_HERE, "results"),
                   help="output folder (default benchmarks/results)")
    return p.parse_args(argv)


def main(argv=None):
    """
    Run the selected benchmarks, print the table and save the results.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 if every case ran and its results matched, otherwise 1.

    Raises
    ------
    ImportError
        If the NumCompute package cannot be found.
    ValueError
        If ``--repeat`` is below 1.

    Notes
    -----
    A case that raises is reported and skipped; the others still run.

    Time: dominated by the baselines (O(n d) up to O(P N) per case).
    Space: O(n d) for the largest generated dataset.
    """
    args = _parse_args(argv)
    pkg = _import_package()
    bench = _load_harness(pkg)

    cases = build_cases(pkg)
    if args.only:
        cases = [c for c in cases if args.only.lower() in c.name.lower()]
        if not cases:
            print(f"no case name contains {args.only!r}")
            return 1

    rows, failed = [], False
    for case in cases:
        for n in (case.quick_sizes if args.quick else case.sizes):
            rng = np.random.default_rng(args.seed)  # same data for loop and vec
            data = case.make(n, rng)
            try:
                row = bench.compare(case.name, n, case.loop, case.vec, data,
                                    case.same, repeat=args.repeat,
                                    baseline=case.baseline)
            except Exception as exc:  # report and keep going
                print(f"FAILED {case.name} n={n}: {type(exc).__name__}: {exc}")
                failed = True
                continue
            failed = failed or not row["ok"]
            rows.append(row)
            print(f"  {case.name:<42} n={n:>9,}  loop {row['loop_ms']:>10.3f} ms  "
                  f"vec {row['vec_ms']:>9.3f} ms  {row['speedup']:>7.1f}x  "
                  f"{'ok' if row['ok'] else 'MISMATCH'}", flush=True)

    if not rows:
        return 1
    env = bench.environment_info(args.seed)
    table = bench.format_table(rows, env)
    print("\n" + table)
    os.makedirs(args.out, exist_ok=True)
    bench.write_csv(rows, os.path.join(args.out, "benchmark_results.csv"))
    with open(os.path.join(args.out, "benchmark_results.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nsaved results to {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
