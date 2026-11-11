"""
benchmarking.py
===============
Micro-benchmark harness for comparing a vectorised implementation with a
Python-loop baseline: timing, correctness gate, environment notes and table
output. Uses only the standard library and NumPy.

Conventions
-----------
- All times are wall-clock seconds per call, measured with
  ``time.perf_counter`` after a warm-up call.
- Every comparison first checks that the loop and vectorised results agree;
  a speed-up is only meaningful if both compute the same thing.
- Time/space complexity is given for the harness itself; the cost of the
  user's functions is written C_f.
"""

import csv
import gc
import os
import platform
import sys
import time

import numpy as np

__all__ = [
    "time_callable",
    "environment_info",
    "compare",
    "format_table",
    "write_csv",
]


def _check_positive_int(value, name, minimum):
    """
    Validate that `value` is an integer >= `minimum`.

    Parameters
    ----------
    value : int
        Value to check. Booleans are rejected.
    name : str
        Argument name used in the error message.
    minimum : int
        Smallest allowed value.

    Returns
    -------
    int
        `value` as a Python int.

    Raises
    ------
    TypeError
        If `value` is not an integer.
    ValueError
        If `value` is smaller than `minimum`.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def time_callable(fn, *args, repeat=5, number=1, warmup=1):
    """
    Time ``fn(*args)`` and summarise the per-call duration.

    Parameters
    ----------
    fn : callable
        Function to time.
    *args
        Positional arguments passed to `fn` on every call. They are not
        copied, so `fn` must not modify them.
    repeat : int, default 5
        Number of timed batches (>= 1).
    number : int, default 1
        Calls per batch (>= 1). Each batch time is divided by `number`.
    warmup : int, default 1
        Untimed calls before measuring (>= 0), to fill caches and imports.

    Returns
    -------
    dict
        Keys 'min', 'median', 'mean', 'std' (seconds per call, floats) and
        'repeat' (int). 'std' is the population standard deviation of the
        `repeat` batch times.

    Raises
    ------
    TypeError
        If `fn` is not callable or `repeat`, `number`, `warmup` is not an
        integer.
    ValueError
        If `repeat` or `number` is below 1, or `warmup` is negative.

    Notes
    -----
    The median is the headline figure because it is robust to one-off
    slowdowns; the minimum is the best-case estimate.

    Time: O((warmup + repeat * number) * C_f). Space: O(repeat).
    """
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")
    repeat = _check_positive_int(repeat, "repeat", 1)
    number = _check_positive_int(number, "number", 1)
    warmup = _check_positive_int(warmup, "warmup", 0)

    for _ in range(warmup):
        fn(*args)
    samples = np.empty(repeat)
    for r in range(repeat):
        gc.collect()
        start = time.perf_counter()
        for _ in range(number):
            fn(*args)
        samples[r] = (time.perf_counter() - start) / number
    return {
        "min": float(samples.min()),
        "median": float(np.median(samples)),
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "repeat": repeat,
    }


def environment_info(seed=None):
    """
    Collect the environment notes that make a benchmark reproducible.

    Parameters
    ----------
    seed : int, optional
        Random seed used to generate the benchmark data; recorded as-is.

    Returns
    -------
    dict of str to str
        Python version, NumPy version, platform string, processor, logical
        CPU count and the seed.

    Raises
    ------
    None

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": str(os.cpu_count()),
        "seed": "n/a" if seed is None else str(seed),
    }


def compare(name, n, loop_fn, vec_fn, args, same, repeat=5, baseline="python loop"):
    """
    Compare a loop baseline with a vectorised implementation on one input.

    Parameters
    ----------
    name : str
        Case name shown in the table.
    n : int
        Problem size shown in the table (for example the number of rows).
    loop_fn : callable
        Baseline: ``loop_fn(*args)``.
    vec_fn : callable
        Vectorised implementation: ``vec_fn(*args)``.
    args : tuple
        Arguments shared by both functions; they must not be modified.
    same : callable
        ``same(loop_result, vec_result) -> bool``; True if the two results
        agree (use ``np.allclose`` or ``np.array_equal`` inside).
    repeat : int, default 5
        Timed batches per implementation, as in `time_callable`.
    baseline : str, default 'python loop'
        Short description of what `loop_fn` is, shown in the table.

    Returns
    -------
    dict
        Keys 'case', 'n', 'baseline', 'loop_ms', 'vec_ms', 'speedup'
        (loop median / vectorised median) and 'ok' (bool from `same`).

    Raises
    ------
    TypeError
        If `loop_fn`, `vec_fn` or `same` is not callable, or `args` is not
        a tuple.
    ValueError
        Propagated from `time_callable` for a bad `repeat`.

    Notes
    -----
    Both functions run once before timing to check agreement, so a wrong
    vectorised implementation is flagged (ok=False) rather than timed
    silently.

    Time: O(C_loop + C_vec) times the number of runs. Space: O(1) extra.
    """
    for label, fn in (("loop_fn", loop_fn), ("vec_fn", vec_fn), ("same", same)):
        if not callable(fn):
            raise TypeError(f"{label} must be callable, got {type(fn).__name__}")
    if not isinstance(args, tuple):
        raise TypeError(f"args must be a tuple, got {type(args).__name__}")

    ok = bool(same(loop_fn(*args), vec_fn(*args)))
    t_loop = time_callable(loop_fn, *args, repeat=repeat)
    t_vec = time_callable(vec_fn, *args, repeat=repeat)
    return {
        "case": name,
        "n": int(n),
        "baseline": baseline,
        "loop_ms": t_loop["median"] * 1e3,
        "vec_ms": t_vec["median"] * 1e3,
        "speedup": t_loop["median"] / t_vec["median"] if t_vec["median"] > 0 else float("inf"),
        "ok": ok,
    }


def format_table(rows, env=None):
    """
    Format comparison rows as a Markdown table, with optional environment notes.

    Parameters
    ----------
    rows : sequence of dict
        Output rows of `compare`.
    env : dict of str to str, optional
        Output of `environment_info`; printed as a bullet list above the
        table.

    Returns
    -------
    str
        Markdown text ready to paste into the README or report.

    Raises
    ------
    ValueError
        If `rows` is empty.
    KeyError
        If a row lacks a key produced by `compare`.

    Notes
    -----
    Time: O(R) for R rows. Space: O(R).
    """
    if len(rows) == 0:
        raise ValueError("rows must not be empty")
    lines = []
    if env:
        lines.append("**Environment**")
        lines.extend(f"- {key}: {value}" for key, value in env.items())
        lines.append("")
    lines.append("| case | n | baseline | loop (ms) | vectorised (ms) | speed-up | results match |")
    lines.append("|---|---:|---|---:|---:|---:|:---:|")
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['n']:,} | {r['baseline']} | {r['loop_ms']:.3f} | "
            f"{r['vec_ms']:.3f} | {r['speedup']:.1f}x | {'yes' if r['ok'] else 'NO'} |"
        )
    return "\n".join(lines)


def write_csv(rows, path):
    """
    Write comparison rows to a CSV file.

    Parameters
    ----------
    rows : sequence of dict
        Output rows of `compare`.
    path : str or path-like
        Destination file; its folder must already exist.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `rows` is empty.
    OSError
        If the file cannot be written.

    Notes
    -----
    Time: O(R). Space: O(1) extra.
    """
    if len(rows) == 0:
        raise ValueError("rows must not be empty")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
