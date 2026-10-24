"""
baselines.py
============
Deliberately naive pure-Python implementations (plain loops, lists, the
standard library) used as the "loop" side of the benchmarks. They are
reference baselines, not part of the toolkit: they exist so that the
vectorised NumCompute functions can be timed against, and checked against,
straightforward code.

Conventions
-----------
- Inputs are plain Python lists (converted once, outside the timed region).
- Results are lists; the benchmark converts them with ``np.asarray`` before
  comparing, which is also outside the timed region.
- n is the number of rows or elements, d the number of columns.

Only the standard library is used.
"""

import bisect
import heapq
import math

__all__ = [
    "loop_standard_scale",
    "loop_minmax_scale",
    "loop_one_hot",
    "loop_rank_average",
    "loop_percentile",
    "loop_topk_values",
    "loop_linear_search",
    "loop_confusion_matrix",
    "loop_roc_auc",
    "loop_mean_var_nan",
    "loop_histogram",
    "loop_grad_central",
]


def loop_standard_scale(rows):
    """
    Z-score every column with explicit loops (population variance).

    Parameters
    ----------
    rows : list of list of float, shape (n, d)
        Data without NaN.

    Returns
    -------
    list of list of float, shape (n, d)
        ``(x - mean) / std`` per column; a zero-variance column maps to 0.

    Raises
    ------
    ZeroDivisionError
        If `rows` is empty.

    Notes
    -----
    Time: O(n d). Space: O(n d).
    """
    n, d = len(rows), len(rows[0])
    out = [[0.0] * d for _ in range(n)]
    for j in range(d):
        total = 0.0
        for i in range(n):
            total += rows[i][j]
        mean = total / n
        sq = 0.0
        for i in range(n):
            sq += (rows[i][j] - mean) ** 2
        std = math.sqrt(sq / n) or 1.0
        for i in range(n):
            out[i][j] = (rows[i][j] - mean) / std
    return out


def loop_minmax_scale(rows):
    """
    Scale every column to [0, 1] with explicit loops.

    Parameters
    ----------
    rows : list of list of float, shape (n, d)
        Data without NaN.

    Returns
    -------
    list of list of float, shape (n, d)
        ``(x - min) / (max - min)`` per column; a constant column maps to 0.

    Raises
    ------
    ZeroDivisionError
        If `rows` is empty.

    Notes
    -----
    Time: O(n d). Space: O(n d).
    """
    n, d = len(rows), len(rows[0])
    out = [[0.0] * d for _ in range(n)]
    for j in range(d):
        lo = hi = rows[0][j]
        for i in range(1, n):
            v = rows[i][j]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        span = (hi - lo) or 1.0
        for i in range(n):
            out[i][j] = (rows[i][j] - lo) / span
    return out


def loop_one_hot(values):
    """
    One-hot encode one categorical column with a dict lookup per element.

    Parameters
    ----------
    values : list of str, length n
        Category of each sample.

    Returns
    -------
    list of list of float, shape (n, k)
        Columns follow the sorted unique categories (k of them).

    Raises
    ------
    None

    Notes
    -----
    Time: O(n k) (each row is a length-k list). Space: O(n k).
    """
    cats = sorted(set(values))
    index = {c: i for i, c in enumerate(cats)}
    out = []
    for v in values:
        row = [0.0] * len(cats)
        row[index[v]] = 1.0
        out.append(row)
    return out


def loop_rank_average(values):
    """
    Average ranks (1-based, ties share the mean rank) with a tie-group loop.

    Parameters
    ----------
    values : list of float, length n
        Data without NaN.

    Returns
    -------
    list of float, length n
        Rank of each element.

    Raises
    ------
    None

    Notes
    -----
    Time: O(n log n). Space: O(n).
    """
    n = len(values)
    order = sorted(range(n), key=values.__getitem__)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def loop_percentile(values, qs):
    """
    Linear-interpolation percentiles from a sorted list.

    Parameters
    ----------
    values : list of float, length n
        Data without NaN, n >= 1.
    qs : list of float
        Percentile levels in [0, 100].

    Returns
    -------
    list of float, length len(qs)
        The percentile for each level.

    Raises
    ------
    IndexError
        If `values` is empty.

    Notes
    -----
    Time: O(n log n + len(qs)). Space: O(n).
    """
    s = sorted(values)
    n = len(s)
    out = []
    for q in qs:
        pos = q / 100.0 * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        out.append(s[lo] + (s[hi] - s[lo]) * (pos - lo))
    return out


def loop_topk_values(values, k):
    """
    The k largest values, largest first, using a heap.

    Parameters
    ----------
    values : list of float, length n
        Data without NaN.
    k : int
        Number of values to return.

    Returns
    -------
    list of float, length min(k, n)
        Descending order.

    Raises
    ------
    None

    Notes
    -----
    Time: O(n log k). Space: O(k).
    """
    return heapq.nlargest(k, values)


def loop_linear_search(sorted_values, queries):
    """
    Leftmost insertion index of each query by scanning from the left.

    Parameters
    ----------
    sorted_values : list of float, length n
        Data sorted ascending.
    queries : list of float
        Values to look up.

    Returns
    -------
    list of tuple (int, bool)
        ``(insertion_index, found)`` for each query.

    Raises
    ------
    None

    Notes
    -----
    Time: O(n q) for q queries. Space: O(q).
    """
    n = len(sorted_values)
    out = []
    for x in queries:
        i = 0
        while i < n and sorted_values[i] < x:
            i += 1
        out.append((i, i < n and sorted_values[i] == x))
    return out


def loop_confusion_matrix(y_true, y_pred, labels):
    """
    Confusion matrix by counting each (true, predicted) pair in a loop.

    Parameters
    ----------
    y_true, y_pred : list of int, length n
        True and predicted labels.
    labels : list of int, length k
        Class order of the rows and columns.

    Returns
    -------
    list of list of int, shape (k, k)
        ``cm[i][j]`` = samples of class labels[i] predicted as labels[j].

    Raises
    ------
    KeyError
        If a label is not in `labels`.

    Notes
    -----
    Time: O(n + k^2). Space: O(k^2).
    """
    index = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[index[t]][index[p]] += 1
    return cm


def loop_roc_auc(y_true, scores):
    """
    ROC AUC as the probability that a random positive outranks a negative.

    Parameters
    ----------
    y_true : list of int, length n
        Labels, 1 for positive and 0 for negative.
    scores : list of float, length n
        Higher means more likely positive.

    Returns
    -------
    float
        Pairwise (Mann-Whitney) AUC; tied pairs count 0.5.

    Raises
    ------
    ZeroDivisionError
        If one class is missing.

    Notes
    -----
    Time: O(P N) for P positives and N negatives. Space: O(n).
    """
    pos = [s for t, s in zip(y_true, scores) if t == 1]
    neg = [s for t, s in zip(y_true, scores) if t == 0]
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def loop_mean_var_nan(rows):
    """
    Column mean and population variance, skipping NaN, with explicit loops.

    Parameters
    ----------
    rows : list of list of float, shape (n, d)
        Data where NaN means missing.

    Returns
    -------
    means, variances : list of float, each of length d
        NaN for a column with no valid value.

    Raises
    ------
    None

    Notes
    -----
    Time: O(n d). Space: O(d).
    """
    d = len(rows[0])
    means, variances = [], []
    for j in range(d):
        total, count = 0.0, 0
        for row in rows:
            v = row[j]
            if v == v:  # False only for NaN
                total += v
                count += 1
        if count == 0:
            means.append(float("nan"))
            variances.append(float("nan"))
            continue
        m = total / count
        sq = 0.0
        for row in rows:
            v = row[j]
            if v == v:
                sq += (v - m) ** 2
        means.append(m)
        variances.append(sq / count)
    return means, variances


def loop_histogram(values, edges):
    """
    Histogram counts with one binary search per element.

    Parameters
    ----------
    values : list of float, length n
        Data without NaN.
    edges : list of float, length B + 1
        Strictly increasing bin edges. Bins are [a, b) except the last,
        which also contains its right edge.

    Returns
    -------
    list of int, length B
        Count per bin; values outside the edges are not counted.

    Raises
    ------
    None

    Notes
    -----
    Time: O(n log B). Space: O(B).
    """
    n_bins = len(edges) - 1
    counts = [0] * n_bins
    for v in values:
        if edges[0] <= v <= edges[-1]:
            idx = bisect.bisect_right(edges, v) - 1
            counts[min(idx, n_bins - 1)] += 1
    return counts


def loop_grad_central(f, x, h=1e-5):
    """
    Central-difference gradient where `f` works on plain Python lists.

    Parameters
    ----------
    f : callable
        Scalar function of a list of floats.
    x : list of float, length n
        Point at which to differentiate.
    h : float, default 1e-5
        Step size (> 0).

    Returns
    -------
    list of float, length n
        ``(f(x + h e_j) - f(x - h e_j)) / (2 h)`` for each j.

    Raises
    ------
    None

    Notes
    -----
    2n calls to `f`; with an O(n) `f` the total is Time O(n^2), Space O(n).
    """
    out = []
    for j in range(len(x)):
        xp = list(x)
        xm = list(x)
        xp[j] += h
        xm[j] -= h
        out.append((f(xp) - f(xm)) / (2.0 * h))
    return out
