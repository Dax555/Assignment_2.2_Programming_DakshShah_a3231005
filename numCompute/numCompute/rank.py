"""
rank.py
=======
Ranking and percentile utilities built using NumPy.
Functions: rank, percentile_rank, percentile.

Conventions
-----------
- Inputs are array_like with at least one dimension. The work runs along
  `axis`; the output of rank and percentile_rank has the shape of the input.
- rank and percentile_rank reject NaN with ValueError. percentile ignores NaN
  and gives NaN for a slice that has no valid value.
- No Python loop runs over rows: one stable argsort along the axis, followed
  by cumulative operations.
- Complexity is given in terms of n, the length of the axis being ranked, and
  N, the total number of elements of the input (N = n * number of slices).
"""

import numpy as np

_METHODS = ("average", "min", "max", "dense", "ordinal")
_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")


def _check(x, method, axis):
    """
    Validate the arguments of `rank` and normalise the axis.

    Parameters
    ----------
    x : array_like
        Input data.
    method : str
        Tie method; must be one of the supported names.
    axis : int
        Axis to rank along; negative values count from the end.

    Returns
    -------
    (np.ndarray, int)
        The input as an array, and `axis` as a non-negative integer.

    Raises
    ------
    ValueError
        If `method` is unknown, the input is 0-d, a float input
        contains NaN, or `axis` is out of range.

    Notes
    -----
    The NaN test is only run for floating dtypes.

    Time: O(N) for the NaN scan.
    Space: O(1) beyond the converted array.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    x = np.asarray(x)
    if x.ndim == 0:
        raise ValueError("x must be at least 1-D")
    if x.dtype.kind == "f" and np.isnan(x).any():
        raise ValueError("x contains NaN; remove or replace it first")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError(f"axis {axis} is out of range for {x.ndim}-D input")
    return x, axis % x.ndim


def _rank_last(x, method, descending):
    """
    Rank the values along the last axis of an array.

    Parameters
    ----------
    x : np.ndarray, shape (..., n)
        Array with n >= 1 along the last axis and no NaN.
    method : str
        One of "average", "min", "max", "dense", "ordinal".
    descending : bool
        If True, the largest value gets rank 1.

    Returns
    -------
    np.ndarray, same shape as `x`
        Ranks starting at 1. Float dtype for "average", integer dtype
        otherwise.

    Notes
    -----
    Sorts once with a stable argsort, marks where each group of equal values
    starts and ends, and uses running maximum / minimum / cumulative sum to
    give every element the start, end or group number of its tie group.
    The ranks are then written back to the original positions.
    Descending ranks for "min", "max", "average" and "dense" are derived
    from the ascending ones. Descending "ordinal" ranks the reversed array
    and flips the result, so ties are still broken by original position.

    Time: O(N log n).
    Space: O(N).
    """
    n = x.shape[-1]
    if method == "ordinal" and descending:
        return n + 1 - _rank_last(x[..., ::-1], "ordinal", False)[..., ::-1]

    order = np.argsort(x, axis=-1, kind="stable")
    xs = np.take_along_axis(x, order, axis=-1)
    pos = np.broadcast_to(np.arange(1, n + 1), x.shape)

    if method == "ordinal":
        vals = pos
    else:
        diff = xs[..., 1:] != xs[..., :-1]
        ones = np.ones(x.shape[:-1] + (1,), dtype=bool)
        first = np.concatenate([ones, diff], axis=-1)
        last = np.concatenate([diff, ones], axis=-1)
        start = np.maximum.accumulate(np.where(first, pos, 0), axis=-1)
        end = np.minimum.accumulate(
            np.where(last, pos, n + 1)[..., ::-1], axis=-1)[..., ::-1]
        dense = np.cumsum(first, axis=-1)
        if descending:
            start, end = n + 1 - end, n + 1 - start
            dense = dense[..., -1:] + 1 - dense
        vals = {"min": start, "max": end, "dense": dense,
                "average": (start + end) / 2.0}[method]

    out = np.empty(x.shape, dtype=vals.dtype)
    np.put_along_axis(out, order, vals, axis=-1)
    return out


def rank(data, method="average", axis=-1, descending=False):
    """
    Rank the values of `data` along an axis, with five ways to handle ties.

    Rank 1 is the smallest value (or the largest when descending=True). For the
    values [10, 20, 20, 30] the methods give:

        average -> 1, 2.5, 2.5, 4
        min     -> 1, 2, 2, 4
        max     -> 1, 3, 3, 4
        dense   -> 1, 2, 2, 3
        ordinal -> 1, 2, 3, 4

    Parameters
    ----------
    data : array_like, at least 1-D
        Values to rank. Text values are allowed; NaN is not.
    method : {"average", "min", "max", "dense", "ordinal"}, default "average"
        How tied values are ranked.
    axis : int, default -1
        Axis along which to rank.
    descending : bool, default False
        If True, rank from largest to smallest.

    Returns
    -------
    np.ndarray, same shape as `data`
        Float array for method="average", integer array for the other
        methods. An empty axis gives an empty float array.

    Raises
    ------
    ValueError
        If `method` is unknown, `data` is 0-d or contains NaN, or
        `axis` is out of range.

    Notes
    -----
    With "ordinal", tied values are ranked in order of appearance (also when
    descending). The input is not modified.

    Time: O(N log n).
    Space: O(N).
    """
    x, axis = _check(data, method, axis)
    if x.shape[axis] == 0:
        return np.empty(x.shape, dtype=float)
    moved = np.moveaxis(x, axis, -1)
    return np.moveaxis(_rank_last(moved, method, descending), -1, axis)


def percentile_rank(data, axis=-1, descending=False):
    """
    Score each value from 0 (lowest) to 1 (highest) using its average rank.

    Parameters
    ----------
    data : array_like, at least 1-D
        Values to score; NaN is not allowed.
    axis : int, default -1
        Axis along which to score.
    descending : bool, default False
        If True, the largest value scores 0.

    Returns
    -------
    np.ndarray, same shape as `data`
        Float array equal to (rank - 1) / (n - 1), where n is the length of
        `axis`. Tied values share a score. If n <= 1 every score is 0.

    Raises
    ------
    ValueError
        If `data` is 0-d or contains NaN, or `axis` is out of range.

    Notes
    -----
    Built on `rank` with method="average".

    Time: O(N log n).
    Space: O(N).
    """
    r = rank(data, "average", axis, descending)
    n = r.shape[axis]
    if n <= 1:
        return np.zeros_like(r)
    return (r - 1.0) / (n - 1.0)


def percentile(data, q, interpolation="linear", axis=-1):
    """
    Compute the q-th percentile of `data` along an axis, ignoring NaN.

    Parameters
    ----------
    data : array_like, at least 1-D
        Numeric values. NaN is ignored; infinity is not allowed.
    q : float or array_like of shape (Q,)
        Percentile(s) to compute, each in [0, 100].
    interpolation : {"linear", "lower", "higher", "midpoint"}, default "linear"
        What to return when the percentile falls between two data points:
        the linear interpolation, the lower point, the higher point, or
        their midpoint.
    axis : int, default -1
        Axis along which to compute.

    Returns
    -------
    np.ndarray or float
        `axis` is removed from the shape. If `q` is a sequence, the result
        has a new first axis of length Q. A slice with no valid value gives
        NaN, and so does an empty axis.

    Raises
    ------
    ValueError
        If `interpolation` is unknown, `data` is 0-d, not numeric or
        contains infinity, `axis` is out of range, `q` has more than
        one dimension, or `q` is NaN or outside [0, 100].

    Notes
    -----
    Each slice is sorted once (NaN goes last). The position of the percentile
    is q / 100 * (n_valid - 1), where n_valid counts the non-NaN values of
    that slice, and the two neighbouring values are combined according to
    `interpolation`. The results agree with np.percentile / np.nanpercentile.

    Time: O(N log n + Q * S), where S is the number of slices.
    Space: O(N + Q * S).
    """
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(
            f"interpolation must be one of {_INTERPOLATIONS}, got {interpolation!r}")
    x = np.asarray(data)
    if x.ndim == 0:
        raise ValueError("data must be at least 1-D")
    try:
        x = x.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data must be numeric ({exc})") from exc
    if np.isinf(x).any():
        raise ValueError("data contains infinity; remove or replace it first")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError(f"axis {axis} is out of range for {x.ndim}-D input")
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim > 1:
        raise ValueError("q must be a scalar or 1-D")
    if np.isnan(q_arr).any() or (q_arr < 0).any() or (q_arr > 100).any():
        raise ValueError("q must lie in [0, 100]")
    scalar_q = q_arr.ndim == 0
    qs = q_arr.reshape(-1)

    x = np.moveaxis(x, axis, -1)
    n_valid = (~np.isnan(x)).sum(axis=-1)
    s = np.sort(x, axis=-1)
    if s.shape[-1] == 0:
        out = np.full((len(qs),) + x.shape[:-1], np.nan)
        return out[0] if scalar_q else out

    qcol = qs.reshape((-1,) + (1,) * n_valid.ndim)
    pos = qcol / 100.0 * (np.maximum(n_valid, 1) - 1)
    lo = np.floor(pos).astype(np.intp)
    hi = np.ceil(pos).astype(np.intp)
    big = np.broadcast_to(s, (len(qs),) + s.shape)
    v_lo = np.take_along_axis(big, lo[..., None], axis=-1)[..., 0]
    v_hi = np.take_along_axis(big, hi[..., None], axis=-1)[..., 0]

    if interpolation == "linear":
        out = v_lo + (v_hi - v_lo) * (pos - lo)
    elif interpolation == "lower":
        out = v_lo
    elif interpolation == "higher":
        out = v_hi
    else:
        out = (v_lo + v_hi) / 2.0
    out = np.where(n_valid == 0, np.nan, out)
    return out[0] if scalar_q else out
