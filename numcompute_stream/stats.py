"""
stats.py
========
Descriptive statistics built using NumPy: count, mean, var, std, min, max,
quantile, median, mode, histogram, a describe() summary, and chunk-based
(streaming) versions of all of them:

- RunningStats        count, mean, variance, std, min, max (Welford / Chan)
- StreamingQuantile   quantiles from a reservoir sample or a sliding window
- StreamingHistogram  fixed-edge histogram, optionally over a sliding window
- StreamStats         all of the above behind one update_stats(X_chunk) call
- update_stats        the same as a function: state = update_stats(chunk, state)

Conventions
-----------
- Inputs are at least 1-D. `axis=None` reduces over all values; for 2-D data
  `axis=0` gives one result per column and `axis=1` one result per row.
  Higher-dimensional input works the same way, reducing along `axis`.
  mode and histogram take 1-D data, describe takes 1-D or 2-D data.
- min and max are not in __all__, so "from stats import *" does not hide
  Python's built-in min and max. Call them as stats.min and stats.max.
- NaN is treated as missing and ignored. Infinity is rejected with ValueError.
- A statistic with no valid values (or too few for `ddof`) is NaN.
- Time/space complexity is given in terms of N, the total number of elements
  of the input.

"""

import numpy as np

from .utils import INTERPOLATIONS as _INTERPOLATIONS
from .utils import MISSING_TOKENS as _MISSING_TOKENS
from .utils import as_float_array, check_axis

__all__ = [
    "count", "mean", "var", "std", "quantile", "median",
    "mode", "histogram", "describe", "describe_categorical",
    "format_summary", "format_histogram", "RunningStats",
    "StreamingQuantile", "StreamingHistogram", "StreamStats", "update_stats",
]

def _as_float_array(data, name="data"):
    """
    Convert the input to a float array and check it is at least 1-D without infinity.

    Parameters
    ----------
    data : array_like
        Numeric input. NaN is allowed and means "missing".
    name : str, default "data"
        Name used in error messages.

    Returns
    -------
    np.ndarray of float64, at least 1-D
        The data as a float array (no copy if already float64).

    Raises
    ------
    ValueError
        If the data is complex or cannot be converted to float, is 0-D, or
        contains infinity.

    Notes
    -----
    Time: O(N). Space: O(1) if already float, otherwise O(N).
    """
    arr = as_float_array(data, name, allow_inf=False).astype(float, copy=False)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be at least 1-D, got a scalar")
    return arr

def _check_axis(arr, axis):
    """
    Validate `axis` (None or an int in range) and return it as a non-negative int or None.

    Parameters
    ----------
    arr : np.ndarray
        Array the axis refers to.
    axis : None or int
        Axis to validate; negative values count from the last axis.

    Returns
    -------
    int or None
        None if `axis` is None, otherwise the axis in the range [0, arr.ndim).

    Raises
    ------
    TypeError
        If `axis` is neither None nor an integer (booleans are rejected).
    ValueError
        If `axis` is out of range for `arr`.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    return check_axis(axis, arr.ndim, allow_none=True)


def _check_ddof(ddof):
    """
    Validate the delta degrees of freedom.

    Parameters
    ----------
    ddof : int
        Value subtracted from the number of observations in the variance
        divisor.

    Returns
    -------
    int
        `ddof` as a Python int.

    Raises
    ------
    TypeError
        If `ddof` is not an integer (booleans are rejected).
    ValueError
        If `ddof` is negative.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)):
        raise TypeError(f"ddof must be an integer, got {type(ddof).__name__}")
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
    return int(ddof)


def _finish(x):
    """
    Turn a 0-d result into a NumPy scalar and leave larger arrays unchanged.

    Parameters
    ----------
    x : array_like
        Result of a reduction.

    Returns
    -------
    scalar or np.ndarray
        ``np.asarray(x)[()]``: a scalar for 0-d input, otherwise the array.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    return np.asarray(x)[()]


def _divide(num, den):
    """
    Element-wise division that gives NaN wherever the denominator is not positive.

    Parameters
    ----------
    num : array_like
        Numerator.
    den : array_like
        Denominator, broadcastable with `num`.

    Returns
    -------
    np.ndarray of float
        ``num / den`` where ``den > 0``, NaN elsewhere, with the broadcast
        shape of `num` and `den`.

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _mean(arr, axis, keepdims=False):
    """
    Mean of the non-NaN values along `axis`.

    Parameters
    ----------
    arr : np.ndarray of float, at least 1-D
        Validated input.
    axis : int or None
        Validated axis.
    keepdims : bool, default False
        Keep the reduced axis with length 1 so the result broadcasts against
        `arr`.

    Returns
    -------
    np.ndarray of float
        The mean, NaN where a slice has no valid values. Shape is that of
        `arr` with `axis` removed (or kept with length 1).

    Notes
    -----
    The values are divided by a power of two near the largest |value| before
    summing, so the sum cannot overflow (mean([1e308, 1e308]) is 1e308, not
    inf). Dividing by a power of two is exact, so the result is unchanged.

    Time: O(N). Space: O(N).
    """
    valid = ~np.isnan(arr)
    n = valid.sum(axis=axis, keepdims=keepdims)
    biggest = np.where(valid, np.abs(arr), 0.0).max(axis=axis, keepdims=True, initial=0.0)
    # Dividing by 2**(exponent - 1) keeps every scaled value below 2, so the
    # sum stays finite.
    scale = np.ldexp(1.0, np.frexp(biggest)[1] - 1)
    total = np.where(valid, arr / scale, 0.0).sum(axis=axis, keepdims=keepdims)
    if not keepdims:
        scale = scale.reshape(()) if axis is None else np.squeeze(scale, axis=axis)
    return _divide(total, n) * scale

def count(data, axis=None):
    """
    Number of non-NaN values (integer result).

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data.
    axis : None or int, default None
        Axis to count along; None counts all values.

    Returns
    -------
    int or np.ndarray of int
        A scalar when `axis` is None or `data` is 1-D; otherwise shape (d,)
        for axis=0 or (n,) for axis=1.

    Raises
    ------
    TypeError
        If `axis` is not None or an integer.
    ValueError
        If `data` is not numeric, 0-D, or contains infinity, or if
        `axis` is out of range.

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    arr = _as_float_array(data)
    ax = _check_axis(arr, axis)
    return _finish((~np.isnan(arr)).sum(axis=ax))


def mean(data, axis=None):
    """
    Mean of the non-NaN values.

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data.
    axis : None or int, default None
        Axis to reduce; None uses all values.

    Returns
    -------
    float or np.ndarray of float
        Scalar when `axis` is None or `data` is 1-D, otherwise shape (d,) for
        axis=0 or (n,) for axis=1. NaN where there are no valid values.

    Raises
    ------
    TypeError
        If `axis` is not None or an integer.
    ValueError
        If `data` is not numeric, 0-D, or contains infinity, or if
        `axis` is out of range.

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    arr = _as_float_array(data)
    ax = _check_axis(arr, axis)
    return _finish(_mean(arr, ax))


def var(data, axis=None, ddof=0):
    """
    Variance of the non-NaN values (two-pass formula, divides by n - ddof).

    NaN when n - ddof <= 0. The default ddof=0 matches StandardScaler.

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data.
    axis : None or int, default None
        Axis to reduce; None uses all values.
    ddof : int, default 0
        Delta degrees of freedom: the divisor is n - ddof, where n is the
        number of non-NaN values.

    Returns
    -------
    float or np.ndarray of float
        Scalar when `axis` is None or `data` is 1-D, otherwise shape (d,) for
        axis=0 or (n,) for axis=1.

    Raises
    ------
    TypeError
        If `axis` or `ddof` is not an integer (axis may also be None).
    ValueError
        If `data` is not numeric, 0-D, or contains infinity;
        `axis` is out of range; or `ddof` is negative.

    Notes
    -----
    The two-pass formula (subtract the mean first, then average the squares)
    avoids the cancellation error of ``E[x^2] - E[x]^2``.

    Time: O(N). Space: O(N).
    """
    arr = _as_float_array(data)
    ax = _check_axis(arr, axis)
    ddof = _check_ddof(ddof)
    valid = ~np.isnan(arr)
    n = valid.sum(axis=ax)
    centre = _mean(arr, ax, keepdims=True)
    squares = np.where(valid, (arr - centre) ** 2, 0.0).sum(axis=ax)
    return _finish(_divide(squares, n - ddof))


def std(data, axis=None, ddof=0):
    """
    Standard deviation of the non-NaN values (square root of var).

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data.
    axis : None or int, default None
        Axis to reduce; None uses all values.
    ddof : int, default 0
        Delta degrees of freedom, as in `var`.

    Returns
    -------
    float or np.ndarray of float
        Same shape rules as `var`; NaN where the variance is NaN.

    Raises
    ------
    TypeError
        If `axis` or `ddof` is not an integer (axis may also be None).
    ValueError
        If `data` is not numeric, 0-D, or contains infinity;
        `axis` is out of range; or `ddof` is negative.

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    return _finish(np.sqrt(var(data, axis=axis, ddof=ddof)))


def min(data, axis=None):
    """
    Smallest non-NaN value.

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data.
    axis : None or int, default None
        Axis to reduce; None uses all values.

    Returns
    -------
    float or np.ndarray of float
        Scalar when `axis` is None or `data` is 1-D, otherwise shape (d,) for
        axis=0 or (n,) for axis=1. NaN where there are no valid values
        (including empty input).

    Raises
    ------
    TypeError
        If `axis` is not None or an integer.
    ValueError
        If `data` is not numeric, 0-D, or contains infinity, or if
        `axis` is out of range.

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    arr = _as_float_array(data)
    ax = _check_axis(arr, axis)
    valid = ~np.isnan(arr)
    result = np.where(valid, arr, np.inf).min(axis=ax, initial=np.inf)
    return _finish(np.where(valid.any(axis=ax), result, np.nan))


def max(data, axis=None):
    """
    Largest non-NaN value.

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data.
    axis : None or int, default None
        Axis to reduce; None uses all values.

    Returns
    -------
    float or np.ndarray of float
        Scalar when `axis` is None or `data` is 1-D, otherwise shape (d,) for
        axis=0 or (n,) for axis=1. NaN where there are no valid values
        (including empty input).

    Raises
    ------
    TypeError
        If `axis` is not None or an integer.
    ValueError
        If `data` is not numeric, 0-D, or contains infinity, or if
        `axis` is out of range.

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    arr = _as_float_array(data)
    ax = _check_axis(arr, axis)
    valid = ~np.isnan(arr)
    result = np.where(valid, arr, -np.inf).max(axis=ax, initial=-np.inf)
    return _finish(np.where(valid.any(axis=ax), result, np.nan))


def _check_q(q):
    """
    Validate the quantile level(s).

    Parameters
    ----------
    q : float or sequence of float
        Quantile level(s), each in [0, 1].

    Returns
    -------
    qs : np.ndarray of float, shape (k,)
        The levels as a 1-D array (k = 1 for a scalar).
    is_scalar : bool
        True if `q` was a scalar.

    Raises
    ------
    ValueError
        If `q` is not numeric, has more than one dimension, is empty,
        contains NaN, or lies outside [0, 1].

    Notes
    -----
    Time: O(k). Space: O(k).
    """
    try:
        qs = np.asarray(q, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"q must be numeric ({exc})") from exc
    if qs.ndim > 1:
        raise ValueError(f"q must be a scalar or a 1-D sequence, got shape {qs.shape}")
    if qs.size == 0 or np.isnan(qs).any() or (qs < 0).any() or (qs > 1).any():
        raise ValueError("q must be non-empty and lie in [0, 1]")
    return np.atleast_1d(qs), qs.ndim == 0


def _take_last(sorted_arr, idx):
    """
    Pick one element along the last axis of `sorted_arr` for every slice.

    Parameters
    ----------
    sorted_arr : np.ndarray, shape (..., L)
        Array to index.
    idx : array_like of int, shape (...)
        Index along the last axis for each leading position.

    Returns
    -------
    np.ndarray, shape (...)
        ``sorted_arr[..., idx]`` taken slice by slice.

    Notes
    -----
    Time: O(S) for S slices. Space: O(S).
    """
    return np.take_along_axis(sorted_arr, np.asarray(idx)[..., None], axis=-1)[..., 0]


def _quantile_sorted(sorted_arr, n, q, interpolation):
    """
    Quantile q of each slice of an already sorted array.

    Parameters
    ----------
    sorted_arr : np.ndarray, shape (..., L)
        Values sorted along the last axis, with NaN at the end (as np.sort
        does), so the first n[...] entries of each slice are valid.
    n : np.ndarray of int, shape (...)
        Number of non-NaN values in each slice.
    q : float
        Quantile level in [0, 1].
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}
        How to resolve a quantile that falls between two data points.

    Returns
    -------
    np.ndarray of float, shape (...)
        The quantile of each slice; NaN where n == 0.

    Notes
    -----
    Time: O(S) for S slices. Space: O(S).
    """
    pos = np.where(n > 0, q * (n - 1), 0.0)
    # Snap positions that are integers up to rounding error, so q*(n-1) such
    # as 2.9999999999 selects the intended element.
    rounded = np.round(pos)
    pos = np.where(np.isclose(pos, rounded, rtol=0.0, atol=1e-9), rounded, pos)
    low = np.floor(pos).astype(np.intp)
    high = np.ceil(pos).astype(np.intp)
    low_val = _take_last(sorted_arr, low)
    high_val = _take_last(sorted_arr, high)
    if interpolation == "lower":
        result = low_val
    elif interpolation == "higher":
        result = high_val
    elif interpolation == "midpoint":
        result = (low_val + high_val) / 2.0
    else:
        result = low_val + (high_val - low_val) * (pos - low)
    return np.where(n > 0, result, np.nan)


def quantile(data, q, axis=None, interpolation="linear"):
    """
    Quantile(s) q in [0, 1] of the non-NaN values.

    interpolation is 'linear', 'lower', 'higher' or 'midpoint'. A scalar q
    gives the same shape as the other statistics; a sequence of k values
    puts a leading axis of length k in front of that shape.

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data.
    q : float or sequence of float
        Quantile level(s), each in [0, 1].
    axis : None or int, default None
        Axis to reduce; None uses all values.
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default 'linear'
        How to resolve a quantile that falls between two data points.

    Returns
    -------
    float or np.ndarray of float
        For scalar q: a scalar when `axis` is None or `data` is 1-D,
        otherwise shape (d,) for axis=0 or (n,) for axis=1. For a sequence
        of k levels: the same shape with a leading axis of length k, e.g.
        (k,) or (k, d). NaN where there are no valid values.

    Raises
    ------
    TypeError
        If `axis` is not None or an integer.
    ValueError
        If `data` is not numeric, 0-D, or contains infinity;
        `axis` is out of range; `interpolation` is unknown; or `q` is not
        numeric, empty, more than 1-D, or outside [0, 1].

    Notes
    -----
    Time: O(N log L) for L the length of the reduced axis (dominated by the
    sort), plus O(k S) for k levels and S slices. Space: O(N).
    """
    arr = _as_float_array(data)
    ax = _check_axis(arr, axis)
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {_INTERPOLATIONS}, got {interpolation!r}")
    qs, scalar_q = _check_q(q)

    a = arr.reshape(-1) if ax is None else np.moveaxis(arr, ax, -1)
    n = (~np.isnan(a)).sum(axis=-1)

    if a.shape[-1] == 0:
        results = [np.full(np.shape(n), np.nan) for _ in qs]
    else:
        sorted_arr = np.sort(a, axis=-1)
        results = [_quantile_sorted(sorted_arr, n, qi, interpolation) for qi in qs]

    out = results[0] if scalar_q else np.stack(results, axis=0)
    return _finish(out)


def median(data, axis=None):
    """
    Median of the non-NaN values (the 0.5 quantile).

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data.
    axis : None or int, default None
        Axis to reduce; None uses all values.

    Returns
    -------
    float or np.ndarray of float
        Same shape rules as `quantile` with a scalar q; NaN where there are
        no valid values.

    Raises
    ------
    TypeError
        If `axis` is not None or an integer.
    ValueError
        If `data` is not numeric, 0-D, or contains infinity, or if
        `axis` is out of range.

    Notes
    -----
    Time: O(N log L) for L the length of the reduced axis. Space: O(N).
    """
    return quantile(data, 0.5, axis=axis)


def mode(data):
    """
    Most frequent value of a 1-D array of numbers or strings, with its count.

    Returns (value, frequency); ties go to the smallest value. NaNs are ignored.
    Raises ValueError if there is no value.

    Parameters
    ----------
    data : array_like, shape (n,)
        Numbers or strings.

    Returns
    -------
    value : scalar
        The most frequent value, as a NumPy scalar of the dtype of `data`.
    frequency : int
        How many times `value` occurs.

    Raises
    ------
    ValueError
        If `data` is not 1-D, or has no non-NaN value (including empty
        input).

    Notes
    -----
    Time: O(n log n) (np.unique sorts). Space: O(n).
    """
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind == "f":
        arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("mode needs at least one non-missing value")
    values, counts = np.unique(arr, return_counts=True)
    i = int(np.argmax(counts))
    return values[i], int(counts[i])


def histogram(data, bins=10, range=None, density=False):
    """
    Histogram of a 1-D array: returns (counts, bin_edges).

    bins is a positive integer (equal-width bins over `range`, default the
    data minimum to maximum) or a 1-D increasing array of edges. Bins are
    half-open [a, b) except the last, which also includes its right edge.
    NaNs and values outside the edges are not counted. With density=True the
    result integrates to 1 over the bins.

    Parameters
    ----------
    data : array_like, shape (n,)
        Numeric data.
    bins : int or array_like of float, default 10
        Number of equal-width bins (>= 1), or a 1-D strictly increasing array
        of B + 1 finite bin edges.
    range : (float, float), optional
        (low, high) with low < high, both finite; only allowed when `bins` is
        an integer. Named `range` to match np.histogram; the built-in range is
        not used inside this function, so nothing breaks. Defaults to the
        minimum and maximum of the non-NaN data
        ((0, 1) for empty data). If low equals high the interval is widened
        by 0.5 on each side.
    density : bool, default False
        If True, return counts divided by (total count * bin width), so the
        histogram integrates to 1 (all zeros if no value falls in the bins).

    Returns
    -------
    counts : np.ndarray, shape (B,)
        Integer counts, or floats if `density` is True.
    bin_edges : np.ndarray of float, shape (B + 1,)
        The bin edges.

    Raises
    ------
    TypeError
        If `density` is not a bool.
    ValueError
        If `data` is not numeric, not 1-D, or contains infinity; `bins` is an
        integer below 1 or an invalid edge array; `range` is invalid or is
        combined with explicit edges.

    Notes
    -----
    Time: O(n log B) for the bin lookup plus O(n + B) for the counting.
    Space: O(n + B).
    """
    arr = _as_float_array(data)
    if arr.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {arr.shape}")
    if not isinstance(density, (bool, np.bool_)):
        raise TypeError(f"density must be a bool, got {type(density).__name__}")
    x = arr[~np.isnan(arr)]

    if isinstance(bins, (int, np.integer)) and not isinstance(bins, (bool, np.bool_)):
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        if range is None:
            low, high = (float(x.min()), float(x.max())) if x.size else (0.0, 1.0)
        else:
            try:
                low, high = (float(v) for v in range)
            except (TypeError, ValueError) as exc:
                raise ValueError("range must be a (low, high) pair of numbers") from exc
            if not (np.isfinite(low) and np.isfinite(high) and low < high):
                raise ValueError(f"range must satisfy low < high with finite values, got {range!r}")
        if low == high:
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, int(bins) + 1)
    else:
        if range is not None:
            raise ValueError("range cannot be combined with explicit bin edges")
        try:
            edges = np.asarray(bins, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("bins must be a positive integer or a 1-D array of edges") from exc
        if (edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0)):
            raise ValueError("bins must be a positive integer or a 1-D increasing array of edges")

    n_bins = edges.size - 1
    inside = x[(x >= edges[0]) & (x <= edges[-1])]
    # side="right" makes bins half-open [a, b); min() moves a value equal to
    # the last edge back into the last bin, which includes its right edge.
    index = np.minimum(np.searchsorted(edges, inside, side="right") - 1, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)

    if density:
        total = counts.sum()
        widths = np.diff(edges)
        counts = counts / (total * widths) if total > 0 else np.zeros(n_bins)
    return counts, edges


def describe(data, names=None):
    """
    Per-column summary of numeric data (like pandas describe).

    Returns a dict with 'columns' (names) and one array per statistic, each of
    shape (n_columns,): count, missing, mean, std, min, 25%, 50%, 75%, max.
    A 1-D input is treated as a single column. Pass the dict to
    format_summary to print it as a table.

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Numeric data; NaN means missing.
    names : sequence of str, optional
        Column names, one per column. Defaults to 'col0', 'col1', ...

    Returns
    -------
    dict
        'columns' is a list of d column names. The other keys ('count',
        'missing', 'mean', 'std', 'min', '25%', '50%', '75%', 'max') each map
        to an array of shape (d,). 'count' and 'missing' are integer arrays;
        the others are floats, NaN for a column with no valid values. 'std'
        uses ddof=0.

    Raises
    ------
    ValueError
        If `data` is not numeric, not 1-D or 2-D, or contains infinity, or
        if the number of `names` differs from the number of columns.

    Notes
    -----
    Time: O(N log n) for n rows (the quartiles need a sort). Space: O(N).
    """
    arr = _as_float_array(data)
    if arr.ndim > 2:
        raise ValueError(f"data must be 1-D or 2-D, got shape {arr.shape}")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_cols = arr.shape[1]
    if names is None:
        names = [f"col{j}" for j in range(n_cols)]
    else:
        names = [str(nm) for nm in names]
        if len(names) != n_cols:
            raise ValueError(f"names must have {n_cols} entries, got {len(names)}")

    n_valid = count(arr, axis=0)
    quartiles = quantile(arr, [0.25, 0.5, 0.75], axis=0)
    return {
        "columns": names,
        "count": n_valid,
        "missing": arr.shape[0] - n_valid,
        "mean": mean(arr, axis=0),
        "std": std(arr, axis=0),
        "min": min(arr, axis=0),
        "25%": quartiles[0],
        "50%": quartiles[1],
        "75%": quartiles[2],
        "max": max(arr, axis=0),
    }


def describe_categorical(data):
    """
    Summary of one text column: count, missing, unique, top (mode) and freq.

    Empty strings and the tokens nan / na / none / null count as missing.

    Parameters
    ----------
    data : array_like, shape (n,)
        Values of one column; they are converted to strings, stripped of
        surrounding whitespace, and compared case-insensitively against the
        missing-value tokens.

    Returns
    -------
    dict
        'count' (number of present values), 'missing', 'unique' (number of
        distinct present values), 'top' (most frequent present value as a str,
        ties to the smallest, None if there is none) and 'freq' (its count).

    Raises
    ------
    ValueError
        If `data` is not 1-D.

    Notes
    -----
    Time: O(n log n) (sorting inside np.unique) for strings of bounded
    length. Space: O(n).
    """
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {arr.shape}")
    text = np.char.strip(arr.astype(str))
    missing = np.isin(np.char.lower(text), _MISSING_TOKENS)
    present = text[~missing]
    if present.size == 0:
        return {"count": 0, "missing": int(arr.shape[0]), "unique": 0, "top": None, "freq": 0}
    top, freq = mode(present)
    return {
        "count": int(present.size),
        "missing": int(missing.sum()),
        "unique": int(np.unique(present).size),
        "top": str(top),
        "freq": int(freq),
    }


def _format_value(v, precision):
    """
    Format one table cell.

    Parameters
    ----------
    v : int or float
        Value to format.
    precision : int
        Number of decimals for floats.

    Returns
    -------
    str
        The integer as text, "nan" for NaN, otherwise a fixed-point float.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if np.isnan(v):
        return "nan"
    return f"{v:.{precision}f}"


def format_summary(summary, precision=3):
    """
    Format the dict returned by describe() as an aligned text table.

    Parameters
    ----------
    summary : dict
        Output of `describe`: a 'columns' list plus one array of shape (d,)
        per statistic.
    precision : int, default 3
        Number of decimals for floating-point cells.

    Returns
    -------
    str
        Multi-line table with a header row ("stat" followed by the column
        names) and one row per statistic. The statistic names are
        left-aligned and the values right-aligned.

    Raises
    ------
    KeyError
        If `summary` has no 'columns' key (it is not a `describe` result).

    Notes
    -----
    Time: O(S d) for S statistics and d columns. Space: O(S d).
    """
    header = ["stat"] + list(summary["columns"])
    rows = [
        [key] + [_format_value(v, precision) for v in summary[key]]
        for key in summary if key != "columns"
    ]
    table = [header] + rows
    widths = [int(np.max([len(r[i]) for r in table])) for i in range(len(header))]
    lines = []
    for r in table:
        cells = [r[0].ljust(widths[0])] + [r[i].rjust(widths[i]) for i in range(1, len(r))]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def format_histogram(counts, edges, width=40):
    """
    Format histogram() output as text bars, one line per bin.

    Parameters
    ----------
    counts : array_like, shape (B,)
        Bin counts (integers) or densities (floats) from `histogram`.
    edges : array_like, shape (B + 1,)
        Bin edges from `histogram`.
    width : int, default 40
        Length in characters of the longest bar (>= 1).

    Returns
    -------
    str
        One line per bin of the form ``[low, high) | ### value``; the last
        bin is closed with ``]``. Bar lengths are proportional to the counts.

    Raises
    ------
    ValueError
        If `counts` or `edges` is not 1-D, `edges` does not have exactly one
        more entry than `counts`, or `width` is not a positive integer.

    Notes
    -----
    Time: O(B + B * width). Space: O(B * width).
    """
    counts = np.asarray(counts)
    edges = np.asarray(edges, dtype=float)
    if counts.ndim != 1 or edges.ndim != 1 or edges.size != counts.size + 1:
        raise ValueError("edges must have exactly one more entry than counts")
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    peak = float(np.max(counts)) if counts.size else 0.0
    last = counts.size - 1
    lines = []
    for i, c in enumerate(counts):
        close = "]" if i == last else ")"
        bar = "#" * (int(round(width * c / peak)) if peak > 0 else 0)
        label = f"[{edges[i]:8.3f}, {edges[i + 1]:8.3f}{close}"
        value = str(int(c)) if np.issubdtype(counts.dtype, np.integer) else f"{c:.4f}"
        lines.append(f"{label} | {bar} {value}")
    return "\n".join(lines)

class RunningStats:
    """
    Streaming count, mean, variance, min and max, updated one chunk at a time.

    Each chunk is summarised (count, mean and M2, the sum of squared
    deviations from the mean) and merged into the running totals with
    Welford's update in the parallel form of Chan et al. Old chunks are never
    needed again, so data larger than memory can be summarised.

    Attributes
    ----------
    n_ : np.ndarray of float, shape () or (d,)
        Non-NaN values seen so far (per column for 2-D chunks).
    mean_ : np.ndarray of float, shape () or (d,)
        Running mean.
    M2_ : np.ndarray of float, shape () or (d,)
        Running sum of squared deviations from the mean.
    min_, max_ : np.ndarray of float, shape () or (d,)
        Running minimum and maximum (+inf / -inf until a value is seen).

    Notes
    -----
    Chunks are 1-D (one stream of numbers) or 2-D (n, d) with one result per
    column. Every chunk must have the same number of columns. NaN is ignored,
    and infinity is rejected, as in the rest of this module. Merging the
    chunk means instead of summing x and x**2 avoids the cancellation error
    of E[x^2] - E[x]^2, so the results match var() on all of the data.
    """

    def __init__(self):
        """
        Create an empty accumulator.

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        self.reset()

    def reset(self):
        """
        Forget everything seen so far.

        Returns
        -------
        RunningStats
            The same object (`self`).

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        self.n_ = self.mean_ = self.M2_ = self.min_ = self.max_ = None
        self._shape = None
        return self

    def partial_fit(self, data):
        """
        Merge one chunk into the running statistics.

        Parameters
        ----------
        data : array_like, shape (n,) or (n, d)
            Chunk of numbers; NaN is ignored. May have n = 0.

        Returns
        -------
        RunningStats
            The same object (`self`).

        Raises
        ------
        ValueError
            If `data` is not numeric, contains infinity, is not 1-D or 2-D,
            or has a different number of columns than the first chunk.

        Notes
        -----
        `update` is an alias.

        Time: O(n d). Space: O(n d) temporary, O(d) stored.
        """
        arr = _as_float_array(data)
        if arr.ndim > 2:
            raise ValueError(f"data must be 1-D or 2-D, got shape {arr.shape}")
        if self._shape is None:
            self._shape = arr.shape[1:]
            self.n_ = np.zeros(self._shape)
            self.mean_ = np.zeros(self._shape)
            self.M2_ = np.zeros(self._shape)
            self.min_ = np.full(self._shape, np.inf)
            self.max_ = np.full(self._shape, -np.inf)
        elif arr.shape[1:] != self._shape:
            raise ValueError(
                f"chunk has shape {arr.shape}, expected (n,) + {self._shape}"
            )

        valid = ~np.isnan(arr)
        n_b = valid.sum(axis=0).astype(float)
        mean_b = np.where(n_b > 0, _mean(arr, 0), 0.0)
        M2_b = np.where(valid, (arr - mean_b) ** 2, 0.0).sum(axis=0)

        # Chan et al.: combine the two (count, mean, M2) summaries in one step.
        n = self.n_ + n_b
        ratio = np.where(n > 0, _divide(n_b, n), 0.0)
        delta = mean_b - self.mean_
        self.mean_ = self.mean_ + delta * ratio
        self.M2_ = self.M2_ + M2_b + delta ** 2 * self.n_ * ratio
        self.n_ = n
        self.min_ = np.minimum(self.min_, np.where(valid, arr, np.inf).min(axis=0, initial=np.inf))
        self.max_ = np.maximum(self.max_, np.where(valid, arr, -np.inf).max(axis=0, initial=-np.inf))
        return self

    update = partial_fit
    update_stats = partial_fit

    def _check_fitted(self):
        """
        Raise an error if no chunk has been seen yet.

        Raises
        ------
        RuntimeError
            If neither `partial_fit` nor `update` has been called.

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        if self._shape is None:
            raise RuntimeError("RunningStats has seen no data; call partial_fit first")

    def count(self):
        """
        Number of non-NaN values seen.

        Returns
        -------
        int or np.ndarray of int, shape (d,)

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.

        Notes
        -----
        Time: O(d). Space: O(d).
        """
        self._check_fitted()
        return _finish(self.n_.astype(np.int64))

    def mean(self):
        """
        Mean of the values seen; NaN where nothing has been seen.

        Returns
        -------
        float or np.ndarray of float, shape (d,)

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.

        Notes
        -----
        Time: O(d). Space: O(d).
        """
        self._check_fitted()
        return _finish(np.where(self.n_ > 0, self.mean_, np.nan))

    def var(self, ddof=0):
        """
        Variance of the values seen, M2 / (n - ddof).

        Parameters
        ----------
        ddof : int, default 0
            Delta degrees of freedom, as in `var`.

        Returns
        -------
        float or np.ndarray of float, shape (d,)
            NaN where n - ddof <= 0.

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.
        TypeError, ValueError
            If `ddof` is not a non-negative integer.

        Notes
        -----
        Time: O(d). Space: O(d).
        """
        self._check_fitted()
        ddof = _check_ddof(ddof)
        return _finish(_divide(self.M2_, self.n_ - ddof))

    def std(self, ddof=0):
        """
        Standard deviation of the values seen (square root of `var`).

        Parameters
        ----------
        ddof : int, default 0
            Delta degrees of freedom.

        Returns
        -------
        float or np.ndarray of float, shape (d,)

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.
        TypeError, ValueError
            If `ddof` is not a non-negative integer.

        Notes
        -----
        Time: O(d). Space: O(d).
        """
        return _finish(np.sqrt(self.var(ddof)))

    def min(self):
        """
        Smallest value seen; NaN where nothing has been seen.

        Returns
        -------
        float or np.ndarray of float, shape (d,)

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.

        Notes
        -----
        Time: O(d). Space: O(d).
        """
        self._check_fitted()
        return _finish(np.where(self.n_ > 0, self.min_, np.nan))

    def max(self):
        """
        Largest value seen; NaN where nothing has been seen.

        Returns
        -------
        float or np.ndarray of float, shape (d,)

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.

        Notes
        -----
        Time: O(d). Space: O(d).
        """
        self._check_fitted()
        return _finish(np.where(self.n_ > 0, self.max_, np.nan))


# ---------------------------------------------------------------------------
# Streaming quantiles, histograms and the update_stats API
# ---------------------------------------------------------------------------
def _check_positive_int(value, name, allow_none=False):
    """
    Validate a positive integer option.

    Parameters
    ----------
    value : object
        Value to check.
    name : str
        Option name for error messages.
    allow_none : bool, default False
        Whether None is accepted.

    Returns
    -------
    int or None

    Raises
    ------
    TypeError
        If `value` is not an integer.
    ValueError
        If `value` < 1.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _as_chunk(data, shape, owner):
    """
    Validate one chunk for a streaming statistic.

    Parameters
    ----------
    data : array_like, shape (n,) or (n, d)
        Chunk of numbers; NaN allowed, infinity rejected.
    shape : tuple or None
        Trailing shape fixed by the first chunk (() or (d,)), or None.
    owner : str
        Class name for error messages.

    Returns
    -------
    np.ndarray of float, shape (n,) or (n, d)

    Raises
    ------
    ValueError
        If the chunk is not numeric, contains infinity, is not 1-D or 2-D, or
        has a different number of columns than the first chunk.

    Notes
    -----
    Time: O(n d). Space: O(1) if already float.
    """
    arr = _as_float_array(data)
    if arr.ndim > 2:
        raise ValueError(f"{owner}: a chunk must be 1-D or 2-D, got shape {arr.shape}")
    if shape is not None and arr.shape[1:] != shape:
        raise ValueError(
            f"{owner}: chunk has shape {arr.shape}, expected (n,) + {shape} "
            "as set by the first chunk"
        )
    return arr


class StreamingQuantile:
    """
    Quantiles of a stream, updated one chunk at a time.

    An exact quantile needs all of the data, which a stream cannot keep. Two
    bounded-memory modes are offered:

    - whole stream (``window=None``): a uniform random sample of at most
      `capacity` rows is kept by reservoir sampling, and the quantile of that
      sample is returned. While the stream is shorter than `capacity` the
      result is exact; afterwards it is an estimate whose error shrinks like
      1 / sqrt(capacity).
    - sliding window (``window=N``): the N most recent rows are kept and
      their exact quantile is returned.

    Parameters
    ----------
    capacity : int, default 4096
        Reservoir size in rows (whole-stream mode).
    window : int, optional
        Sliding-window length in rows. Default: the whole stream.
    seed : int or None, default 0
        Seed of the reservoir's random generator, so runs are reproducible.

    Attributes
    ----------
    n_seen_ : int
        Rows received so far.

    Notes
    -----
    Chunks are 1-D, or 2-D (n, d) with one quantile per column; rows are
    sampled whole, so columns stay aligned. NaN is ignored when the quantile
    is taken, and infinity is rejected, as in the rest of this module.

    The reservoir update is vectorised: row number i (0-based, counted over
    the whole stream) draws a slot j uniformly from [0, i] and replaces slot
    j if j < capacity. That is Vitter's Algorithm R applied to a whole chunk
    with one random draw and one fancy assignment.
    """

    def __init__(self, capacity=4096, window=None, seed=0):
        """
        Create an empty quantile sketch; see the class docstring.

        Raises
        ------
        TypeError, ValueError
            If `capacity` or `window` is not a positive integer.

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        self.capacity = _check_positive_int(capacity, "capacity")
        self.window = _check_positive_int(window, "window", allow_none=True)
        self.seed = seed
        self.reset()

    def reset(self):
        """
        Forget everything seen so far.

        Returns
        -------
        StreamingQuantile
            The same object (`self`).

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        self._rows = None
        self._shape = None
        self.n_seen_ = 0
        self._rng = np.random.default_rng(self.seed)
        return self

    def update(self, data):
        """
        Add one chunk.

        Parameters
        ----------
        data : array_like, shape (n,) or (n, d)
            Chunk of numbers. May have n = 0.

        Returns
        -------
        StreamingQuantile
            The same object (`self`).

        Raises
        ------
        ValueError
            As for RunningStats.partial_fit.

        Notes
        -----
        `partial_fit` is an alias.
        Time: O(n d). Space: O(min(capacity, window) d) stored.
        """
        arr = _as_chunk(data, self._shape, "StreamingQuantile")
        if self._shape is None:
            self._shape = arr.shape[1:]
            self._rows = np.empty((0,) + self._shape)
        n = arr.shape[0]
        if n == 0:
            return self
        if self.window is not None:
            self._rows = np.concatenate([self._rows, arr])[-self.window:]
            self.n_seen_ += n
            return self
        # fill the reservoir first, then replace slots at random
        room = self.capacity - self._rows.shape[0]
        if room > 0:
            self._rows = np.concatenate([self._rows, arr[:room]])
        # note: this module defines its own min/max, so the built-ins are avoided
        rest = arr[room:] if room > 0 else arr
        first = self.n_seen_ + (n - rest.shape[0])
        if rest.shape[0]:
            index = first + np.arange(rest.shape[0])
            slot = self._rng.integers(0, index + 1)
            keep = slot < self.capacity
            self._rows[slot[keep]] = rest[keep]
        self.n_seen_ += n
        return self

    partial_fit = update

    def result(self, q=0.5, interpolation="linear"):
        """
        Current quantile estimate.

        Parameters
        ----------
        q : float or array_like of float in [0, 1], default 0.5
            Quantile(s) to compute.
        interpolation : {"linear", "lower", "higher", "midpoint"}
            As in quantile().

        Returns
        -------
        float or np.ndarray
            Shape q.shape for 1-D chunks, q.shape + (d,) for 2-D chunks.
            NaN where no valid value has been seen.

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.
        ValueError
            If `q` is outside [0, 1] or `interpolation` is unknown.

        Notes
        -----
        Time: O(m log m) for m stored rows. Space: O(m d).
        """
        if self._shape is None:
            raise RuntimeError("StreamingQuantile has seen no data; call update first")
        if self._rows.shape[0] == 0:
            shape = np.shape(q) + self._shape
            return _finish(np.full(shape, np.nan))
        return quantile(self._rows, q, axis=0, interpolation=interpolation)


class StreamingHistogram:
    """
    Histogram of a stream with fixed bin edges, updated one chunk at a time.

    Parameters
    ----------
    bins : int or array_like of float, default 10
        Number of equal-width bins over `range`, or B + 1 increasing edges.
    range : (float, float), optional
        (low, high) of the histogram. Required when `bins` is an integer,
        because the edges of a stream must be fixed before the data arrives.
    window : int, optional
        Sliding window: only the `window` most recent values are counted.
        Default: the whole stream.

    Attributes
    ----------
    edges_ : np.ndarray of float, shape (B + 1,)
        Bin edges. Bins are half-open [a, b) except the last, which includes
        its right edge (the same rule as histogram()).
    n_below_, n_above_, n_nan_ : int
        Values below the first edge, above the last edge, and NaN values
        (whole-stream mode: since the last reset; window mode: inside the
        window). They are not part of the counts.
    n_seen_ : int
        Values received so far.

    Notes
    -----
    Data of any shape is flattened: the histogram describes one stream of
    numbers. Whole-stream mode stores only the B counts; window mode stores
    the last `window` values and recounts them.
    """

    def __init__(self, bins=10, range=None, window=None):
        """
        Create an empty histogram; see the class docstring.

        Raises
        ------
        ValueError
            If `bins` is an integer and `range` is missing or invalid, or the
            edges are not finite and strictly increasing.
        TypeError
            If `window` is not a positive integer.

        Notes
        -----
        Time: O(B). Space: O(B).
        """
        if isinstance(bins, (int, np.integer)) and not isinstance(bins, (bool, np.bool_)):
            if bins < 1:
                raise ValueError(f"bins must be >= 1, got {bins}")
            if range is None:
                raise ValueError(
                    "StreamingHistogram needs range=(low, high) when bins is an "
                    "integer: the edges must be fixed before the stream starts"
                )
            low, high = (float(v) for v in range)
            if not (np.isfinite(low) and np.isfinite(high) and low < high):
                raise ValueError(f"range must be finite with low < high, got {range!r}")
            edges = np.linspace(low, high, int(bins) + 1)
        else:
            if range is not None:
                raise ValueError("range is only allowed when bins is an integer")
            edges = np.asarray(bins, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("bin edges must be a 1-D array with at least 2 values")
            if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
                raise ValueError("bin edges must be finite and strictly increasing")
        self.edges_ = edges
        self.window = _check_positive_int(window, "window", allow_none=True)
        self.reset()

    def reset(self):
        """
        Forget everything seen so far (the edges are kept).

        Returns
        -------
        StreamingHistogram
            The same object (`self`).

        Notes
        -----
        Time: O(B). Space: O(B).
        """
        self._counts = np.zeros(self.edges_.size - 1, dtype=np.int64)
        self._below = self._above = self._nan = 0
        self._values = None
        self.n_seen_ = 0
        return self

    def _count(self, values):
        """
        Count values into the bins.

        Parameters
        ----------
        values : np.ndarray of float, shape (n,)

        Returns
        -------
        counts : np.ndarray of int64, shape (B,)
        below, above, nan : int

        Notes
        -----
        One searchsorted and one bincount; no loop over values.
        Time: O(n log B). Space: O(n + B).
        """
        nan = np.isnan(values)
        x = values[~nan]
        low, high = self.edges_[0], self.edges_[-1]
        below, above = x < low, x > high
        inside = x[~(below | above)]
        idx = np.searchsorted(self.edges_, inside, side="right") - 1
        idx = np.minimum(idx, self._counts.size - 1)   # right edge joins the last bin
        counts = np.bincount(idx, minlength=self._counts.size).astype(np.int64)
        return counts, int(below.sum()), int(above.sum()), int(nan.sum())

    def update(self, data):
        """
        Add one chunk.

        Parameters
        ----------
        data : array_like, any shape
            Chunk of numbers (flattened). NaN is counted in `n_nan_`.

        Returns
        -------
        StreamingHistogram
            The same object (`self`).

        Raises
        ------
        ValueError
            If `data` is not numeric or contains infinity.

        Notes
        -----
        `partial_fit` is an alias.
        Time: O(n log B). Space: O(n) temporary.
        """
        values = _as_float_array(data).ravel()
        self.n_seen_ += values.size
        if self.window is not None:
            joined = values if self._values is None else np.concatenate([self._values, values])
            self._values = joined[-self.window:]
            return self
        counts, below, above, nan = self._count(values)
        self._counts += counts
        self._below += below
        self._above += above
        self._nan += nan
        return self

    partial_fit = update

    def _current(self):
        """Counts and outlier tallies for the current mode (see `_count`)."""
        if self.window is None:
            return self._counts.copy(), self._below, self._above, self._nan
        if self._values is None:
            return self._counts.copy(), 0, 0, 0
        return self._count(self._values)

    @property
    def n_below_(self):
        """Values below the first edge."""
        return self._current()[1]

    @property
    def n_above_(self):
        """Values above the last edge."""
        return self._current()[2]

    @property
    def n_nan_(self):
        """NaN values."""
        return self._current()[3]

    def result(self, density=False):
        """
        Current histogram.

        Parameters
        ----------
        density : bool, default False
            If True, scale the counts so the histogram integrates to 1.

        Returns
        -------
        counts : np.ndarray, shape (B,)
            int64 counts, or float densities (all zeros if nothing is inside).
        edges : np.ndarray of float, shape (B + 1,)

        Notes
        -----
        Time: O(B) (O(window log B) with a window). Space: O(B).
        """
        counts = self._current()[0]
        if not density:
            return counts, self.edges_.copy()
        total = counts.sum()
        widths = np.diff(self.edges_)
        dens = np.zeros(counts.size)
        if total > 0:
            dens = counts / (total * widths)
        return dens, self.edges_.copy()


class StreamStats:
    """
    All streaming statistics of a dataset behind one ``update_stats`` call.

    Combines RunningStats (count, mean, variance, min, max),
    StreamingQuantile and, optionally, one StreamingHistogram per column.

    Parameters
    ----------
    quantiles : sequence of float in [0, 1], default (0.25, 0.5, 0.75)
        Quantiles reported by `result`.
    capacity : int, default 4096
        Reservoir size of the quantile estimate.
    window : int, optional
        Sliding window (in rows) for the quantiles and histograms. The
        moments always cover the whole stream.
    hist_bins : int, optional
        If given, keep a histogram with this many bins per column;
        `hist_range` is then required.
    hist_range : (float, float), optional
        Range of the histograms.
    seed : int or None, default 0
        Seed for the quantile reservoir.

    Examples
    --------
    >>> s = StreamStats()
    >>> _ = s.update_stats([[1.0, 10.0], [2.0, 20.0]])
    >>> _ = s.update_stats([[3.0, 30.0]])
    >>> s.result()["mean"].tolist()
    [2.0, 20.0]
    """

    def __init__(self, quantiles=(0.25, 0.5, 0.75), capacity=4096, window=None,
                 hist_bins=None, hist_range=None, seed=0):
        """
        Create an empty summary; see the class docstring.

        Raises
        ------
        ValueError, TypeError
            Invalid quantiles, capacity, window or histogram settings.

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        self.quantiles = _check_q(quantiles)[0]
        self._moments = RunningStats()
        self._quantile = StreamingQuantile(capacity=capacity, window=window, seed=seed)
        self._hist_bins = _check_positive_int(hist_bins, "hist_bins", allow_none=True)
        if self._hist_bins is not None:
            # validates the range once, up front
            StreamingHistogram(self._hist_bins, hist_range, window)
        self._hist_range = hist_range
        self._window = window
        self._hists = None

    def reset(self):
        """
        Forget everything seen so far.

        Returns
        -------
        StreamStats
            The same object (`self`).

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        self._moments.reset()
        self._quantile.reset()
        self._hists = None
        return self

    def update_stats(self, X_chunk):
        """
        Update every statistic with one chunk.

        Parameters
        ----------
        X_chunk : array_like, shape (n,) or (n, d)
            Chunk of numbers; NaN is ignored, infinity is rejected.

        Returns
        -------
        StreamStats
            The same object (`self`).

        Raises
        ------
        ValueError
            If the chunk is not numeric, contains infinity, is not 1-D or 2-D,
            or has a different number of columns than the first chunk. No
            statistic is updated in that case.

        Notes
        -----
        `update` and `partial_fit` are aliases. The histogram loop runs over
        columns (each has its own counts), never over rows.
        Time: O(n d log B). Space: O(n d) temporary.
        """
        arr = _as_chunk(X_chunk, self._moments._shape, "StreamStats")
        self._moments.partial_fit(arr)
        self._quantile.update(arr)
        if self._hist_bins is not None:
            cols = arr.reshape(arr.shape[0], -1)
            if self._hists is None:
                self._hists = [
                    StreamingHistogram(self._hist_bins, self._hist_range, self._window)
                    for _ in range(cols.shape[1])
                ]
            for j, hist in enumerate(self._hists):
                hist.update(cols[:, j])
        return self

    update = update_stats
    partial_fit = update_stats

    def result(self):
        """
        Current summary.

        Returns
        -------
        dict
            "count", "mean", "var", "std", "min", "max": float, or np.ndarray
            of shape (d,) for 2-D chunks. "quantiles": dict mapping each q to
            its value(s). "histogram": list of (counts, edges), one per
            column, only when `hist_bins` was given.

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.

        Notes
        -----
        Time: O(d + m log m) for m stored quantile rows. Space: O(d).
        """
        m = self._moments
        out = {
            "count": m.count(), "mean": m.mean(), "var": m.var(), "std": m.std(),
            "min": m.min(), "max": m.max(),
        }
        qs = np.atleast_1d(self.quantiles)
        values = self._quantile.result(qs)
        out["quantiles"] = {float(q): values[i] for i, q in enumerate(qs)}
        if self._hists is not None:
            out["histogram"] = [h.result() for h in self._hists]
        return out


def update_stats(X_chunk, state=None, **options):
    """
    Functional form of the streaming API: fold one chunk into a summary.

    Parameters
    ----------
    X_chunk : array_like, shape (n,) or (n, d)
        Chunk of numbers.
    state : StreamStats, optional
        Summary returned by an earlier call. None starts a new one.
    **options
        Passed to StreamStats when a new summary is created (quantiles,
        capacity, window, hist_bins, hist_range, seed).

    Returns
    -------
    StreamStats
        The updated summary; call ``.result()`` to read it.

    Raises
    ------
    TypeError
        If `state` is not a StreamStats, or options are given together with
        an existing state.
    ValueError
        As for StreamStats.update_stats.

    Notes
    -----
    Time and space as for StreamStats.update_stats.

    Examples
    --------
    >>> state = None
    >>> for chunk in ([1.0, 2.0], [3.0, 4.0]):
    ...     state = update_stats(chunk, state)
    >>> float(state.result()["mean"])
    2.5
    """
    if state is None:
        state = StreamStats(**options)
    elif not isinstance(state, StreamStats):
        raise TypeError(f"state must be a StreamStats or None, got {type(state).__name__}")
    elif options:
        raise TypeError("options can only be given when state is None")
    return state.update_stats(X_chunk)
