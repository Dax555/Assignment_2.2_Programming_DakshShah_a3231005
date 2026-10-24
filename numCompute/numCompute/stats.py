"""
stats.py
========
Descriptive statistics built using NumPy: count, mean, var, std, min, max,
quantile, median, mode, histogram, and a describe() summary.

Conventions
-----------
- Inputs are 1-D or 2-D. `axis=None` reduces over all values; for 2-D data
  `axis=0` gives one result per column and `axis=1` one result per row.
- NaN is treated as missing and ignored. Infinity is rejected with ValueError.
- A statistic with no valid values (or too few for `ddof`) is NaN.
- Time/space complexity is given in terms of N, the total number of elements
  of the input.

"""

import numpy as np

__all__ = [
    "count", "mean", "var", "std", "min", "max", "quantile", "median",
    "mode", "histogram", "describe", "describe_categorical",
    "format_summary", "format_histogram",
]

_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")
_MISSING_TOKENS = ("", "nan", "none", "null", "na")


def _as_float_array(data, name="data"):
    """
    Convert the input to a float array and check it is 1-D or 2-D without infinity.

    Parameters
    ----------
    data : array_like
        Numeric input. NaN is allowed and means "missing".
    name : str, default "data"
        Name used in error messages.

    Returns
    -------
    np.ndarray of float, shape (n,) or (n, d)
        The data as a float array (no copy if already float).

    Raises
    ------
    ValueError
        If the data cannot be converted to float, is not 1-D or 2-D, or
        contains infinity.

    Notes
    -----
    Time: O(N). Space: O(1) if already float, otherwise O(N).
    """
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; could not convert to float ({exc})") from exc
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinity; remove or replace it before using it")
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
    if axis is None:
        return None
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError(f"axis must be None or an integer, got {type(axis).__name__}")
    if not -arr.ndim <= axis < arr.ndim:
        raise ValueError(f"axis {axis} is out of range for a {arr.ndim}-D array")
    return int(axis) % arr.ndim


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
    arr : np.ndarray of float, shape (n,) or (n, d)
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
    Time: O(N). Space: O(N).
    """
    valid = ~np.isnan(arr)
    n = valid.sum(axis=axis, keepdims=keepdims)
    total = np.where(valid, arr, 0.0).sum(axis=axis, keepdims=keepdims)
    return _divide(total, n)


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
        If `data` is not numeric, not 1-D or 2-D, or contains infinity, or if
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
        If `data` is not numeric, not 1-D or 2-D, or contains infinity, or if
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
        If `data` is not numeric, not 1-D or 2-D, or contains infinity;
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
        If `data` is not numeric, not 1-D or 2-D, or contains infinity;
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
        If `data` is not numeric, not 1-D or 2-D, or contains infinity, or if
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
        If `data` is not numeric, not 1-D or 2-D, or contains infinity, or if
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
        If `data` is not numeric, not 1-D or 2-D, or contains infinity;
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
        If `data` is not numeric, not 1-D or 2-D, or contains infinity, or if
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
        an integer. Defaults to the minimum and maximum of the non-NaN data
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