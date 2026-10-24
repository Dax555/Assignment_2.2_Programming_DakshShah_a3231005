"""
utils.py
Shared helpers used by the other modules: input checks, pairwise distances, activation functions
(relu, sigmoid, softplus, softmax, logsumexp), top-k along an axis, and mini-batching.
Exponentials are always taken after subtracting the maximum, so large inputs don't overflow.
"""

import numpy as np

__all__ = [
    "MISSING_TOKENS",
    "INTERPOLATIONS",
    "check_axis",
    "as_float_array",
    "pairwise_distances",
    "relu",
    "leaky_relu",
    "sigmoid",
    "softplus",
    "logsumexp",
    "softmax",
    "log_softmax",
    "topk_along_axis",
    "batch_slices",
    "iter_batches",
]

# shared by stats.py, rank.py and preprocessing.py so they all agree
MISSING_TOKENS = ("", "nan", "none", "null", "na")
INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")

_METRICS = ("euclidean", "sqeuclidean", "manhattan", "cosine")
_NAN_POLICIES = ("propagate", "omit")
_CHUNK_ELEMENTS = 1 << 22  # about 32 MB of float64 per manhattan chunk


def check_axis(axis, ndim, allow_none=False, name="axis"):
    """Validate an axis argument and return it as a non-negative int.

    Parameters
    ----------
    axis : int or None
        Axis to validate; negative values count from the last axis.
    ndim : int
        Number of dimensions of the array the axis refers to.
    allow_none : bool, default False
        If True, None is accepted and returned unchanged.
    name : str, default 'axis'
        Argument name used in error messages.

    Returns
    -------
    int or None
        The axis in the range [0, ndim), or None if allowed and given.

    Raises
    ------
    TypeError
        If `axis` is not an integer (booleans are rejected) and not an allowed None.
    ValueError
        If `axis` is out of range for `ndim`.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if axis is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be an integer, got None")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(axis).__name__}")
    if not -ndim <= axis < ndim:
        raise ValueError(f"{name} {axis} is out of range for a {ndim}-D array")
    # turn a negative axis into its positive equivalent
    return int(axis) % ndim


def as_float_array(x, name="x", ndim=None, allow_nan=True, allow_inf=True):
    """Convert input to a floating-point array and validate it.

    Parameters
    ----------
    x : array_like
        Numeric input. Integer and boolean data become float64; float16/32/64 data keeps its dtype.
    name : str, default 'x'
        Argument name used in error messages.
    ndim : int or tuple of int, optional
        Required number of dimensions (or the allowed ones).
    allow_nan : bool, default True
        If False, NaN raises ValueError.
    allow_inf : bool, default True
        If False, +/- infinity raises ValueError.

    Returns
    -------
    np.ndarray of float
        The data as a float array (no copy if it already is one).

    Raises
    ------
    ValueError
        If `x` is complex or cannot be converted to float, has the wrong number of dimensions, or contains a disallowed NaN or infinity.
        Complex input is rejected because converting it would silently drop the imaginary part.
    Notes
    -----
    Time: O(N) for the NaN/inf scan, plus O(N) if a conversion is needed.
    Space: O(1) if already float, otherwise O(N).
    """
    if np.iscomplexobj(x):
        raise ValueError(f"{name} is complex, only real numbers are supported")
    try:
        arr = np.asarray(x)
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float64)
        elif arr.dtype.kind != "f":
            # e.g. object or string data; fails if it isn't numeric
            arr = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric; could not convert to float ({exc})"
        ) from exc
    if ndim is not None:
        allowed = (ndim,) if isinstance(ndim, (int, np.integer)) else tuple(ndim)
        if arr.ndim not in allowed:
            raise ValueError(
                f"{name} must have {' or '.join(map(str, allowed))} "
                f"dimension(s), got shape {arr.shape}"
            )
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN; remove or replace it first")
    if not allow_inf and np.isinf(arr).any():
        raise ValueError(f"{name} contains infinity; remove or replace it first")
    return arr


def _check_nan_policy(nan_policy):
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}"
        )
    return nan_policy


def _check_int(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def pairwise_distances(X, Y=None, metric="euclidean", batch_size=None):
    """Distance between every row of X and every row of Y.

    Parameters
    ----------
    X : array_like, shape (n, d)
        First set of points. Must be finite.
    Y : array_like, shape (m, d), optional
        Second set of points. If None, distances are computed between the rows of X and the diagonal is exactly 0.
    metric : {'euclidean', 'sqeuclidean', 'manhattan', 'cosine'}
        Distance to compute. 'cosine' is ``1 - cos(angle)`` in [0, 2]; a zero vector has cosine distance 1 to every other vector.
    batch_size : int, optional
        Rows of X processed at a time by 'manhattan' (>= 1), which bounds the memory of its (batch, m, d) intermediate.
        Ignored by other metrics. Default chooses about 4 million elements per chunk.

    Returns
    -------
    np.ndarray of float64, shape (n, m)
        `D[i, j]` is the distance between `X[i]` and `Y[j]`.

    Raises
    ------
    TypeError
        If `batch_size` is not an integer.
    ValueError
        If an input is not 2-D, not numeric, or contains NaN or infinity, the feature counts differ, `metric` is unknown; or `batch_size` < 1.

    Notes
    -----
    'euclidean' and 'sqeuclidean' use `|x|^2 + |y|^2 - 2 x.y` (one matrix product) after centring both sets on the mean of Y, which removes most of
    the cancellation error, then clip tiny negatives to 0. 'cosine' also uses one matrix product.
    'manhattan' broadcasts in chunks, the only Python loop is over chunks, which exists to bound memory.

    Because of rounding, two identical points in X and Y can get a 'euclidean' distance of about 1e-8 instead of exactly 0.
    Compare distances with a tolerance (e.g. np.isclose with atol=1e-7) rather than ==. The diagonal for Y=None is always exactly 0.

    Time: O(n m d). Space: O(n m) for the result (plus O(b m d) per chunk for 'manhattan').
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    Xa = as_float_array(X, "X", ndim=2, allow_nan=False, allow_inf=False).astype(
        np.float64, copy=False
    )
    self_pair = Y is None
    Ya = (
        Xa
        if self_pair
        else as_float_array(Y, "Y", ndim=2, allow_nan=False, allow_inf=False).astype(
            np.float64, copy=False
        )
    )
    if Xa.shape[1] != Ya.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of features, "
            f"got {Xa.shape[1]} and {Ya.shape[1]}"
        )
    n, d = Xa.shape
    m = Ya.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m))

    if metric in ("euclidean", "sqeuclidean"):
        # centre both sets on Y's mean first: |x|^2 + |y|^2 - 2x.y loses
        # precision when the points are far from the origin
        centre = Ya.mean(axis=0)
        Xc, Yc = Xa - centre, Ya - centre
        xx = np.einsum("ij,ij->i", Xc, Xc)
        yy = xx if self_pair else np.einsum("ij,ij->i", Yc, Yc)
        D = xx[:, None] + yy[None, :] - 2.0 * (Xc @ Yc.T)
        # rounding can leave tiny negatives; clip them so sqrt doesn't give NaN
        np.maximum(D, 0.0, out=D)
        if self_pair:
            np.fill_diagonal(D, 0.0)
        return np.sqrt(D, out=D) if metric == "euclidean" else D

    if metric == "cosine":
        nx = np.sqrt(np.einsum("ij,ij->i", Xa, Xa))
        ny = nx if self_pair else np.sqrt(np.einsum("ij,ij->i", Ya, Ya))
        denom = nx[:, None] * ny[None, :]
        # a zero vector has norm 0, so leave its similarity at 0
        sim = np.divide(Xa @ Ya.T, denom, out=np.zeros((n, m)), where=denom > 0)
        D = np.clip(1.0 - sim, 0.0, 2.0)
        if self_pair:
            np.fill_diagonal(D, 0.0)
        return D

    # manhattan
    if batch_size is None:
        rows = max(1, _CHUNK_ELEMENTS // max(m * d, 1))
    else:
        rows = _check_int(batch_size, "batch_size", 1)
    D = np.empty((n, m))
    for start in range(0, n, rows):  # loop over chunks only, to bound memory
        stop = min(start + rows, n)
        D[start:stop] = np.abs(Xa[start:stop, None, :] - Ya[None, :, :]).sum(axis=2)
    return D


def relu(x):
    """Rectified linear unit, `max(x, 0)`, elementwise.

    Parameters
    ----------
    x : array_like
        Numeric data of any shape.

    Returns
    -------
    np.ndarray of float, same shape as `x`
        `x` where positive, else 0. NaN stays NaN; +inf stays +inf.

    Raises
    ------
    ValueError
        If `x` cannot be converted to float.

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    return np.maximum(as_float_array(x), 0.0)


def leaky_relu(x, alpha=0.01):
    """Leaky rectified linear unit, elementwise.

    Parameters
    ----------
    x : array_like
        Numeric data of any shape.
    alpha : float, default 0.01
        Slope for negative inputs; finite and >= 0.

    Returns
    -------
    np.ndarray of float, same shape as `x`
        `x` where `x >= 0`, else `alpha * x`. NaN stays NaN.

    Raises
    ------
    TypeError
        If `alpha` is not a real number.
    ValueError
        If `x` is not numeric, or `alpha` is negative or not finite.

    Notes
    -----
    Time: O(N). Space: O(N).
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    if not (np.isfinite(alpha) and alpha >= 0):
        raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
    a = as_float_array(x)
    return np.where(a >= 0, a, alpha * a)


def sigmoid(x):
    """Logistic function `1 / (1 + exp(-x))`, elementwise and overflow-free.

    Parameters
    ----------
    x : array_like
        Numeric data of any shape.

    Returns
    -------
    np.ndarray of float, same shape as `x`
        Values in [0, 1]. NaN stays NaN; +inf maps to 1 and -inf to 0.

    Raises
    ------
    ValueError
        If `x` cannot be converted to float.

    Notes
    -----
    Uses `exp(-|x|)` (always <= 1) with the formula chosen by the sign of `x`, so `exp` never overflows even for x = -1000.

    Time: O(N). Space: O(N).
    """
    a = as_float_array(x)
    # exp(-|x|) is at most 1, so it can't overflow
    z = np.exp(-np.abs(a))
    return np.where(a >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def softplus(x):
    """Smooth ReLU `log(1 + exp(x))`, elementwise and overflow-free.

    Parameters
    ----------
    x : array_like
        Numeric data of any shape.

    Returns
    -------
    np.ndarray of float, same shape as `x`
        Values >= 0. NaN stays NaN; +inf maps to +inf and -inf to 0.

    Raises
    ------
    ValueError
        If `x` cannot be converted to float.

    Notes
    -----
    Computed as `max(x, 0) + log1p(exp(-|x|))`, which is exact for large
    |x| and accurate for tiny exp values.

    Time: O(N). Space: O(N).
    """
    a = as_float_array(x)
    return np.maximum(a, 0.0) + np.log1p(np.exp(-np.abs(a)))


def logsumexp(a, axis=None, keepdims=False, nan_policy="propagate"):
    """Stable `log(sum(exp(a)))` along an axis.

    Parameters
    ----------
    a : array_like
        Numeric data of any shape.
    axis : int or None, default None
        Axis to reduce; None reduces over all elements.
    keepdims : bool, default False
        Keep the reduced axis with length 1 so the result broadcasts against `a`.
    nan_policy : {'propagate', 'omit'}, default 'propagate'
        'propagate': a slice containing NaN gives NaN. 'omit': NaN is treated as -inf, i.e. contributes nothing.

    Returns
    -------
    float or np.ndarray of float
        Shape of `a` with `axis` removed (or kept with length 1).
        A slice that is empty or all -inf gives -inf, a slice containing +inf gives +inf.

    Raises
    ------
    TypeError
        If `axis` is not an integer or None, or `keepdims` is not a bool.
    ValueError
        If `a` is not numeric, `axis` is out of range, or `nan_policy` is unknown.

    Notes
    -----
    Subtracts the slice maximum before exponentiating, so `logsumexp([1000, 1000])` is `1000 + log(2)` instead of overflowing.
    A non-finite maximum is replaced by 0 for the shift, which avoids the `-inf - (-inf)` NaN.

    Time: O(N). Space: O(N).
    """
    arr = as_float_array(a, "a")
    ax = check_axis(axis, arr.ndim, allow_none=True)
    if not isinstance(keepdims, (bool, np.bool_)):
        raise TypeError(f"keepdims must be a bool, got {type(keepdims).__name__}")
    if _check_nan_policy(nan_policy) == "omit":
        arr = np.where(np.isnan(arr), -np.inf, arr)
    # shift by the max so exp can't overflow; use 0 if the max is +/-inf,
    # otherwise inf - inf would give NaN
    m = np.max(arr, axis=ax, keepdims=True, initial=-np.inf)
    shift = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        total = np.sum(np.exp(arr - shift), axis=ax, keepdims=True)
        out = np.log(total) + shift
    if keepdims:
        return out
    return out.reshape(())[()] if ax is None else np.squeeze(out, axis=ax)


def softmax(x, axis=-1, nan_policy="propagate"):
    """Stable softmax along an axis, slices sum to 1.

    Parameters
    ----------
    x : array_like, shape (..., L, ...)
        Scores of at least 1-D.
    axis : int, default -1
        Axis along which the probabilities are normalised.
    nan_policy : {'propagate', 'omit'}, default 'propagate'
        'propagate': a slice containing NaN is all NaN. 'omit': NaN entries get probability 0 and the rest is normalised.

    Returns
    -------
    np.ndarray of float, same shape as `x`
        Probabilities in [0, 1] summing to 1 along `axis`.
        A slice that is empty of finite scores (all -inf, or all NaN with 'omit') or that contains +inf is undefined and returned as NaN.

    Raises
    ------
    TypeError
        If `axis` is not an integer.
    ValueError
        If `x` is 0-D or not numeric, `axis` is out of range, or `nan_policy` is unknown.

    Notes
    -----
    `exp(x - max(x))` is used so softmax([1000, 1001]) is finite and the result is unchanged by adding a constant to a slice.

    Time: O(N). Space: O(N).
    """
    arr = as_float_array(x)
    if arr.ndim == 0:
        raise ValueError("x must be at least 1-D")
    ax = check_axis(axis, arr.ndim)
    if _check_nan_policy(nan_policy) == "omit":
        arr = np.where(np.isnan(arr), -np.inf, arr)
    m = np.max(arr, axis=ax, keepdims=True, initial=-np.inf)
    shift = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(arr - shift)
        total = e.sum(axis=ax, keepdims=True)
        # slices with nothing to normalise (total 0) are left as NaN
        out = np.full_like(e, np.nan)
        np.divide(e, total, out=out, where=np.broadcast_to(total > 0, e.shape))
    return out


def log_softmax(x, axis=-1, nan_policy="propagate"):
    """Stable `log(softmax(x))` along an axis.

    Parameters
    ----------
    x : array_like, shape (..., L, ...)
        Scores of at least 1-D.
    axis : int, default -1
        Axis along which to normalise.
    nan_policy : {'propagate', 'omit'}, default 'propagate'
        As in `softmax`; with 'omit', NaN entries give -inf.

    Returns
    -------
    np.ndarray of float, same shape as `x`
        Log-probabilities, <= 0, with `exp` summing to 1 along `axis`.
        Slices where softmax is undefined give NaN.

    Raises
    ------
    TypeError
        If `axis` is not an integer.
    ValueError
        If `x` is 0-D or not numeric, `axis` is out of range, or `nan_policy` is unknown.

    Notes
    -----
    Computed as `x - logsumexp(x)` rather than `log(softmax(x))`,
    so probabilities that underflow to 0 still have a finite log (log_softmax([0, -1000])[1] is -1000).

    Time: O(N). Space: O(N).
    """
    arr = as_float_array(x)
    if arr.ndim == 0:
        raise ValueError("x must be at least 1-D")
    ax = check_axis(axis, arr.ndim)
    if _check_nan_policy(nan_policy) == "omit":
        arr = np.where(np.isnan(arr), -np.inf, arr)
    with np.errstate(invalid="ignore"):
        return arr - logsumexp(arr, axis=ax, keepdims=True)


def topk_along_axis(values, k, axis=-1, largest=True):
    """Select the k largest (or smallest) entries along an axis, for every slice.

    Parameters
    ----------
    values : array_like, shape (..., L, ...)
        Integer or float data of at least 1-D.
    k : int
        Number of entries to keep along `axis`, k >= 0. If k exceeds L, all L entries are returned.
    axis : int, default -1
        Axis to select along.
    largest : bool, default True
        Keep the largest entries if True, the smallest if False.

    Returns
    -------
    top_values : np.ndarray, shape like `values` with `axis` of length k'
        The selected values, best first (descending for largest=True, ascending otherwise), with k' = min(k, L).
    top_indices : np.ndarray of int, same shape as `top_values` Positions of those values along `axis` in the original array.

    Raises
    ------
    TypeError
        If `k` is not an integer, `largest` is not a bool, or `axis` is not an integer.
    ValueError
        If `values` is 0-D or not integer/float data, `k` is negative, or `axis` is out of range.

    Notes
    -----
    NaN ranks below every number, including -inf for largest=True and above every number for largest=False, so NaN is only returned when a slice has
    fewer than k real values. Ties are broken by position: equal values are returned in increasing index order, and when a tie straddles the k-th
    place the earliest positions are selected.
    This matches `sort_search.topk` and is independent of the partition algorithm.

    The selection is fully vectorised: one `np.partition` finds the k-th value of every slice, then boolean masks and cumulative sums choose the
    entries, and only the k selected entries per slice are sorted.

    Time: O(N + S k log k) for S slices. Space: O(N).
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError(f"largest must be a bool, got {type(largest).__name__}")
    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError("values must be at least 1-D")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"values must be integer or float data, got dtype {arr.dtype}")
    ax = check_axis(axis, arr.ndim)

    moved = np.moveaxis(arr, ax, -1)
    lead, length = moved.shape[:-1], moved.shape[-1]
    kk = min(int(k), length)
    if kk == 0:
        shape = lead + (0,)
        out_v = np.moveaxis(np.empty(shape, dtype=arr.dtype), -1, ax)
        out_i = np.moveaxis(np.empty(shape, dtype=np.intp), -1, ax)
        return out_v, out_i

    flat = moved.reshape(-1, length)
    is_nan = (
        np.isnan(flat) if arr.dtype.kind == "f" else np.zeros(flat.shape, dtype=bool)
    )
    worst = -np.inf if largest else np.inf
    key = np.where(is_nan, worst, flat) if arr.dtype.kind == "f" else flat

    pos = length - kk if largest else kk - 1
    thr = np.partition(key, pos, axis=1)[:, pos : pos + 1]
    better = key > thr if largest else key < thr
    need = kk - better.sum(axis=1, keepdims=True)

    # Fill the remaining slots from values tied with the threshold: real
    # values first (earliest position first), then NaN.
    tie = key == thr
    tie_real = tie & ~is_nan
    tie_nan = tie & is_nan
    rank = np.where(
        is_nan,
        tie_real.sum(axis=1, keepdims=True) + np.cumsum(tie_nan, axis=1),
        np.cumsum(tie_real, axis=1),
    )
    selected = better | (tie & (rank <= need))

    idx = np.nonzero(selected)[1].reshape(-1, kk)  # row-major: increasing position
    sel_key = np.take_along_axis(key, idx, axis=1)
    sel_nan = np.take_along_axis(is_nan, idx, axis=1)
    # Best first; descending order via a monotone-decreasing transform that
    # cannot overflow (bitwise not for integers, negation for floats).
    primary = (
        sel_key if not largest else (~sel_key if arr.dtype.kind in "iu" else -sel_key)
    )
    order = np.lexsort((sel_nan, primary), axis=-1)  # stable: ties keep position order
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(flat, idx, axis=1)

    out_v = np.moveaxis(vals.reshape(lead + (kk,)), -1, ax)
    out_i = np.moveaxis(idx.reshape(lead + (kk,)), -1, ax)
    return out_v, out_i


def batch_slices(n, batch_size, drop_last=False):
    """Split `range(n)` into consecutive slices of at most `batch_size`.

    Parameters
    ----------
    n : int
        Number of samples, n >= 0.
    batch_size : int
        Samples per batch, >= 1.
    drop_last : bool, default False
        If True, drop a final batch smaller than `batch_size`.

    Returns
    -------
    list of slice
        `arr[s]` for each slice s covers the rows of one batch, empty if n = 0 (or n < batch_size with drop_last).

    Raises
    ------
    TypeError
        If `n` or `batch_size` is not an integer, or `drop_last` is not a bool.
    ValueError
        If `n` < 0 or `batch_size` < 1.

    Notes
    -----
    The start and stop positions are computed with `np.arange` only the list of slice objects is built in a loop, one per batch.

    Time: O(n / batch_size). Space: O(n / batch_size).

    Examples
    --------
    >>> batch_slices(5, 2)
    [slice(0, 2, None), slice(2, 4, None), slice(4, 5, None)]
    """
    n = _check_int(n, "n", 0)
    batch_size = _check_int(batch_size, "batch_size", 1)
    if not isinstance(drop_last, (bool, np.bool_)):
        raise TypeError(f"drop_last must be a bool, got {type(drop_last).__name__}")
    starts = np.arange(0, n, batch_size)
    stops = np.minimum(starts + batch_size, n)
    if drop_last:
        keep = (stops - starts) == batch_size
        starts, stops = starts[keep], stops[keep]
    return [slice(int(a), int(b)) for a, b in zip(starts, stops)]


def iter_batches(*arrays, batch_size, shuffle=False, seed=None, drop_last=False):
    """Iterate over aligned mini-batches of one or more arrays.

    Parameters
    ----------
    *arrays : array_like
        One or more arrays with the same length n along axis 0 (for example X of shape (n, d) and y of shape (n,)).
    batch_size : int
        Samples per batch, >= 1 (keyword-only).
    shuffle : bool, default False
        If True, visit the samples in a random order, the same for every array so rows stay aligned.
    seed : int, optional
        Seed for the shuffle, so runs are reproducible. Ignored unless `shuffle` is True.
    drop_last : bool, default False
        If True, drop a final batch smaller than `batch_size`.

    Yields
    ------
    tuple of np.ndarray
        One array per input, each with at most `batch_size` rows.
        Without shuffling these are views of the inputs; with shuffling they are copies.

    Raises
    ------
    ValueError
        If no array is given, an array is 0-D, the lengths differ, or `batch_size` < 1.
    TypeError
        If `batch_size` is not an integer, or `shuffle` or `drop_last` is not a bool.

    Notes
    -----
    Arguments are validated when iteration starts. The loop is over batches, which is inherent to mini-batching.

    Time: O(N) in total over all batches (plus O(n) for the permutation when shuffling).
    Space: O(n) for the permutation, O(batch) per batch.

    Examples
    --------
    >>> X, y = np.arange(10).reshape(5, 2), np.arange(5)
    >>> [xb.shape for xb, yb in iter_batches(X, y, batch_size=2)]
    [(2, 2), (2, 2), (1, 2)]
    """
    if len(arrays) == 0:
        raise ValueError("at least one array is required")
    batch_size = _check_int(batch_size, "batch_size", 1)
    for flag, name in ((shuffle, "shuffle"), (drop_last, "drop_last")):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(f"{name} must be a bool, got {type(flag).__name__}")
    arrs = [np.asarray(a) for a in arrays]
    if any(a.ndim == 0 for a in arrs):
        raise ValueError("arrays must be at least 1-D")
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise ValueError(
            f"arrays must have the same length, got {[a.shape[0] for a in arrs]}"
        )
    # one shared permutation keeps the rows of every array aligned
    order = np.random.default_rng(seed).permutation(n) if shuffle else None
    for sl in batch_slices(n, batch_size, drop_last):
        if order is None:
            yield tuple(a[sl] for a in arrs)
        else:
            sel = order[sl]
            yield tuple(a[sel] for a in arrs)