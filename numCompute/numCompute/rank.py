import numpy as np

_METHODS = ("average", "min", "max", "dense", "ordinal")
_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")


def _check(x, method, axis):
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
    x, axis = _check(data, method, axis)
    if x.shape[axis] == 0:
        return np.empty(x.shape, dtype=float)
    moved = np.moveaxis(x, axis, -1)
    return np.moveaxis(_rank_last(moved, method, descending), -1, axis)


def percentile_rank(data, axis=-1, descending=False):
    r = rank(data, "average", axis, descending)
    n = r.shape[axis]
    if n <= 1:
        return np.zeros_like(r)
    return (r - 1.0) / (n - 1.0)


def percentile(data, q, interpolation="linear", axis=-1):
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