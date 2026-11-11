"""
optim.py
========
Finite-difference derivatives built using NumPy: grad and jacobian.

- 'central':  (f(x + h e_i) - f(x - h e_i)) / (2h)  error O(h^2), 2n evaluations.
- 'forward':  (f(x + h e_i) - f(x)) / h             error O(h),   n + 1 evaluations.

The function is called with a fresh 1-D float array every time, so it may
modify its argument without affecting x. With relative=True the step for
x[j] is h * max(1, |x[j]|) instead of a fixed h. A function that returns NaN or
infinity at any evaluation point raises ValueError.

Conventions
-----------
- n is the number of variables (length of x) and m the number of outputs of F.
- C_f denotes the cost of one call to the user's function.

"""

import numpy as np

__all__ = ["grad", "jacobian"]

_METHODS = ("central", "forward")


def _check_callable(func, name):
    """
    Raise TypeError unless `func` is callable.

    Parameters
    ----------
    func : object
        Object to check.
    name : str
        Argument name used in the error message.

    Raises
    ------
    TypeError
        If `func` is not callable.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if not callable(func):
        raise TypeError(f"{name} must be callable, got {type(func).__name__}")


def _as_point(x):
    """
    Copy x into a 1-D float array and check that every value is finite.

    Parameters
    ----------
    x : array_like, shape (n,)
        The point at which a derivative is estimated.

    Returns
    -------
    np.ndarray of float, shape (n,)
        A new array, so later changes to it never affect the caller's `x`.

    Raises
    ------
    ValueError
        If `x` cannot be converted to float, is not 1-D, or contains NaN or
        infinity.

    Notes
    -----
    Time: O(n). Space: O(n).
    """
    try:
        arr = np.array(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"x must be numeric; could not convert to float ({exc})") from exc
    if arr.ndim != 1:
        raise ValueError(
            f"x must be 1-D, got shape {arr.shape}; pass [value] for a single variable"
        )
    if not np.isfinite(arr).all():
        raise ValueError("x must contain only finite values")
    return arr


def _check_step(h):
    """
    Validate the finite-difference step size.

    Parameters
    ----------
    h : int or float
        Step size. Booleans are rejected.

    Returns
    -------
    float
        `h` as a Python float.

    Raises
    ------
    TypeError
        If `h` is not a real number.
    ValueError
        If `h` is not finite and strictly positive.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, float, np.integer, np.floating)):
        raise TypeError(f"h must be a number, got {type(h).__name__}")
    h = float(h)
    if not (np.isfinite(h) and h > 0):
        raise ValueError(f"h must be a positive finite number, got {h}")
    return h

def _step_sizes(x, h, relative):
    """
    Step size for each variable.

    Parameters
    ----------
    x : np.ndarray of float, shape (n,)
        Validated evaluation point.
    h : float
        Validated base step size (> 0).
    relative : bool
        If True, scale the step by max(1, |x[j]|).

    Returns
    -------
    np.ndarray of float, shape (n,)
        `h` for every variable, or h * max(1, |x[j]|) when `relative` is True.

    Raises
    ------
    TypeError
        If `relative` is not a bool.

    Notes
    -----
    A fixed h is too small next to a large |x[j]|: near 1e10 neighbouring
    floats are about 1.9e-6 apart, so x + h is rounded, and the change in f
    is lost in f's own rounding error. Scaling the step with |x[j]| keeps
    both changes well above rounding level, and the max(1, ...) keeps the
    step from shrinking to nothing near x = 0.

    Time: O(n). Space: O(n).
    """
    if not isinstance(relative, (bool, np.bool_)):
        raise TypeError(f"relative must be a bool, got {type(relative).__name__}")
    if relative:
        return h * np.maximum(1.0, np.abs(x))
    return np.full(x.shape, h)

def _check_method(method):
    """
    Raise ValueError unless `method` is 'central' or 'forward'.

    Parameters
    ----------
    method : str
        Name of the difference scheme.

    Raises
    ------
    ValueError
        If `method` is not one of the supported schemes.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


def _evaluate(func, point, name, scalar):
    """
    Call `func` once and return its result as a validated 1-D float array.

    Parameters
    ----------
    func : callable
        Function to evaluate.
    point : np.ndarray, shape (n,)
        Argument passed to `func`.
    name : str
        Name of the user's function, used in error messages.
    scalar : bool
        If True the result must be a single number; if False it may be a
        scalar or a 1-D array.

    Returns
    -------
    np.ndarray of float, shape (m,)
        The result, with m = 1 for a scalar.

    Raises
    ------
    ValueError
        If the result is not numeric, has the wrong shape (not a scalar when
        `scalar` is True, more than 1-D otherwise), or contains NaN or
        infinity.

    Notes
    -----
    Time: O(C_f + m). Space: O(m).
    """
    result = func(point)
    try:
        value = np.asarray(result, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must return numbers; could not convert its result ({exc})") from exc
    if scalar:
        if value.size != 1:
            raise ValueError(
                f"{name} must return a scalar, got shape {value.shape}; "
                "use jacobian for vector-valued functions"
            )
        value = value.reshape(1)
    else:
        if value.ndim > 1:
            raise ValueError(f"{name} must return a scalar or a 1-D array, got shape {value.shape}")
        value = np.atleast_1d(value)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} returned a non-finite value at x = {point.tolist()}")
    return value


def _stack(values, name):
    """
    Stack equal-length 1-D result arrays into one 2-D array.

    Parameters
    ----------
    values : list of np.ndarray, each of shape (m,)
        Results of successive function evaluations.
    name : str
        Name of the user's function, used in the error message.

    Returns
    -------
    np.ndarray, shape (len(values), m)
        The results stacked row by row.

    Raises
    ------
    ValueError
        If the arrays do not all have the same length.

    Notes
    -----
    Time: O(n m) for n arrays of length m. Space: O(n m).
    """
    if len({v.shape[0] for v in values}) > 1:
        raise ValueError(f"{name} must return the same number of outputs at every point")
    return np.stack(values)


def _differences(func, x, h, method, name, scalar):
    """
    Finite-difference estimate of the derivative of every output w.r.t. every input.

    Parameters
    ----------
    func : callable
        Function mapping a 1-D array of shape (n,) to a scalar or a 1-D array
        of shape (m,).
    x : np.ndarray of float, shape (n,)
        Validated evaluation point.
    h : np.ndarray of float, shape (n,)
        Step size for each variable (> 0).
    method : {'central', 'forward'}
        Difference scheme.
    name : str
        Name of the user's function, used in error messages.
    scalar : bool
        If True, `func` must return a scalar and m = 1.

    Returns
    -------
    np.ndarray, shape (n, m)
        Entry [j, i] estimates the derivative of output i with respect to
        x[j]. For n = 0 the result has shape (0, m).

    Raises
    ------
    ValueError
        Propagated from `_evaluate` and `_stack`, or if `func` returns a
        different number of outputs at different points.

    Notes
    -----
    'central' calls `func` 2n times, 'forward' n + 1 times (n = 0 needs at
    most one call, only to find m).

    Time: O(n C_f + n^2 + n m). Space: O(n^2 + n m); the n^2 term is the
    identity matrix whose rows give the step directions.
    """
    n = x.shape[0]
    if n == 0:
        m = 1 if scalar else _evaluate(func, x.copy(), name, scalar).shape[0]
        return np.empty((0, m))

    eye = np.eye(n)
    plus = _stack([_evaluate(func, x + h[j] * eye[j], name, scalar) for j in range(n)], name)

    if method == "central":
        minus = _stack([_evaluate(func, x - h[j] * eye[j], name, scalar) for j in range(n)], name)
        if minus.shape != plus.shape:
            raise ValueError(f"{name} must return the same number of outputs at every point")
        return (plus - minus) / (2.0 * h[:, None])

    base = _evaluate(func, x.copy(), name, scalar)
    if base.shape[0] != plus.shape[1]:
        raise ValueError(f"{name} must return the same number of outputs at every point")
    return (plus - base) / h[:, None]


def grad(f, x, h=1e-5, method="central", relative=False):
    """
    Gradient of a scalar function f at x, estimated with finite differences.

    f maps a 1-D array of shape (n,) to a number. Returns an array of shape
    (n,). 'central' costs 2n calls to f and is much more accurate than
    'forward', which costs n + 1 calls. Raises ValueError if f returns a
    vector (use jacobian) or a non-finite value.

    Parameters
    ----------
    f : callable
        Scalar function ``f(x)`` taking a float array of shape (n,) and
        returning a single finite number.
    x : array_like, shape (n,)
        Point at which to estimate the gradient. Must be 1-D and finite;
        pass ``[value]`` for a function of one variable.
    h : float, default 1e-5
        Step size, finite and > 0. Too large a step increases truncation
        error, too small a step increases round-off error.
    method : {'central', 'forward'}, default 'central'
        Difference scheme.
    relative : bool, default False
        If True, the step for x[j] is h * max(1, |x[j]|), which keeps the
        estimate accurate when x has very large entries.

    Returns
    -------
    np.ndarray of float, shape (n,)
        Estimate of the gradient; entry j approximates df/dx[j]. Empty for
        n = 0.

    Raises
    ------
    TypeError
        If `f` is not callable, `h` is not a number, or `relative` is not abool.
        ValueError
        If `x` is not numeric, 1-D and finite; `h` is not positive and
        finite; `method` is not supported; or `f` returns a non-scalar or a
        NaN/infinite value.

    Notes
    -----
    Error is O(h^2) for 'central' and O(h) for 'forward'.

    Time: O(n C_f + n^2) with 2n ('central') or n + 1 ('forward') calls to
    `f`. Space: O(n^2).
    """
    _check_callable(f, "f")
    point = _as_point(x)
    h = _check_step(h)
    _check_method(method)
    steps = _step_sizes(point, h, relative)
    return np.ascontiguousarray(_differences(f, point, steps, method, "f", True)[:, 0])

def jacobian(F, x, h=1e-5, method="central", relative=False):
    """
    Jacobian of a vector function F at x, estimated with finite differences.

    F maps a 1-D array of shape (n,) to a 1-D array of shape (m,) (a scalar
    counts as m = 1). Returns an array of shape (m, n) with J[i, j] equal to
    the derivative of F_i with respect to x_j. Same cost and accuracy as grad.

    Parameters
    ----------
    F : callable
        Function ``F(x)`` taking a float array of shape (n,) and returning a
        scalar or a 1-D array of shape (m,) of finite numbers. The output
        length must be the same at every point.
    x : array_like, shape (n,)
        Point at which to estimate the Jacobian. Must be 1-D and finite.
    h : float, default 1e-5
        Step size, finite and > 0.
    method : {'central', 'forward'}, default 'central'
        Difference scheme.
    relative : bool, default False
        If True, the step for x[j] is h * max(1, |x[j]|), as in `grad`.

    Returns
    -------
    np.ndarray of float, shape (m, n)
        Estimate of the Jacobian matrix, with J[i, j] approximating
        dF_i/dx_j. Shape (m, 0) for n = 0.

    Raises
    ------
    TypeError
        If `F` is not callable, `h` is not a number, or `relative` is not abool.
        ValueError
        If `x` is not numeric, 1-D and finite; `h` is not positive and
        finite; `method` is not supported; `F` returns more than a 1-D array,
        a non-finite value, or a different number of outputs at different
        points.

    Notes
    -----
    Error is O(h^2) for 'central' and O(h) for 'forward'.

    Time: O(n C_F + n^2 + n m) with 2n ('central') or n + 1 ('forward')
    calls to `F`. Space: O(n^2 + n m).
    """
    _check_callable(F, "F")
    point = _as_point(x)
    h = _check_step(h)
    _check_method(method)
    steps = _step_sizes(point, h, relative)
    return np.ascontiguousarray(_differences(F, point, steps, method, "F", False).T)