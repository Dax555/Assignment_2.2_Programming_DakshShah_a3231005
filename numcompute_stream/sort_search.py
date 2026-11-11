"""
sort_search.py
==============
Sorting, top-k selection and searching utilities built using NumPy.
Sorting part: stable_sort, multi_key_sort.

Conventions
-----------
- stable_sort sorts along `axis` (default -1, the last axis, as in np.sort)
  of an array of any number of dimensions. multi_key_sort takes 1-D keys or
  the columns of a 2-D array. topk, quickselect and binary_search operate on
  1-D arrays; anything else raises ValueError. For several rows at once use
  utils.topk_along_axis.
- NaN is treated as "missing". Sorting puts NaN last; top-k, quickselect and
  binary search ignore NaN and emit a UserWarning saying how many were ignored.
- binary_search expects input that is already sorted (NaN last), as returned
  by stable_sort or np.sort.
- Time/space complexity is given in terms of n, the length of the input.

"""

import warnings

import numpy as np

from .utils import check_axis


def _as_1d_array(values, name="values"):
    """
    Convert the input to a NumPy array and check that it is 1-D.

    Parameters
    ----------
    values : array_like
        Input data.
    name : str, default "values"
        Name used in the error message.

    Returns
    -------
    np.ndarray, shape (n,)
        ``np.asarray(values)``. No copy is made if `values` is already an
        ndarray.

    Raises
    ------
    ValueError
        If the array is not 1-D.

    Notes
    -----
    Time: O(1) for an ndarray, O(n) when converting a list.
    Space: O(1) for an ndarray, O(n) when converting a list.
    """
    arr = np.asarray(values)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")

    return arr


def stable_sort(values, ascending=True, return_indices=False, axis=-1):
    """
    Stable sort along an axis, with NaN always placed last.

    Parameters
    ----------
    values : array_like, at least 1-D
        Data to sort. Any sortable dtype is accepted; only floating dtypes
        can contain NaN.
    ascending : bool, default True
        Sort from smallest to largest if True, otherwise largest to smallest.
    return_indices : bool, default False
        If True, return the sorting permutation instead of the sorted values.
    axis : int, default -1
        Axis to sort along; every 1-D slice along it is sorted on its own.

    Returns
    -------
    np.ndarray, same shape as `values`
        The sorted copy of `values`, or, if `return_indices` is True, an
        integer array `perm` such that
        ``np.take_along_axis(values, perm, axis)`` is the sorted result
        (``values[perm]`` for 1-D input).

    Raises
    ------
    TypeError
        If `ascending` is not a bool, or `axis` is not an integer.
    ValueError
        If `values` is 0-D, or `axis` is out of range.

    Notes
    -----
    Elements that compare equal keep their original relative order, in both
    ascending and descending mode. NaN values are placed after all other
    values in either mode, also in their original relative order.

    One np.argsort(kind="stable") call sorts every slice. NumPy already puts
    NaN last, so descending order sorts a reversed key instead of reversing
    the result: -x for floats (NaN stays NaN) and ~x for integers (no
    overflow, unlike -x). Other dtypes (e.g. strings) cannot hold NaN and use
    the reversed-array trick.

    Time: O(N log n) for N elements and n the length of `axis`. Space: O(N).
    """
    if not isinstance(ascending, (bool, np.bool_)):
        raise TypeError(f"ascending must be a bool, got {type(ascending).__name__}")

    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError("values must be at least 1-D, got a scalar")
    axis = check_axis(axis, arr.ndim)

    if ascending:
        perm = np.argsort(arr, axis=axis, kind="stable")
    elif arr.dtype.kind == "f":
        perm = np.argsort(-arr, axis=axis, kind="stable")
    elif arr.dtype.kind in "iub":
        perm = np.argsort(~arr, axis=axis, kind="stable")
    else:
        # Sort the reversed slices and map the positions back, so ties keep
        # their original order.
        n = arr.shape[axis]
        rev_order = np.argsort(np.flip(arr, axis), axis=axis, kind="stable")
        perm = np.flip((n - 1) - rev_order, axis)

    return perm if return_indices else np.take_along_axis(arr, perm, axis=axis)


def multi_key_sort(keys, ascending=True):
    """
    Sort by several keys; returns the permutation that sorts the data.

    The first key is the primary key, the second breaks ties in the first,
    and so on.

    Parameters
    ----------
    keys : sequence of array_like, each of shape (n,), or np.ndarray of shape (n, K)
        The sort keys, most significant first. All keys must have the same
        length n and at least one key is required. A 2-D NumPy array is read
        column by column: column 0 is the primary key, column 1 the next, and
        so on (so rows are sorted like a table).
    ascending : bool or sequence of bool, default True
        A single bool applies to every key; a sequence must hold one bool per
        key so that each key can have its own direction.

    Returns
    -------
    np.ndarray of int, shape (n,)
        Permutation `perm` such that ``keys[j][perm]`` is ordered by the keys
        in the requested order for every j.

    Raises
    ------
    ValueError
        If `keys` is empty (or a 2-D array with no columns), a key is not 1-D,
        the keys differ in length, or `ascending` is a sequence whose length
        differs from the number of keys.
    TypeError
        If an entry of `ascending` is not a bool.

    Notes
    -----
    Implemented as repeated stable sorts from the least to the most
    significant key. NaN is placed last within each key, in either direction.

    Time: O(K n log n) for K keys. Space: O(n).
    """
    if isinstance(keys, np.ndarray) and keys.ndim == 2:
        # Each column of a 2-D array is one key, the first column most significant.
        keys = list(keys.T)
    if len(keys) == 0:
        raise ValueError("keys must contain at least one array")

    keys = [_as_1d_array(k, f"keys[{i}]") for i, k in enumerate(keys)]
    n = keys[0].shape[0]
    if any(k.shape[0] != n for k in keys):
        raise ValueError("all keys must have the same length")

    if isinstance(ascending, (bool, np.bool_)):
        ascending = [bool(ascending)] * len(keys)
    else:
        ascending = list(ascending)
        if len(ascending) != len(keys):
            raise ValueError(
                f"ascending must be a bool or have one entry per key "
                f"({len(keys)}), got {len(ascending)}"
            )

    perm = np.arange(n)
    for k, asc in zip(reversed(keys), reversed(ascending)):
        local = stable_sort(k[perm], ascending=asc, return_indices=True)
        perm = perm[local]

    return perm


def _check_numeric(arr, func_name):
    """
    Raise TypeError unless `arr` holds integers or floats.

    Parameters
    ----------
    arr : np.ndarray
        Array whose dtype is checked.
    func_name : str
        Name of the calling function, used in the error message.

    Raises
    ------
    TypeError
        If the dtype of `arr` is neither integer nor floating.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise TypeError(
            f"{func_name} needs a numeric (int or float) array, got dtype {arr.dtype}"
        )


def _valid_positions(arr, func_name):
    """
    Positions (in original order) of the non-NaN entries of `arr`.

    If NaNs are present, emit a UserWarning that says how many were ignored.
    Only floating dtypes can hold NaN.

    Parameters
    ----------
    arr : np.ndarray, shape (n,)
        1-D array to scan.
    func_name : str
        Name of the calling function, used in the warning message.

    Returns
    -------
    np.ndarray of int, shape (m,)
        Indices of the non-NaN entries, in increasing order (m <= n).

    Warns
    -----
    UserWarning
        If `arr` contains at least one NaN.

    Notes
    -----
    Time: O(n). Space: O(n).
    """
    if np.issubdtype(arr.dtype, np.floating):
        nan_mask = np.isnan(arr)
        n_nan = int(np.count_nonzero(nan_mask))
        if n_nan:
            warnings.warn(
                f"{func_name}: ignoring {n_nan} NaN value(s) out of {arr.shape[0]}",
                UserWarning,
                stacklevel=3,
            )
        return np.flatnonzero(~nan_mask)
    return np.arange(arr.shape[0])


def topk(values, k, largest=True, return_indices=True):
    """
    Select the k largest (or smallest) values of a 1-D array without sorting
    the whole array.

    Parameters
    ----------
    values : array_like, shape (n,)
        Numeric (integer or float) data. NaN values are ignored.
    k : int
        Number of elements to select, with k >= 0. If k is larger than the
        number of non-NaN values, all non-NaN values are returned.
    largest : bool, default True
        Select the k largest values if True, the k smallest if False.
    return_indices : bool, default True
        Return the positions of the selected values if True, the values
        themselves if False.

    Returns
    -------
    np.ndarray, shape (k',)
        With k' = min(k, number of non-NaN values). Indices (integer dtype,
        positions in the original `values`) or values, ordered best first:
        descending for largest=True, ascending for largest=False. Tied values
        are ordered by their original position, and when a tie straddles the
        k-th place the earliest positions are the ones selected.

    Raises
    ------
    TypeError
        If `k` is not an integer, `largest` or `return_indices` is not a bool,
        or `values` is not numeric.
    ValueError
        If `k` is negative or `values` is not 1-D.

    Warns
    -----
    UserWarning
        If `values` contains NaN.

    Notes
    -----
    Uses np.argpartition to find the boundary value, then sorts only the k
    selected elements.

    Time: O(n + k log k) on average. Space: O(n).
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError(f"largest must be a bool, got {type(largest).__name__}")
    if not isinstance(return_indices, (bool, np.bool_)):
        raise TypeError(
            f"return_indices must be a bool, got {type(return_indices).__name__}"
        )

    arr = _as_1d_array(values)
    _check_numeric(arr, "topk")

    ok_idx = _valid_positions(arr, "topk")      # warns if NaNs are present
    ok_vals = arr[ok_idx]
    m = ok_vals.shape[0]

    k = min(int(k), m)

    if k == 0:
        idx = np.empty(0, dtype=ok_idx.dtype)
    else:
        # Position of the boundary value in ascending order (0-based).
        kth = (m - k) if largest else (k - 1)
        part = np.argpartition(ok_vals, kth)
        threshold = ok_vals[part[kth]]

        if largest:
            sel_mask = ok_vals > threshold          # strictly better
        else:
            sel_mask = ok_vals < threshold
        needed = k - int(np.count_nonzero(sel_mask))

        # Fill the remaining slots from values tied with the threshold,
        # earliest positions first.
        tied = np.flatnonzero(ok_vals == threshold)[:needed]
        sel_mask[tied] = True

        local_sel = np.flatnonzero(sel_mask)        # original order
        order = stable_sort(ok_vals[local_sel], ascending=not largest,
                            return_indices=True)
        idx = ok_idx[local_sel[order]]

    return idx if return_indices else arr[idx]


def quickselect(values, k):
    """
    Return the k-th smallest value of a 1-D array (0-based k) without sorting.

    Parameters
    ----------
    values : array_like, shape (n,)
        Numeric (integer or float) data. NaN values are ignored.
    k : int
        0-based rank to select among the non-NaN values, so k = 0 is the
        minimum and k = m - 1 the maximum, where m is the number of non-NaN
        values.

    Returns
    -------
    scalar
        The k-th smallest non-NaN value, as a NumPy scalar of the dtype of
        `values`.

    Raises
    ------
    TypeError
        If `k` is not an integer or `values` is not numeric.
    ValueError
        If `values` is not 1-D, or `k` is outside 0 <= k < m (this includes
        the case where every value is NaN or the array is empty).

    Warns
    -----
    UserWarning
        If `values` contains NaN.

    Notes
    -----
    Randomised quickselect with a three-way partition (less than, equal to,
    greater than the pivot), so many duplicates do not slow it down. The pivot
    is random, but the returned value does not depend on it.

    Time: O(n) expected, O(n^2) worst case. Space: O(n).
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")

    arr = _as_1d_array(values)
    _check_numeric(arr, "quickselect")

    work = arr[_valid_positions(arr, "quickselect")]   # warns if NaNs are present
    m = work.shape[0]
    if not 0 <= k < m:
        raise ValueError(
            f"k must satisfy 0 <= k < {m} (number of non-NaN values), got {k}"
        )
    k = int(k)

    rng = np.random.default_rng()
    while True:
        pivot = work[rng.integers(work.shape[0])]
        less = work[work < pivot]
        n_less = less.shape[0]
        n_equal = int(np.count_nonzero(work == pivot))

        if k < n_less:                       # answer is among the smaller values
            work = less
        elif k < n_less + n_equal:           # answer equals the pivot
            return pivot
        else:
            k -= n_less + n_equal
            work = work[work > pivot]


def binary_search(sorted_values, x, return_index=True, method="loop", check_sorted=False):
    """
    Binary search for x in a 1-D array that is already sorted in ascending order.

    Returns (index, found) with the leftmost insertion index, or
    (value, found) with the matching value (None if absent) when
    return_index is False. NaNs at the end of the array are ignored with a
    warning.

    Parameters
    ----------
    sorted_values : array_like, shape (n,)
        Data sorted in ascending order, with any NaN at the end (as returned
        by stable_sort or np.sort). It is not sorted here; pass
        check_sorted=True to have this checked.
    x : scalar
        Value to look for. Must not be NaN.
    return_index : bool, default True
        If True, return the insertion index; if False, return the matching
        value (or None).
    method : {"loop", "searchsorted"}, default "loop"
        "loop" is a hand-written binary search, "searchsorted" uses
        np.searchsorted. Both give the same answer.
    check_sorted : bool, default False
        If True, first check that `sorted_values` is sorted with NaN only at
        the end. This costs O(n), so it is off by default.

    Returns
    -------
    index : int
        Leftmost position at which `x` could be inserted to keep
        `sorted_values` ordered (ignoring the NaNs at the end). Returned if
        `return_index` is True.
    value : scalar or None
        The element equal to `x`, or None if absent. Returned instead of
        `index` if `return_index` is False.
    found : bool
        True if `x` occurs in `sorted_values`.

    Raises
    ------
    TypeError
        If `return_index` or `check_sorted` is not a bool.
    ValueError
        If `x` is not a scalar, `x` is NaN, `sorted_values` is not 1-D,
        `method` is unknown, or `check_sorted` is True and the input is not
        sorted.

    Warns
    -----
    UserWarning
        If `sorted_values` ends with NaN.

    Notes
    -----
    The input is not sorted here; unsorted input gives a meaningless answer
    unless check_sorted=True catches it. The number of NaNs is found with a
    second binary search (NaN sorts last), so no O(n) scan is needed.

    Time: O(log n), or O(n) with check_sorted=True.
    Space: O(1).
    """
    if not isinstance(return_index, (bool, np.bool_)):
        raise TypeError(f"return_index must be a bool, got {type(return_index).__name__}")
    if not isinstance(check_sorted, (bool, np.bool_)):
        raise TypeError(f"check_sorted must be a bool, got {type(check_sorted).__name__}")
    if method not in ("loop", "searchsorted"):
        raise ValueError(f"method must be 'loop' or 'searchsorted', got {method!r}")
    if np.ndim(x) != 0:
        raise ValueError("x must be a scalar")
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        raise ValueError("x must not be NaN")

    arr = _as_1d_array(sorted_values, "sorted_values")
    n = arr.shape[0]
    if np.issubdtype(arr.dtype, np.floating) and n and np.isnan(arr[-1]):
        # NaN sorts last, so the valid values are the ones before the first NaN.
        n = int(np.searchsorted(arr, np.nan, side="left"))
        warnings.warn(
            f"binary_search: ignoring {arr.shape[0] - n} NaN value(s) out of {arr.shape[0]}",
            UserWarning,
            stacklevel=2,
        )
    valid = arr[:n]

    if check_sorted:
        has_nan = np.issubdtype(valid.dtype, np.floating) and np.isnan(valid).any()
        if has_nan or np.any(valid[1:] < valid[:-1]):
            raise ValueError("sorted_values must be sorted in ascending order with NaN last")

    if method == "searchsorted":
        low = int(np.searchsorted(valid, x, side="left"))
    else:
        low, high = 0, n
        while low < high:
            mid = (low + high) // 2
            if valid[mid] < x:
                low = mid + 1
            else:
                high = mid

    found = bool(low < n and valid[low] == x)

    if return_index:
        return low, found
    return (valid[low] if found else None), found