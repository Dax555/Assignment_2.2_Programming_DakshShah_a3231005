"""
preprocessing.py
================
Streaming preprocessing built using NumPy.
Classes: StandardScaler, MinMaxScaler, Imputer, OneHotEncoder,
ColumnTransformer.

Every transformer works in two modes:

- batch     : fit(X) then transform(X), or fit_transform(X).
- streaming : partial_fit(chunk) (alias update) once per chunk. Statistics are
  merged chunk by chunk and old data is never needed.

API (all classes)
-----------------
- fit(X) -> self             forget everything, then learn from X
- partial_fit(X) -> self     learn from one more chunk (alias: update)
- transform(X) -> X_out      apply what has been learned so far
- fit_transform(X) -> X_out  fit, then transform
- reset() -> self            forget everything

Conventions
-----------
- X is 2-D with shape (n_samples, n_features). Statistics are per feature
  (along axis 0) and transform returns the same number of rows.
- A chunk may have zero rows; it is accepted and changes nothing.
- The number of features is fixed by the first chunk; a different number
  later raises ValueError.
- Missing values are NaN in float columns and "", "nan", "none", "null", "na"
  in text columns. +/-inf is rejected with ValueError.
- Everything is vectorised over samples. The only Python loops run over
  columns (categories differ per column) or over the steps of a chain,
  never over rows.
- Complexity is given in terms of n (rows in the chunk) and d (features).
  C is the total number of categories over all columns.
"""

import numpy as np

_MISSING_TOKENS = ("", "nan", "none", "null", "na")


def _as_float_2d(X, name="X"):
    """
    Validate X as a finite-or-NaN float array of shape (n_samples, n_features).

    Parameters
    ----------
    X : array_like
        Input data.
    name : str, default "X"
        Name used in error messages.

    Returns
    -------
    np.ndarray of float, shape (n, d)
        `X` converted to float. NaN is allowed.

    Raises
    ------
    ValueError
        If `X` cannot be converted to float, is not 2-D, or contains
        infinity.

    Notes
    -----
    Time: O(n d).
    Space: O(1) if `X` is already float, otherwise O(n d).
    """
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; could not convert to float ({exc})") from exc
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {X.shape}")
    if np.isinf(X).any():
        raise ValueError(f"{name} contains infinity; remove or replace it before using it")
    return X


def _as_2d(X, name="X"):
    """
    Validate X as a 2-D array of any dtype (used for text or mixed data).

    Parameters
    ----------
    X : array_like
        Input data.
    name : str, default "X"
        Name used in error messages.

    Returns
    -------
    np.ndarray, shape (n, d)
        `np.asarray(X)`; no copy if `X` is already an array.

    Raises
    ------
    ValueError
        If `X` is not 2-D.

    Notes
    -----
    Time: O(1) for an ndarray, O(n d) for a list.
    Space: O(1) for an ndarray, O(n d) for a list.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {X.shape}")
    return X


def _check_n_features(expected, X):
    """
    Check that X has the expected number of features.

    Parameters
    ----------
    expected : int
        Number of features fixed by the first chunk.
    X : np.ndarray, shape (n, d)
        Array to check.

    Raises
    ------
    ValueError
        If the number of columns of `X` differs from `expected`.

    Notes
    -----
    Time: O(1).
    Space: O(1).
    """
    if X.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, got {X.shape[1]}")


def _as_str_column(col):
    """
    Convert a 1-D array of any dtype to stripped unicode strings.

    Parameters
    ----------
    col : array_like, shape (n,)
        One column of data.

    Returns
    -------
    np.ndarray of str, shape (n,)
        The values as strings with surrounding whitespace removed.

    Notes
    -----
    Time: O(n).
    Space: O(n).
    """
    return np.char.strip(np.asarray(col).astype(str))


def _is_missing(str_col):
    """
    Mark which strings count as missing values.

    Missing means "", "nan", "none", "null" or "na" in any letter case.

    Parameters
    ----------
    str_col : np.ndarray of str, shape (n,)
        Strings, already stripped.

    Returns
    -------
    np.ndarray of bool, shape (n,)
        True where the entry is a missing-value token.

    Notes
    -----
    Time: O(n).
    Space: O(n).
    """
    return np.isin(np.char.lower(str_col), _MISSING_TOKENS)


class StandardScaler:
    """
    Standardise features to zero mean and unit variance, incrementally.

    Running statistics are kept per feature as a count, a mean and the sum of
    squared deviations from the mean (M2). A new chunk is merged with Chan et
    al.'s parallel update, which is numerically stable. Accumulating a sum and
    a sum of squares instead loses precision when the mean is large compared
    with the spread.

    Parameters
    ----------
    with_mean : bool, default True
        Subtract the mean.
    with_std : bool, default True
        Divide by the standard deviation.

    Attributes
    ----------
    n_ : np.ndarray, shape (d,)
        Non-NaN values seen per feature.
    mean_ : np.ndarray, shape (d,)
        Running mean.
    M2_ : np.ndarray, shape (d,)
        Running sum of squared deviations.
    var_ : np.ndarray, shape (d,)
        Population variance (ddof=0), M2_ / n_.
    scale_ : np.ndarray, shape (d,)
        Standard deviation used for scaling; 1.0 where the variance is zero.
    n_features_ : int or None
        Number of features, fixed by the first chunk.

    Notes
    -----
    NaN is ignored when fitting and passed through by `transform`. A
    zero-variance feature is mapped to 0 (the scale is set to 1, so there is
    never a division by zero).
    """
    def __init__(self, with_mean=True, with_std=True):
        """
        Create an unfitted scaler.

        Parameters
        ----------
        with_mean : bool, default True
            Subtract the mean.
        with_std : bool, default True
            Divide by the standard deviation.

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.reset()

    def reset(self):
        """
        Forget all learned statistics.

        Returns
        -------
        StandardScaler
            The same object (`self`).

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        self.n_features_ = None
        self.n_ = self.mean_ = self.M2_ = None
        return self

    @property
    def var_(self):
        """
        Population variance of each feature (0 where nothing has been seen).

        Returns
        -------
        np.ndarray, shape (d,)
            M2_ / n_, computed without dividing by zero.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(d).
        Space: O(d).
        """
        self._check_fitted()
        out = np.zeros_like(self.M2_)
        np.divide(self.M2_, self.n_, out=out, where=self.n_ > 0)
        return out

    @property
    def scale_(self):
        """
        Standard deviation used for scaling.

        Returns
        -------
        np.ndarray, shape (d,)
            sqrt(var_), with zeros replaced by 1.0.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(d).
        Space: O(d).
        """
        std = np.sqrt(self.var_)
        std[std == 0.0] = 1.0
        return std

    def partial_fit(self, X):
        """
        Merge one chunk into the running statistics.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Chunk of data; NaN is allowed. May have n = 0.

        Returns
        -------
        StandardScaler
            The same object (`self`).

        Raises
        ------
        ValueError
            If `X` is not 2-D, not numeric, contains infinity, or has a
            different number of features than the first chunk.

        Notes
        -----
        Computes the count, mean and M2 of the chunk (ignoring NaN) and merges
        them with the running values using Chan's formula. `update` is an alias.

        Time: O(n d).
        Space: O(n d) temporary, O(d) stored.
        """
        X = _as_float_2d(X)
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self.n_ = np.zeros(self.n_features_)
            self.mean_ = np.zeros(self.n_features_)
            self.M2_ = np.zeros(self.n_features_)
        else:
            _check_n_features(self.n_features_, X)

        valid = ~np.isnan(X)
        n_b = valid.sum(axis=0).astype(float)
        total = np.where(valid, X, 0.0).sum(axis=0)
        mean_b = np.divide(total, n_b, out=np.zeros_like(total), where=n_b > 0)
        M2_b = (np.where(valid, X - mean_b, 0.0) ** 2).sum(axis=0)

        n_a = self.n_
        n = n_a + n_b
        ratio = np.divide(n_b, n, out=np.zeros_like(n), where=n > 0)
        delta = mean_b - self.mean_
        self.mean_ = self.mean_ + delta * ratio
        self.M2_ = self.M2_ + M2_b + delta ** 2 * n_a * ratio
        self.n_ = n
        return self

    update = partial_fit

    def fit(self, X):
        """
        Forget everything, then learn from X.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training data.

        Returns
        -------
        StandardScaler
            The same object (`self`).

        Raises
        ------
        ValueError
            As for `partial_fit`.

        Notes
        -----
        Time: O(n d).
        Space: O(n d) temporary, O(d) stored.
        """
        return self.reset().partial_fit(X)

    def transform(self, X):
        """
        Standardise X with the statistics learned so far.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to transform; NaN is passed through.

        Returns
        -------
        np.ndarray, shape (n, d)
            (X - mean_) / scale_, according to `with_mean` and `with_std`.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.
        ValueError
            If `X` is not 2-D, not numeric, contains infinity, or has the
            wrong number of features.

        Notes
        -----
        Time: O(n d).
        Space: O(n d).
        """
        self._check_fitted()
        X = _as_float_2d(X)
        _check_n_features(self.n_features_, X)
        out = X.copy()
        if self.with_mean:
            out -= self.mean_
        if self.with_std:
            out /= self.scale_
        return out

    def fit_transform(self, X):
        """
        Fit on X, then transform it.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to fit and transform.

        Returns
        -------
        np.ndarray, shape (n, d)
            The standardised data.

        Raises
        ------
        ValueError
            As for `fit` and `transform`.

        Notes
        -----
        Time: O(n d).
        Space: O(n d).
        """
        return self.fit(X).transform(X)

    def _check_fitted(self):
        """
        Raise an error if no data has been seen yet.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        if self.n_features_ is None:
            raise RuntimeError("StandardScaler is not fitted; call fit/partial_fit first")


class MinMaxScaler:
    """
    Scale each feature to `feature_range` using a running minimum and maximum.

    A stream can later show values outside the range seen so far, so the
    output may fall outside `feature_range` unless clip=True.

    Parameters
    ----------
    feature_range : tuple (min, max), default (0.0, 1.0)
        Target range; min must be smaller than max.
    clip : bool, default False
        Limit the output to `feature_range`.

    Attributes
    ----------
    n_ : np.ndarray, shape (d,)
        Non-NaN values seen per feature.
    min_ : np.ndarray, shape (d,)
        Running minimum.
    max_ : np.ndarray, shape (d,)
        Running maximum.
    n_features_ : int or None
        Number of features, fixed by the first chunk.

    Notes
    -----
    A feature that is constant so far maps to the lower bound. A feature
    that has not seen any valid value yet passes through as NaN. NaN is
    ignored when fitting.
    """
    def __init__(self, feature_range=(0.0, 1.0), clip=False):
        """
        Create an unfitted scaler.

        Parameters
        ----------
        feature_range : tuple (min, max), default (0.0, 1.0)
            Target range.
        clip : bool, default False
            Limit the output to `feature_range`.

        Raises
        ------
        ValueError
            If `feature_range` does not satisfy min < max.

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        lo, hi = feature_range
        if not lo < hi:
            raise ValueError(f"feature_range must satisfy min < max, got {feature_range!r}")
        self.feature_range = (float(lo), float(hi))
        self.clip = clip
        self.reset()

    def reset(self):
        """
        Forget all learned statistics.

        Returns
        -------
        MinMaxScaler
            The same object (`self`).

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        self.n_features_ = None
        self.n_ = self.min_ = self.max_ = None
        return self

    def partial_fit(self, X):
        """
        Update the running minimum and maximum with one chunk.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Chunk of data; NaN is allowed. May have n = 0.

        Returns
        -------
        MinMaxScaler
            The same object (`self`).

        Raises
        ------
        ValueError
            If `X` is not 2-D, not numeric, contains infinity, or has a
            different number of features than the first chunk.

        Notes
        -----
        An empty chunk is a no-op. `update` is an alias.

        Time: O(n d).
        Space: O(n d) temporary, O(d) stored.
        """
        X = _as_float_2d(X)
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self.n_ = np.zeros(self.n_features_)
            self.min_ = np.full(self.n_features_, np.inf)
            self.max_ = np.full(self.n_features_, -np.inf)
        else:
            _check_n_features(self.n_features_, X)
        valid = ~np.isnan(X)
        self.n_ = self.n_ + valid.sum(axis=0)
        self.min_ = np.minimum(self.min_, np.where(valid, X, np.inf).min(axis=0, initial=np.inf))
        self.max_ = np.maximum(self.max_, np.where(valid, X, -np.inf).max(axis=0, initial=-np.inf))
        return self

    update = partial_fit

    def fit(self, X):
        """
        Forget everything, then learn from X.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training data.

        Returns
        -------
        MinMaxScaler
            The same object (`self`).

        Raises
        ------
        ValueError
            As for `partial_fit`.

        Notes
        -----
        Time: O(n d).
        Space: O(n d) temporary, O(d) stored.
        """
        return self.reset().partial_fit(X)

    def transform(self, X):
        """
        Scale X to `feature_range` using the minimum and maximum seen so far.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to transform.

        Returns
        -------
        np.ndarray, shape (n, d)
            Scaled data. Constant features give the lower bound; features with
            no valid value so far give NaN.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.
        ValueError
            If `X` is not 2-D, not numeric, contains infinity, or has the
            wrong number of features.

        Notes
        -----
        Time: O(n d).
        Space: O(n d).
        """
        self._check_fitted()
        X = _as_float_2d(X)
        _check_n_features(self.n_features_, X)
        lo, hi = self.feature_range
        seen = self.n_ > 0
        data_min = np.where(seen, self.min_, 0.0)
        data_range = np.where(seen, self.max_ - self.min_, 0.0)
        scale = np.divide(hi - lo, data_range, out=np.zeros_like(data_range),
                          where=data_range > 0)
        out = (X - data_min) * scale + lo
        out = np.where(seen, out, np.nan)
        return np.clip(out, lo, hi) if self.clip else out

    def fit_transform(self, X):
        """
        Fit on X, then transform it.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to fit and transform.

        Returns
        -------
        np.ndarray, shape (n, d)
            The scaled data.

        Raises
        ------
        ValueError
            As for `fit` and `transform`.

        Notes
        -----
        Time: O(n d).
        Space: O(n d).
        """
        return self.fit(X).transform(X)

    def _check_fitted(self):
        """
        Raise an error if no data has been seen yet.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        if self.n_features_ is None:
            raise RuntimeError("MinMaxScaler is not fitted; call fit/partial_fit first")


class Imputer:
    """
    Fill missing values, learning the fill value from the stream.

    Parameters
    ----------
    strategy : {"mean", "constant", "most_frequent"}, default "mean"
        "mean": running mean per column (numeric data only).
        "constant": always `fill_value`.
        "most_frequent": running mode per column (ties go to the smallest
        value); works for float and for text columns.
    fill_value : scalar, default 0.0
        Used by "constant", and by "mean" for a column that has not seen
        any valid value yet.

    Attributes
    ----------
    n_ : np.ndarray, shape (d,)
        Non-NaN values seen per column (strategy "mean").
    mean_ : np.ndarray, shape (d,)
        Running mean per column (strategy "mean").
    statistics_ : list
        Current fill value per column (None = unknown yet).
    n_features_ : int or None
        Number of features, fixed by the first chunk.

    Notes
    -----
    Missing means NaN in float columns and "", "nan", "none", "null", "na"
    in text columns. The median is not offered because it cannot be computed
    exactly in one pass over a stream without storing the data. A
    "most_frequent" column with nothing seen yet is left unchanged. For
    high-cardinality numeric columns prefer "mean", because "most_frequent"
    keeps one counter per distinct value.
    """
    _STRATEGIES = ("mean", "constant", "most_frequent")

    def __init__(self, strategy="mean", fill_value=0.0):
        """
        Create an unfitted imputer.

        Parameters
        ----------
        strategy : str, default "mean"
            One of "mean", "constant", "most_frequent".
        fill_value : scalar, default 0.0
            Fill value for "constant" and for unseen "mean" columns.

        Raises
        ------
        ValueError
            If `strategy` is not one of the supported names.

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        if strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.fill_value = fill_value
        self.reset()

    def reset(self):
        """
        Forget all learned statistics.

        Returns
        -------
        Imputer
            The same object (`self`).

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        self.n_features_ = None
        self.n_ = self.mean_ = None
        self._counts = None
        return self

    @property
    def statistics_(self):
        """
        Current fill value for each column.

        Returns
        -------
        list, length d
            Floats for "mean", `fill_value` for "constant", the mode (or None if
            nothing has been seen) for "most_frequent".

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(d) for "mean" and "constant"; O(D log D) for "most_frequent", where D is the number of distinct values.
        Space: O(d).
        """
        self._check_fitted()
        if self.strategy == "constant":
            return [self.fill_value] * self.n_features_
        if self.strategy == "mean":
            return [float(m) if n > 0 else self.fill_value for m, n in zip(self.mean_, self.n_)]
        return [max(sorted(c), key=c.__getitem__) if c else None for c in self._counts]

    def partial_fit(self, X):
        """
        Update the running mean or the value counts with one chunk.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Chunk of data (float, text or mixed). May have n = 0.

        Returns
        -------
        Imputer
            The same object (`self`).

        Raises
        ------
        ValueError
            If `X` is not 2-D, has a different number of features than the
            first chunk, or (strategy "mean") is not numeric or contains infinity.

        Notes
        -----
        "mean" merges the chunk mean with the running mean. "most_frequent"
        adds the counts of each distinct non-missing value, looping over columns
        only. "constant" learns nothing. `update` is an alias.

        Time: O(n d) for "mean"; O(d n log n) for "most_frequent".
        Space: O(n d) temporary; "most_frequent" also stores one counter per distinct value.
        """
        X = _as_2d(X)
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self.n_ = np.zeros(self.n_features_)
            self.mean_ = np.zeros(self.n_features_)
            self._counts = [dict() for _ in range(self.n_features_)]
        else:
            _check_n_features(self.n_features_, X)

        if self.strategy == "mean":
            Xf = _as_float_2d(X)
            valid = ~np.isnan(Xf)
            n_b = valid.sum(axis=0).astype(float)
            total = np.where(valid, Xf, 0.0).sum(axis=0)
            mean_b = np.divide(total, n_b, out=np.zeros_like(total), where=n_b > 0)
            n = self.n_ + n_b
            ratio = np.divide(n_b, n, out=np.zeros_like(n), where=n > 0)
            self.mean_ = self.mean_ + (mean_b - self.mean_) * ratio
            self.n_ = n
        elif self.strategy == "most_frequent":
            for j in range(self.n_features_):
                col = X[:, j]
                if col.dtype.kind == "f":
                    vals = col[~np.isnan(col)]
                else:
                    s = _as_str_column(col)
                    vals = s[~_is_missing(s)]
                uniq, cnt = np.unique(vals, return_counts=True)
                for v, k in zip(uniq.tolist(), cnt.tolist()):
                    self._counts[j][v] = self._counts[j].get(v, 0) + k
        return self

    update = partial_fit

    def fit(self, X):
        """
        Forget everything, then learn from X.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training data.

        Returns
        -------
        Imputer
            The same object (`self`).

        Raises
        ------
        ValueError
            As for `partial_fit`.

        Notes
        -----
        Time: As for `partial_fit`.
        Space: As for `partial_fit`.
        """
        return self.reset().partial_fit(X)

    def transform(self, X):
        """
        Replace missing entries by the current statistics.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to fill; same layout as the fitted data.

        Returns
        -------
        np.ndarray, shape (n, d)
            A copy with missing entries filled. Float input gives a float array;
            other input gives an object array.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.
        ValueError
            If `X` is not 2-D or has the wrong number of features.

        Notes
        -----
        Columns with an unknown fill value (None) are left unchanged.

        Time: O(n d).
        Space: O(n d).
        """
        self._check_fitted()
        X = _as_2d(X)
        _check_n_features(self.n_features_, X)
        stats = self.statistics_
        if X.dtype.kind == "f":
            vec = np.array([np.nan if v is None else float(v) for v in stats])
            return np.where(np.isnan(X), vec[None, :], X)
        out = X.astype(object)
        for j, v in enumerate(stats):
            if v is not None:
                out[_is_missing(_as_str_column(X[:, j])), j] = v
        return out

    def fit_transform(self, X):
        """
        Fit on X, then transform it.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to fit and fill.

        Returns
        -------
        np.ndarray, shape (n, d)
            The filled data.

        Raises
        ------
        ValueError
            As for `fit` and `transform`.

        Notes
        -----
        Time: As for `fit` plus `transform`.
        Space: O(n d).
        """
        return self.fit(X).transform(X)

    def _check_fitted(self):
        """
        Raise an error if no data has been seen yet.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        if self.n_features_ is None:
            raise RuntimeError("Imputer is not fitted; call fit/partial_fit first")


class OneHotEncoder:
    """
    One-hot encode categorical columns, learning categories from a stream.

    A category can show up for the first time in a later chunk. There are
    two options:

    - categories=None (default): categories are learned in `partial_fit`.
      New ones are appended at the end, so existing output columns never
      move, but the output gets wider until `freeze()` is called.
    - categories=[[...], [...]] (one list per column): the set is fixed up
      front, so the output width never changes.

    Parameters
    ----------
    categories : list of lists, optional
        Fixed categories, one list per column.
    handle_unknown : {"ignore", "error"}, default "ignore"
        What to do with a category that was never learned: all zeros, or
        ValueError.

    Attributes
    ----------
    categories_ : list of np.ndarray of str
        Learned categories, one array per input column.
    n_output_features_ : int
        Total number of output columns.
    n_features_ : int or None
        Number of input columns, fixed by the first chunk.

    Notes
    -----
    A missing value ("", "nan", "none", "null", "na") is never a category and
    is always encoded as all zeros.
    """
    def __init__(self, categories=None, handle_unknown="ignore"):
        """
        Create an encoder.

        Parameters
        ----------
        categories : list of lists, optional
            Fixed categories, one list per column.
        handle_unknown : str, default "ignore"
            "ignore" or "error".

        Raises
        ------
        ValueError
            If `handle_unknown` is not "ignore" or "error".

        Notes
        -----
        Time: O(1) without `categories`, O(C) with.
        Space: O(C).
        """
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(
                f"handle_unknown must be 'ignore' or 'error', got {handle_unknown!r}"
            )
        self.handle_unknown = handle_unknown
        self._init_categories = categories
        self.reset()

    def reset(self):
        """
        Forget what was learned; fixed `categories` are restored.

        Returns
        -------
        OneHotEncoder
            The same object (`self`).

        Notes
        -----
        Time: O(1) without `categories`, O(C) with.
        Space: O(C).
        """
        if self._init_categories is None:
            self.n_features_ = None
            self.categories_ = None
            self._frozen = False
        else:
            self.categories_ = [np.array([str(c) for c in cats], dtype=str)
                                for cats in self._init_categories]
            self.n_features_ = len(self.categories_)
            self._frozen = True
        return self

    def freeze(self):
        """
        Stop learning new categories, so the output width is constant.

        Returns
        -------
        OneHotEncoder
            The same object (`self`).

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        self._check_fitted()
        self._frozen = True
        return self

    @property
    def n_output_features_(self):
        """
        Total number of output columns.

        Returns
        -------
        int
            The sum of the number of categories over all columns.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(d).
        Space: O(1).
        """
        self._check_fitted()
        return int(sum(len(c) for c in self.categories_))

    def partial_fit(self, X):
        """
        Learn any new categories in this chunk.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Chunk of categorical data. May have n = 0.

        Returns
        -------
        OneHotEncoder
            The same object (`self`).

        Raises
        ------
        ValueError
            If `X` is not 2-D or has a different number of features than
            the first chunk.

        Notes
        -----
        Does nothing once the encoder is frozen or has fixed categories. New
        categories are added in sorted order after the existing ones. `update`
        is an alias.

        Time: O(d n log n).
        Space: O(n + C).
        """
        X = _as_2d(X)
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self.categories_ = [np.array([], dtype=str) for _ in range(X.shape[1])]
        else:
            _check_n_features(self.n_features_, X)
        if self._frozen:
            return self
        for j in range(self.n_features_):
            col = _as_str_column(X[:, j])
            col = col[~_is_missing(col)]
            new = np.setdiff1d(np.unique(col), self.categories_[j])
            if new.size:
                self.categories_[j] = np.concatenate([self.categories_[j], new])
        return self

    update = partial_fit

    def fit(self, X):
        """
        Forget everything, then learn from X.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training data.

        Returns
        -------
        OneHotEncoder
            The same object (`self`).

        Raises
        ------
        ValueError
            As for `partial_fit`.

        Notes
        -----
        Time: O(d n log n).
        Space: O(n + C).
        """
        return self.reset().partial_fit(X)

    def transform(self, X):
        """
        Encode X as 0/1 columns.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Categorical data.

        Returns
        -------
        np.ndarray of float, shape (n, n_output_features_)
            One block of columns per input column, in the order of `categories_`.
            Missing and (with "ignore") unknown values give all zeros.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.
        ValueError
            If `X` is not 2-D, has the wrong number of features, or
            (with handle_unknown="error") contains an unknown category.

        Notes
        -----
        Time: O(n C).
        Space: O(n C).
        """
        self._check_fitted()
        X = _as_2d(X)
        _check_n_features(self.n_features_, X)
        blocks = []
        for j, cats in enumerate(self.categories_):
            col = _as_str_column(X[:, j])
            block = (col[:, None] == cats[None, :]).astype(float)
            if self.handle_unknown == "error":
                unknown = ~_is_missing(col) & ~block.any(axis=1)
                if unknown.any():
                    raise ValueError(
                        f"Unknown categories in column {j}: {np.unique(col[unknown])[:5].tolist()}"
                    )
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((X.shape[0], 0))

    def fit_transform(self, X):
        """
        Fit on X, then transform it.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to fit and encode.

        Returns
        -------
        np.ndarray of float, shape (n, n_output_features_)
            The encoded data.

        Raises
        ------
        ValueError
            As for `fit` and `transform`.

        Notes
        -----
        Time: O(d n log n + n C).
        Space: O(n C).
        """
        return self.fit(X).transform(X)

    def _check_fitted(self):
        """
        Raise an error if no data has been seen yet.

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.

        Notes
        -----
        Time: O(1).
        Space: O(1).
        """
        if self.n_features_ is None:
            raise RuntimeError("OneHotEncoder is not fitted; call fit/partial_fit first")


class _Chain:
    """
    Run several steps in order.

    In `partial_fit` each step learns from the output of the previous steps,
    using their statistics so far.

    Parameters
    ----------
    steps : list of transformers
        Steps with partial_fit, transform and reset methods.
    """
    def __init__(self, steps):
        """
        Store the steps to run in order.

        Parameters
        ----------
        steps : iterable of transformers
            Steps to run, first to last.

        Notes
        -----
        Time: O(s) for s steps.
        Space: O(s).
        """
        self.steps = list(steps)

    def reset(self):
        """
        Reset every step.

        Returns
        -------
        _Chain
            The same object (`self`).

        Notes
        -----
        Time: O(s).
        Space: O(1).
        """
        for st in self.steps:
            st.reset()
        return self

    def partial_fit(self, X):
        """
        Fit each step on the output of the steps before it.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Chunk of data.

        Returns
        -------
        _Chain
            The same object (`self`).

        Raises
        ------
        ValueError
            As raised by the steps.

        Notes
        -----
        `update` is an alias.

        Time: The sum of the cost of each step's `partial_fit` and `transform`.
        Space: As for the steps.
        """
        for st in self.steps:
            st.partial_fit(X)
            X = st.transform(X)
        return self

    update = partial_fit

    def transform(self, X):
        """
        Pass X through every step in order.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to transform.

        Returns
        -------
        np.ndarray
            The output of the last step.

        Raises
        ------
        ValueError
            As raised by the steps.

        Notes
        -----
        Time: The sum of the cost of each step's `transform`.
        Space: As for the steps.
        """
        for st in self.steps:
            X = st.transform(X)
        return X


def _as_step(t):
    """
    Turn None, one transformer or a list of transformers into a single step.

    Parameters
    ----------
    t : None, transformer, list or tuple
        The step(s) given by the user.

    Returns
    -------
    None or transformer
        None for None or an empty list, a `_Chain` for a non-empty list, otherwise `t` itself.

    Notes
    -----
    Time: O(s) for s steps.
    Space: O(s).
    """
    if isinstance(t, (list, tuple)):
        return _Chain(t) if t else None
    return t


class ColumnTransformer:
    """
    Send numeric and categorical columns through their own steps and join them.

    Numeric columns are converted to float. The output is a float array
    [numeric block | categorical block]. Columns in neither list are dropped.
    The width can grow while the encoder is still learning categories.

    Parameters
    ----------
    numeric : transformer, list of transformers, or None
        Steps for the numeric columns, for example
        [Imputer("mean"), StandardScaler()].
    categorical : transformer, list of transformers, or None
        Steps for the categorical columns, for example
        [Imputer("most_frequent"), OneHotEncoder()].
    numeric_cols : list of int
        Indices of the numeric columns of X.
    categorical_cols : list of int
        Indices of the categorical columns of X.

    Attributes
    ----------
    n_features_ : int or None
        Number of input columns, fixed by the first chunk.
    """
    def __init__(self, numeric=None, categorical=None, numeric_cols=(), categorical_cols=()):
        """
        Set up the numeric and categorical branches.

        Parameters
        ----------
        numeric : transformer, list or None
            Steps for the numeric columns.
        categorical : transformer, list or None
            Steps for the categorical columns.
        numeric_cols : sequence of int, default ()
            Numeric column indices.
        categorical_cols : sequence of int, default ()
            Categorical column indices.

        Raises
        ------
        ValueError
            If a column is in both lists, or columns are given for a branch
            whose transformer is None.

        Notes
        -----
        Time: O(d).
        Space: O(d).
        """
        self.numeric = _as_step(numeric)
        self.categorical = _as_step(categorical)
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        if set(self.numeric_cols) & set(self.categorical_cols):
            raise ValueError("A column cannot be both numeric and categorical")
        if (self.numeric is None and self.numeric_cols) or (
            self.categorical is None and self.categorical_cols
        ):
            raise ValueError("Columns were given for a transformer that is None")
        self.n_features_ = None

    def _split(self, X):
        """
        Validate X and split it into its numeric and categorical parts.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Input data (typically an object array).

        Returns
        -------
        (np.ndarray, np.ndarray or None, np.ndarray or None)
            X as an array, the numeric columns as floats, and the categorical
            columns. A part is None if it has no columns.

        Raises
        ------
        ValueError
            If `X` is not 2-D, has the wrong number of features, a column
            index is out of range, or the numeric columns are not numeric.

        Notes
        -----
        Time: O(n d).
        Space: O(n d).
        """
        X = _as_2d(X)
        if self.n_features_ is not None:
            _check_n_features(self.n_features_, X)
        for c in self.numeric_cols + self.categorical_cols:
            if not 0 <= c < X.shape[1]:
                raise ValueError(f"Column index {c} is out of range for {X.shape[1]} columns")
        num = _as_float_2d(X[:, self.numeric_cols], "numeric columns") if self.numeric_cols else None
        cat = X[:, self.categorical_cols] if self.categorical_cols else None
        return X, num, cat

    def reset(self):
        """
        Forget everything learned by both branches.

        Returns
        -------
        ColumnTransformer
            The same object (`self`).

        Notes
        -----
        Time: As for the steps.
        Space: O(1).
        """
        for t in (self.numeric, self.categorical):
            if t is not None:
                t.reset()
        self.n_features_ = None
        return self

    def partial_fit(self, X):
        """
        Route one chunk's columns to their steps and learn from it.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Chunk of data. May have n = 0.

        Returns
        -------
        ColumnTransformer
            The same object (`self`).

        Raises
        ------
        ValueError
            As for `_split` and the steps.

        Notes
        -----
        `update` is an alias.

        Time: O(n d) plus the cost of the steps.
        Space: O(n d) plus the steps.
        """
        X, num, cat = self._split(X)
        self.n_features_ = X.shape[1]
        if num is not None:
            self.numeric.partial_fit(num)
        if cat is not None:
            self.categorical.partial_fit(cat)
        return self

    update = partial_fit

    def fit(self, X):
        """
        Forget everything, then learn from X.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training data.

        Returns
        -------
        ColumnTransformer
            The same object (`self`).

        Raises
        ------
        ValueError
            As for `partial_fit`.

        Notes
        -----
        Time: As for `partial_fit`.
        Space: As for `partial_fit`.
        """
        return self.reset().partial_fit(X)

    def transform(self, X):
        """
        Transform and join the numeric and categorical blocks.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to transform.

        Returns
        -------
        np.ndarray of float, shape (n, n_out)
            [numeric block | categorical block].

        Raises
        ------
        RuntimeError
            If the object has not been fitted yet.
        ValueError
            As for `_split` and the steps.

        Notes
        -----
        Time: O(n d) plus the cost of the steps.
        Space: O(n d) plus the steps.
        """
        if self.n_features_ is None:
            raise RuntimeError("ColumnTransformer is not fitted; call fit/partial_fit first")
        X, num, cat = self._split(X)
        blocks = []
        if num is not None:
            blocks.append(self.numeric.transform(num))
        if cat is not None:
            blocks.append(self.categorical.transform(cat))
        return np.hstack(blocks) if blocks else np.zeros((X.shape[0], 0))

    def fit_transform(self, X):
        """
        Fit on X, then transform it.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Data to fit and transform.

        Returns
        -------
        np.ndarray of float, shape (n, n_out)
            The transformed data.

        Raises
        ------
        ValueError
            As for `fit` and `transform`.

        Notes
        -----
        Time: As for `fit` plus `transform`.
        Space: As for the steps.
        """
        return self.fit(X).transform(X)