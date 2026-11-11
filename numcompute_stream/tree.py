"""
tree.py
=======
A single decision tree classifier, written from scratch with NumPy, that also
learns from a stream by **buffered retraining**.

- ``fit(X, y)`` trains the tree from scratch on the given data.
- ``partial_fit(X_chunk, y_chunk)`` appends the chunk to an internal buffer.
  Once the buffer holds at least ``buffer_size`` samples the tree is retrained
  on the buffered data and the buffer is cleared.
- Between retrains ``predict`` uses the most recently trained tree.

How the tree is built (see classification-tree.html from Week 10)
-----------------------------------------------------------------
1. try every question ``X[:, f] <= t`` (every feature, every threshold),
2. keep the one with the biggest information gain
   ``I(parent) - (n_left * I(left) + n_right * I(right)) / n``,
3. split the rows and repeat on each side,
4. stop when a rule says so (max_depth, min_samples_split, a pure node, or no
   question lowers the impurity).

Step 1 is the expensive one and it is fully vectorised: for one node, all
features and all thresholds are scored with one sort and one cumulative sum
(see ``_best_split``). The only Python loops are over tree nodes while
building and over tree levels while predicting, never over samples.

Conventions
-----------
- X has shape (n_samples, n_features) and is numeric. y has shape
  (n_samples,) and may hold any label type (ints, strings, ...).
- NaN in X means "missing". A missing value always goes to the right child,
  both in training and in prediction, and "missing vs not missing" is itself
  a question the tree may ask. Infinity in X is rejected with ValueError.
- Ties: equal gains are resolved towards the lowest feature index, then the
  lowest threshold. A leaf with tied class counts predicts the class that
  comes first in ``classes_`` (the same rule as ``np.bincount(y).argmax()``).
- Complexity uses n (samples at a node or in a call), d (features), K
  (classes) and D (tree depth).

"""

import time

import numpy as np

from .pipeline import Estimator
from .utils import NotFittedError, as_float_array

__all__ = ["DecisionTreeClassifier", "DecisionTree", "gini", "entropy"]

_CRITERIA = ("gini", "entropy")
_LEAF = -1
# gains within this distance of the best one count as ties (rounding noise)
_GAIN_TOL = 1e-12


# --------------------------------------------------------------- impurity
def _proportions(counts):
    """Turn class counts into class proportions along the last axis.

    Parameters
    ----------
    counts : np.ndarray, shape (..., K)
        Non-negative class counts.

    Returns
    -------
    np.ndarray of float, shape (..., K)
        counts / counts.sum(-1); all zeros where the total is zero.

    Notes
    -----
    Time: O(size). Space: O(size).
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)


def gini(counts):
    """Gini impurity ``1 - sum(p**2)`` from class counts.

    Parameters
    ----------
    counts : array_like, shape (..., K)
        Class counts; the last axis runs over the classes.

    Returns
    -------
    np.ndarray of float, shape (...)
        0 for a pure (or empty) group, at most ``1 - 1/K``.

    Notes
    -----
    Time: O(size). Space: O(size).

    Examples
    --------
    >>> float(gini([5, 5]))
    0.5
    """
    p = _proportions(counts)
    empty = np.asarray(counts, dtype=float).sum(axis=-1) == 0
    return np.where(empty, 0.0, 1.0 - np.einsum("...k,...k->...", p, p))


def entropy(counts):
    """Shannon entropy ``-sum(p * log2(p))`` in bits, from class counts.

    Parameters
    ----------
    counts : array_like, shape (..., K)
        Class counts; the last axis runs over the classes.

    Returns
    -------
    np.ndarray of float, shape (...)
        0 for a pure (or empty) group, at most ``log2(K)``.

    Notes
    -----
    Classes with p = 0 contribute 0 (the limit of p log p), so log2(0) is
    never evaluated. Time: O(size). Space: O(size).

    Examples
    --------
    >>> float(entropy([5, 5]))
    1.0
    """
    p = _proportions(counts)
    safe = np.where(p > 0, p, 1.0)  # log2(1) = 0, so empty classes add nothing
    return -(p * np.log2(safe)).sum(axis=-1)


_IMPURITY = {"gini": gini, "entropy": entropy}


# ------------------------------------------------------------- validation
def _check_int(value, name, minimum, allow_none=False):
    """Validate an integer hyper-parameter.

    Parameters
    ----------
    value : object
        Value to check.
    name : str
        Parameter name for error messages.
    minimum : int
        Smallest allowed value.
    allow_none : bool, default False
        Whether None is accepted (and returned unchanged).

    Returns
    -------
    int or None

    Raises
    ------
    TypeError
        If `value` is not an integer (bool is rejected too).
    ValueError
        If `value` < `minimum`.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_X(X, name="X"):
    """Validate a feature matrix.

    Parameters
    ----------
    X : array_like, shape (n, d)
        Numeric features; NaN allowed.

    Returns
    -------
    np.ndarray of float64, shape (n, d)

    Raises
    ------
    ValueError
        If `X` is not numeric, not 2-D, or contains infinity.

    Notes
    -----
    Time: O(n d). Space: O(1) if already float64, else O(n d).
    """
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (n_samples, n_features), got shape "
            f"{arr.shape}; reshape a single feature with X.reshape(-1, 1)"
        )
    return as_float_array(arr, name, allow_inf=False).astype(np.float64, copy=False)


def _check_y(y, n_samples):
    """Validate a label vector against the number of rows of X.

    Parameters
    ----------
    y : array_like, shape (n,) or (n, 1)
        Class labels of any type.
    n_samples : int
        Number of rows of X.

    Returns
    -------
    np.ndarray, shape (n,)

    Raises
    ------
    TypeError
        If `y` is None.
    ValueError
        If `y` is not 1-D, has a different length than X, or contains NaN.

    Notes
    -----
    Time: O(n). Space: O(1) for an ndarray.
    """
    if y is None:
        raise TypeError("y is required: a classifier needs the labels of every chunk")
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D with shape (n_samples,), got shape {y.shape}")
    if y.shape[0] != n_samples:
        raise ValueError(
            f"X and y must have the same number of samples, got {n_samples} and {y.shape[0]}"
        )
    if y.dtype.kind == "f" and np.isnan(y).any():
        raise ValueError("y contains NaN; drop the unlabelled rows first")
    return y


# ------------------------------------------------------------ split search
def _best_split(X, y_idx, n_classes, criterion="gini", features=None):
    """Find the question ``X[:, f] <= t`` with the largest information gain.

    Every feature and every threshold is scored at once:

    1. sort each feature column (one ``argsort`` along axis 0),
    2. one-hot the labels in that order and take a cumulative sum, which
       gives the class counts on the left of every possible cut,
    3. the right counts are ``total - left``, the impurities and the weighted
       gain follow by broadcasting,
    4. a cut is only valid between two different values, and the best valid
       cut is taken with one ``argmax``.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, d)
        Features of the samples at this node; NaN allowed.
    y_idx : np.ndarray of int, shape (n,)
        Class index of each sample, in ``range(n_classes)``.
    n_classes : int
        K, the number of classes.
    criterion : {"gini", "entropy"}, default "gini"
        Impurity measure.
    features : np.ndarray of int, shape (f,), optional
        Feature indices to search (in increasing order). Default: all.

    Returns
    -------
    feature : int
        Index of the chosen feature, or -1 if no valid cut lowers the impurity.
    threshold : float
        Samples with ``X[:, feature] <= threshold`` go left (NaN goes right).
        NaN when `feature` is -1.
    gain : float
        Information gain of the cut (0.0 when `feature` is -1).

    Notes
    -----
    NaN sorts last, so the missing values of a column sit at the end. The cut
    between the last finite value and the first NaN is allowed and uses the
    last finite value as its threshold: it asks "is this feature present?".
    The threshold between two finite values is their midpoint, computed as
    ``a + (b - a) / 2`` so it cannot overflow; if rounding lifts it onto the
    upper value, the lower value is used so the two sides stay separated.

    Ties: all gains within 1e-12 of the best count as equal, and the first
    in (feature, threshold) order wins, so the result is deterministic.

    Time: O(d n log n + d n K). Space: O(d n K).
    """
    n = X.shape[0]
    if n < 2:
        return _LEAF, np.nan, 0.0
    if features is None:
        features = np.arange(X.shape[1])
    Xf = X[:, features]                                       # (n, f)
    order = np.argsort(Xf, axis=0, kind="stable")
    xs = np.take_along_axis(Xf, order, axis=0)                # sorted values
    onehot = (y_idx[order][:, :, None] == np.arange(n_classes)).astype(float)

    left = np.cumsum(onehot, axis=0)                          # (n, f, K)
    total = left[-1]                                          # (f, K)
    left = left[:-1]                                          # cut after row i
    right = total[None, :, :] - left
    n_left = np.arange(1, n, dtype=float)[:, None]            # (n-1, 1)
    n_right = n - n_left

    impurity = _IMPURITY[criterion]
    before = impurity(total[0])
    after = (n_left * impurity(left) + n_right * impurity(right)) / n
    gain = before - after                                     # (n-1, f)

    lo, hi = xs[:-1], xs[1:]
    lo_nan, hi_nan = np.isnan(lo), np.isnan(hi)
    with np.errstate(invalid="ignore"):
        valid = (lo < hi) | (~lo_nan & hi_nan)
    gain = np.where(valid, gain, -np.inf)

    best = gain.max()
    if not best > _GAIN_TOL:
        return _LEAF, np.nan, 0.0
    # feature-major order: lowest feature index first, then lowest threshold
    flat = (gain >= best - _GAIN_TOL).T.ravel()
    k = int(np.argmax(flat))
    f, i = divmod(k, n - 1)
    a, b = xs[i, f], xs[i + 1, f]
    if np.isnan(b):
        threshold = a
    else:
        threshold = a + (b - a) / 2.0
        if not threshold < b:
            threshold = a
    return int(features[f]), float(threshold), float(gain[i, f])


class DecisionTreeClassifier(Estimator):
    """A depth-limited classification tree with buffered streaming updates.

    Parameters
    ----------
    max_depth : int or None, default 5
        Largest number of questions on any root-to-leaf path (>= 1).
        None grows until another stop rule applies.
    min_samples_split : int, default 2
        A node with fewer samples than this becomes a leaf (>= 2).
    max_features : int, float, {"sqrt", "log2"} or None, default None
        How many features each node may look at. None: all of them. int: that
        many. float in (0, 1]: that fraction (at least 1). "sqrt" / "log2":
        the square root / base-2 log of the number of features. A random
        subset is drawn at every node from `random_state`.
    buffer_size : int, default 500
        ``partial_fit`` retrains the tree once this many samples are buffered
        (>= 1).
    criterion : {"gini", "entropy"}, default "gini"
        Impurity measure used for the information gain.
    random_state : int or None, default None
        Seed for the feature subsets (only used when `max_features` selects
        fewer than all features).

    Attributes
    ----------
    classes_ : np.ndarray, shape (K,)
        Sorted class labels. In streaming mode this is every class seen so
        far, so the columns of ``predict_proba`` never change meaning.
    n_features_ : int
        Number of features, fixed by the first call.
    feature_, threshold_ : np.ndarray, shape (n_nodes,)
        Question at each node: ``X[:, feature_] <= threshold_``. A leaf has
        feature -1 and threshold NaN.
    left_, right_ : np.ndarray of int, shape (n_nodes,)
        Child node ids (-1 at a leaf).
    value_ : np.ndarray of float, shape (n_nodes, K)
        Class counts of the training samples that reached each node.
    impurity_, n_node_samples_, depth_ : np.ndarray, shape (n_nodes,)
        Impurity, sample count and depth of each node.
    n_retrains_ : int
        Number of times ``partial_fit`` (or ``flush``) rebuilt the tree.
    n_samples_seen_ : int
        Samples received through ``partial_fit`` so far.
    buffer_count_ : int
        Samples currently waiting in the buffer.
    last_fit_seconds_ : float
        Wall-clock time of the most recent tree build.
    total_fit_seconds_ : float
        Wall-clock time of all tree builds since the last reset.

    Notes
    -----
    Buffered retraining trades freshness for cost. A small `buffer_size`
    retrains often on little data (fast builds, noisy trees); a large one
    retrains rarely on more data (slower builds, steadier trees, a longer
    wait before the model reacts to a change). Each retrain uses only the
    buffered samples, so the tree always describes the most recent buffer.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> tree = DecisionTreeClassifier(max_depth=2, buffer_size=4)
    >>> tree.partial_fit(X[:2], y[:2]).is_fitted      # buffer not full yet
    False
    >>> tree.partial_fit(X[2:], y[2:]).predict([[0.5], [2.5]]).tolist()
    [0, 1]
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion="gini", random_state=None):
        """Create an untrained tree; see the class docstring for the parameters.

        Raises
        ------
        TypeError
            If an integer parameter is not an integer, or `max_features` has
            an unsupported type.
        ValueError
            If a parameter is out of range or `criterion` is unknown.

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        self.max_depth = _check_int(max_depth, "max_depth", 1, allow_none=True)
        self.min_samples_split = _check_int(min_samples_split, "min_samples_split", 2)
        self.buffer_size = _check_int(buffer_size, "buffer_size", 1)
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
        self.criterion = criterion
        self.max_features = self._check_max_features(max_features)
        self.random_state = random_state
        self.reset()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _check_max_features(max_features):
        """Validate `max_features` (see the class docstring).

        Returns
        -------
        The value unchanged.

        Raises
        ------
        TypeError, ValueError
            Unsupported type or out-of-range value.

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        if max_features is None or max_features in ("sqrt", "log2"):
            return max_features
        if isinstance(max_features, (bool, np.bool_)):
            raise TypeError("max_features must not be a bool")
        if isinstance(max_features, (int, np.integer)):
            if max_features < 1:
                raise ValueError(f"max_features must be >= 1, got {max_features}")
            return int(max_features)
        if isinstance(max_features, (float, np.floating)):
            if not 0.0 < max_features <= 1.0:
                raise ValueError(
                    f"a float max_features must be in (0, 1], got {max_features}"
                )
            return float(max_features)
        raise TypeError(
            "max_features must be None, an int, a float in (0, 1], 'sqrt' or "
            f"'log2', got {max_features!r}"
        )

    def _n_split_features(self, d):
        """Number of features each node searches, for d input features.

        Returns
        -------
        int in [1, d]

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d))) if d > 1 else 1
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(mf, d)

    @property
    def is_fitted(self):
        """True once a tree has been built (by ``fit`` or by a retrain)."""
        return self.feature_ is not None

    def _check_fitted(self):
        """Raise NotFittedError if no tree has been built yet.

        Raises
        ------
        NotFittedError
            Before ``fit``, or while the first buffer is still filling.
        """
        if not self.is_fitted:
            raise NotFittedError(
                "DecisionTreeClassifier has no trained tree yet: call fit(X, y), "
                f"or keep calling partial_fit until {self.buffer_size} samples "
                f"are buffered ({self.buffer_count_} so far), or call flush()"
            )

    def reset(self):
        """Forget the tree, the buffer and every counter.

        Returns
        -------
        DecisionTreeClassifier
            The same object (`self`).

        Notes
        -----
        Time: O(1). Space: O(1).
        """
        self.classes_ = None
        self.n_features_ = None
        self.feature_ = self.threshold_ = None
        self.left_ = self.right_ = None
        self.value_ = self.impurity_ = None
        self.n_node_samples_ = self.depth_ = None
        self._buffer_X, self._buffer_y = [], []
        self.buffer_count_ = 0
        self.n_samples_seen_ = 0
        self.n_retrains_ = 0
        self.last_fit_seconds_ = 0.0
        self.total_fit_seconds_ = 0.0
        self._rng = np.random.default_rng(self.random_state)
        return self

    # ----------------------------------------------------------- building
    def _build(self, X, y_idx):
        """Grow the tree on (X, y_idx) and store it as flat arrays.

        Parameters
        ----------
        X : np.ndarray of float, shape (n, d)
        y_idx : np.ndarray of int, shape (n,)
            Class index of every sample in ``range(len(classes_))``.

        Notes
        -----
        Nodes are processed from an explicit stack (left child first), so a
        deep tree cannot hit Python's recursion limit. The loop runs once per
        node; all the work inside a node is vectorised over its samples.
        Storing the tree as arrays lets ``predict`` move every sample one
        level down with a single indexing step.

        Time: O(D d n log n) for a balanced tree. Space: O(d n K) for the
        largest node plus O(n_nodes K) for the stored tree.
        """
        start = time.perf_counter()
        n, d = X.shape
        K = self.classes_.size
        m = self._n_split_features(d)
        impurity = _IMPURITY[self.criterion]

        feature, threshold, left, right = [], [], [], []
        value, node_impurity, n_samples, depths = [], [], [], []

        def new_node(counts, depth):
            """Append a leaf with these class counts and return its node id."""
            feature.append(_LEAF)
            threshold.append(np.nan)
            left.append(_LEAF)
            right.append(_LEAF)
            value.append(counts)
            node_impurity.append(float(impurity(counts)))
            n_samples.append(int(counts.sum()))
            depths.append(depth)
            return len(feature) - 1

        root_counts = np.bincount(y_idx, minlength=K).astype(float)
        stack = [(new_node(root_counts, 0), np.arange(n), 0)]
        while stack:                       # one iteration per node
            node, rows, depth = stack.pop()
            counts = value[node]
            if ((self.max_depth is not None and depth >= self.max_depth)
                    or rows.size < self.min_samples_split
                    or np.count_nonzero(counts) <= 1):
                continue                   # stays a leaf
            Xn, yn = X[rows], y_idx[rows]
            subset = None
            if m < d:
                subset = np.sort(self._rng.choice(d, size=m, replace=False))
            f, t, _ = _best_split(Xn, yn, K, self.criterion, subset)
            if f == _LEAF and subset is not None:
                # the random subset had nothing useful; look at every feature
                f, t, _ = _best_split(Xn, yn, K, self.criterion, None)
            if f == _LEAF:
                continue                   # no question helps
            go_left = Xn[:, f] <= t        # NaN compares False -> right
            l_rows, r_rows = rows[go_left], rows[~go_left]
            l_id = new_node(np.bincount(y_idx[l_rows], minlength=K).astype(float), depth + 1)
            r_id = new_node(np.bincount(y_idx[r_rows], minlength=K).astype(float), depth + 1)
            feature[node], threshold[node] = f, t
            left[node], right[node] = l_id, r_id
            stack.append((r_id, r_rows, depth + 1))
            stack.append((l_id, l_rows, depth + 1))

        self.feature_ = np.array(feature, dtype=np.intp)
        self.threshold_ = np.array(threshold, dtype=float)
        self.left_ = np.array(left, dtype=np.intp)
        self.right_ = np.array(right, dtype=np.intp)
        self.value_ = np.vstack(value)
        self.impurity_ = np.array(node_impurity)
        self.n_node_samples_ = np.array(n_samples, dtype=np.intp)
        self.depth_ = np.array(depths, dtype=np.intp)
        self.last_fit_seconds_ = time.perf_counter() - start
        self.total_fit_seconds_ += self.last_fit_seconds_

    def fit(self, X, y):
        """Train the tree from scratch on (X, y).

        Any earlier tree, the streaming buffer and the counters are discarded.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Numeric features; NaN means missing.
        y : array_like, shape (n_samples,)
            Class labels of any type.

        Returns
        -------
        DecisionTreeClassifier
            The same object (`self`).

        Raises
        ------
        TypeError
            If `y` is None.
        ValueError
            If X is not 2-D or not numeric, contains infinity, has no rows,
            or y has the wrong shape or contains NaN.

        Notes
        -----
        Time: O(D d n log n) for a balanced tree. Space: O(d n K).
        """
        X = _check_X(X)
        y = _check_y(y, X.shape[0])
        if X.shape[0] == 0:
            raise ValueError("cannot fit a tree on 0 samples")
        self.reset()
        self.n_features_ = X.shape[1]
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._build(X, y_idx.reshape(-1))
        return self

    # ---------------------------------------------------------- streaming
    def partial_fit(self, X, y=None, classes=None):
        """Add one chunk to the buffer; retrain when the buffer is full.

        Parameters
        ----------
        X : array_like, shape (n_chunk, n_features)
            Chunk of features. May have 0 rows (nothing happens).
        y : array_like, shape (n_chunk,)
            Labels of the chunk. Required.
        classes : array_like, shape (K,), optional
            Every class the stream can produce. Passing it on the first call
            fixes the column order of ``predict_proba`` from the start;
            otherwise classes are added as they appear.

        Returns
        -------
        DecisionTreeClassifier
            The same object (`self`).

        Raises
        ------
        TypeError
            If `y` is None.
        ValueError
            Bad shapes, non-numeric X, infinity in X, NaN in y, or a different
            number of features than the first chunk. A rejected chunk is not
            added to the buffer.

        Notes
        -----
        The chunk is appended to the buffer. When the buffer holds at least
        `buffer_size` samples the tree is rebuilt on all buffered samples and
        the buffer is cleared, so a chunk larger than `buffer_size` triggers a
        retrain on its own. Until the first retrain there is no tree and
        ``predict`` raises NotFittedError; afterwards ``predict`` keeps using
        the last tree while the next buffer fills. `update` is an alias.

        Time: O(n_chunk d) to buffer, plus one tree build on the buffer when
        it fills. Space: O(buffer_size d) for the buffer.
        """
        X = _check_X(X)
        y = _check_y(y, X.shape[0])
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
        elif X.shape[1] != self.n_features_:
            raise ValueError(
                f"expected {self.n_features_} features (set by the first chunk), "
                f"got {X.shape[1]}"
            )
        if classes is not None:
            self._add_classes(np.asarray(classes).ravel())
        if X.shape[0] == 0:
            return self
        # copies, so later changes to the caller's arrays cannot alter the buffer
        self._buffer_X.append(X.copy())
        self._buffer_y.append(y.copy())
        self.buffer_count_ += X.shape[0]
        self.n_samples_seen_ += X.shape[0]
        if self.buffer_count_ >= self.buffer_size:
            self._retrain()
        return self

    update = partial_fit

    def _add_classes(self, labels):
        """Merge new labels into ``classes_`` and widen ``value_`` to match.

        Parameters
        ----------
        labels : np.ndarray, shape (m,)
            Labels that may or may not be known already.

        Notes
        -----
        Existing columns of ``value_`` are moved to the positions their
        classes have in the new sorted ``classes_``; new classes get zero
        counts, so the current tree's predictions do not change.
        Time: O((K + m) log(K + m) + n_nodes K). Space: O(n_nodes K).
        """
        if self.classes_ is None:
            self.classes_ = np.unique(labels)
            return
        merged = np.unique(np.concatenate([self.classes_, labels]))
        if merged.size == self.classes_.size:
            return
        if self.value_ is not None:
            widened = np.zeros((self.value_.shape[0], merged.size))
            widened[:, np.searchsorted(merged, self.classes_)] = self.value_
            self.value_ = widened
        self.classes_ = merged

    def _retrain(self):
        """Rebuild the tree on the buffered samples and clear the buffer.

        Notes
        -----
        Time: one tree build on the buffer. Space: O(buffer d K).
        """
        X = np.concatenate(self._buffer_X, axis=0)
        y = np.concatenate(self._buffer_y, axis=0)
        self._add_classes(y)
        y_idx = np.searchsorted(self.classes_, y)
        self._build(X, y_idx)
        self._buffer_X, self._buffer_y = [], []
        self.buffer_count_ = 0
        self.n_retrains_ += 1

    def flush(self):
        """Retrain now on whatever is buffered, even if the buffer is not full.

        Useful at the end of a stream, so the last partial buffer is not lost.

        Returns
        -------
        DecisionTreeClassifier
            The same object (`self`). Nothing happens if the buffer is empty.

        Notes
        -----
        Time: one tree build on the buffer. Space: O(buffer d K).
        """
        if self.buffer_count_ > 0:
            self._retrain()
        return self

    # ---------------------------------------------------------- predicting
    def apply(self, X):
        """Leaf reached by every sample.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of int, shape (n_samples,)
            Node id of the leaf each sample ends in.

        Raises
        ------
        NotFittedError
            If no tree has been built yet.
        ValueError
            If X is not 2-D, not numeric, contains infinity or has the wrong
            number of features.

        Notes
        -----
        All samples walk down together: each pass of the loop moves every
        sample that is not yet in a leaf one level down, so the loop runs at
        most D times (tree depth), never once per sample. A NaN feature
        compares False with ``<=`` and therefore goes right, as in training.

        Time: O(n D). Space: O(n).
        """
        self._check_fitted()
        X = _check_X(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        node = np.zeros(X.shape[0], dtype=np.intp)
        for _ in range(int(self.depth_.max())):      # one pass per tree level
            feat = self.feature_[node]
            rows = np.flatnonzero(feat != _LEAF)
            if rows.size == 0:
                break
            cur = node[rows]
            go_left = X[rows, feat[rows]] <= self.threshold_[cur]
            node[rows] = np.where(go_left, self.left_[cur], self.right_[cur])
        return node

    def predict_proba(self, X):
        """Class proportions of the leaf each sample falls in.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of float, shape (n_samples, K)
            Row i sums to 1; column k belongs to ``classes_[k]``.

        Raises
        ------
        NotFittedError, ValueError
            As for ``apply``.

        Notes
        -----
        Time: O(n D + n K). Space: O(n K).
        """
        return _proportions(self.value_[self.apply(X)])

    def predict(self, X):
        """Predict the majority class of the leaf each sample falls in.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Labels with the dtype of ``classes_``.

        Raises
        ------
        NotFittedError, ValueError
            As for ``apply``.

        Notes
        -----
        In streaming mode this always uses the most recently trained tree;
        samples waiting in the buffer have no influence until the next
        retrain. Tied counts go to the class that comes first in ``classes_``.

        Time: O(n D + n K). Space: O(n).
        """
        leaves = self.apply(X)
        return self.classes_[np.argmax(self.value_[leaves], axis=1)]

    def score(self, X, y):
        """Accuracy of ``predict(X)`` against y.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,)

        Returns
        -------
        float in [0, 1]

        Raises
        ------
        NotFittedError, ValueError
            As for ``apply``; also if there are no samples.

        Notes
        -----
        Time: O(n D). Space: O(n).
        """
        pred = self.predict(X)
        y = _check_y(y, pred.shape[0])
        if y.size == 0:
            raise ValueError("cannot score 0 samples")
        return float(np.mean(pred == y))

    # ---------------------------------------------------------- inspection
    def get_depth(self):
        """Depth of the trained tree (0 for a single leaf).

        Raises
        ------
        NotFittedError
            If no tree has been built yet.
        """
        self._check_fitted()
        return int(self.depth_.max())

    def get_n_leaves(self):
        """Number of leaves of the trained tree.

        Raises
        ------
        NotFittedError
            If no tree has been built yet.
        """
        self._check_fitted()
        return int(np.count_nonzero(self.feature_ == _LEAF))

    @property
    def n_nodes_(self):
        """Total number of nodes (questions and leaves) of the trained tree."""
        self._check_fitted()
        return int(self.feature_.size)

    def export_text(self, feature_names=None, precision=3):
        """The tree as indented text, one line per node.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names for the features; default ``x0, x1, ...``.
        precision : int, default 3
            Decimal places for thresholds.

        Returns
        -------
        str
            e.g. ``petal_length <= 2.450`` followed by indented ``yes`` / ``no``
            branches; leaves show the predicted class and the class counts.

        Raises
        ------
        NotFittedError
            If no tree has been built yet.
        ValueError
            If `feature_names` has the wrong length.

        Notes
        -----
        Presentation only, so it loops over the nodes.
        Time: O(n_nodes K). Space: O(n_nodes).
        """
        self._check_fitted()
        if feature_names is None:
            feature_names = [f"x{j}" for j in range(self.n_features_)]
        elif len(feature_names) != self.n_features_:
            raise ValueError(
                f"feature_names needs {self.n_features_} names, got {len(feature_names)}"
            )
        lines = []
        stack = [(0, "", "")]
        while stack:
            node, indent, label = stack.pop()
            counts = self.value_[node].astype(int).tolist()
            if self.feature_[node] == _LEAF:
                cls = self.classes_[int(np.argmax(self.value_[node]))]
                lines.append(f"{indent}{label}predict {cls!s}  counts={counts}")
                continue
            name = feature_names[self.feature_[node]]
            lines.append(
                f"{indent}{label}{name} <= {self.threshold_[node]:.{precision}f} ?"
                f"  (n={self.n_node_samples_[node]}, {self.criterion}="
                f"{self.impurity_[node]:.{precision}f})"
            )
            child = indent + "    "
            stack.append((int(self.right_[node]), child, "no:  "))
            stack.append((int(self.left_[node]), child, "yes: "))
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"DecisionTreeClassifier(max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_features={self.max_features!r}, buffer_size={self.buffer_size}, "
            f"criterion={self.criterion!r}, random_state={self.random_state!r})"
        )


# Short name used in the assignment brief: DecisionTree(max_depth=5, buffer_size=500)
DecisionTree = DecisionTreeClassifier
