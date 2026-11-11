"""
metrics.py
Metrics for scoring a model's predictions.
Classification: accuracy, precision, recall, f1, confusion_matrix.
Regression: mse. Binary classification: roc_curve, auc, roc_auc_score.
All functions are vectorised and work with any label type, including strings.

Streaming metrics
-----------------
Every batch metric has a streaming class with the same three methods:

- update(y_true_chunk, y_pred_chunk) -> self   add one chunk
- result()                                     current value
- reset() -> self                              forget everything

StreamingConfusionMatrix, StreamingAccuracy, StreamingPrecision,
StreamingRecall, StreamingF1, StreamingMSE and StreamingAUC accumulate over
the whole stream by default. Passing ``window=N`` turns any of them into a
rolling-window metric over the N most recent samples. A chunk may be empty
(nothing changes) and ``result()`` is NaN until there is something to score.
"""

import numpy as np

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
    "roc_curve",
    "auc",
    "roc_auc_score",
    "StreamingConfusionMatrix",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingMSE",
    "StreamingAUC",
]

VALID_AVERAGES = ("binary", "macro", "micro", "weighted", None)


def _as_1d(a, name):
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D (or shape (n, 1)), got shape {a.shape}")
    return a


def _check_pair(y_true, y_pred, names=("y_true", "y_pred")):
    a = _as_1d(y_true, names[0])
    b = _as_1d(y_pred, names[1])
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same length, "
            f"got {a.shape[0]} and {b.shape[0]}"
        )
    if a.shape[0] == 0:
        raise ValueError(f"{names[0]} and {names[1]} must not be empty")
    for arr, name in ((a, names[0]), (b, names[1])):
        if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN")
    return a, b


def _safe_divide(num, den, zero_division):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, float(zero_division))
    np.divide(num, den, out=out, where=den != 0)
    return out


def _label_index(y, labels):
    k = labels.size
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    pos = np.clip(np.searchsorted(sorted_labels, y), 0, k - 1)
    found = sorted_labels[pos] == y
    return order[pos], found


def _per_class_counts(y_true, y_pred, labels):
    k = labels.size
    t_idx, t_ok = _label_index(y_true, labels)
    p_idx, p_ok = _label_index(y_pred, labels)
    # a true positive is where both labels are known classes and they match
    hit = t_ok & p_ok & (t_idx == p_idx)
    tp = np.bincount(t_idx[hit], minlength=k)
    pred_pos = np.bincount(p_idx[p_ok], minlength=k)
    true_pos = np.bincount(t_idx[t_ok], minlength=k)
    return tp, pred_pos, true_pos


def _prf(y_true, y_pred, average, pos_label, labels, zero_division):
    if average not in VALID_AVERAGES:
        raise ValueError(f"average must be one of {VALID_AVERAGES}, got {average!r}")
    y_true, y_pred = _check_pair(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                f"average='binary' needs at most 2 classes, got {labels.size} "
                f"{labels.tolist()}. Use average='macro', 'micro' or 'weighted'"
            )
        if not np.any(labels == pos_label):
            raise ValueError(
                f"pos_label={pos_label!r} is not one of the labels "
                f"{labels.tolist()}"
            )
        labels = np.array([pos_label], dtype=labels.dtype)

    tp, pred_pos, true_pos = _per_class_counts(y_true, y_pred, labels)
    return _average_prf(tp, pred_pos, true_pos, average, zero_division)


def _average_prf(tp, pred_pos, true_pos, average, zero_division):
    # shared by the batch functions and the streaming classes: turns per-class
    # counts (true positives, predicted positives, actual positives) into
    # precision, recall and F1 under the requested averaging
    if average == "micro":
        # micro: add up the counts over all classes first, then divide once
        tp, pred_pos, true_pos = tp.sum(), pred_pos.sum(), true_pos.sum()
    p = _safe_divide(tp, pred_pos, zero_division)
    r = _safe_divide(tp, true_pos, zero_division)
    f = _safe_divide(2 * p * r, p + r, zero_division)
    if average in ("binary", "micro"):
        return p.item(), r.item(), f.item()
    if average == "macro":
        return float(p.mean()), float(r.mean()), float(f.mean())
    if average == "weighted":
        if true_pos.sum() == 0:
            return (float(zero_division),) * 3
        # weight each class by how many samples it has
        w = true_pos / true_pos.sum()
        return float(p @ w), float(r @ w), float(f @ w)
    return p, r, f


def accuracy(y_true, y_pred):
    """Fraction of predictions that match the true labels.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n,)

    Returns
    -------
    float between 0 and 1

    Raises
    ------
    ValueError
        Empty, not 1D, different lengths, or NaN in the input.

    Notes
    -----
    O(n) time and space.

    Examples
    --------
    >>> accuracy([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Count how often each true class got predicted as each class.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n,)
    labels: array-like of shape (k,), optional
        Class order for the rows and columns. Defaults to every class seen, sorted. Samples with a label outside this list are left out.

    Returns
    -------
    numpy.ndarray of int, shape (k, k)
        cm[i, j] = number of samples that are labels[i] but were predicted as labels[j]. Rows are the truth, columns the predictions.

    Raises
    ------
    ValueError
        Bad input (see accuracy), or labels empty or with duplicates.

    Notes
    -----
    Each (true, predicted) pair becomes one cell number, k * true + pred,and a single bincount counts all of them.
    Reshaping gives the matrix. O(n log k + k^2) time, O(n + k^2) space.

    Examples
    --------
    >>> confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2])
    array([[1, 0, 0],
           [0, 1, 1],
           [0, 0, 1]])
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("labels must be a non-empty 1D array")
        if np.unique(labels).size != labels.size:
            raise ValueError("labels must not contain duplicates")
    k = labels.size
    t_idx, t_ok = _label_index(y_true, labels)
    p_idx, p_ok = _label_index(y_pred, labels)
    keep = t_ok & p_ok
    # turn each (true, predicted) pair into a single cell number, then one bincount counts every cell at once
    cells = k * t_idx[keep] + p_idx[keep]
    return np.bincount(cells, minlength=k * k).reshape(k, k)


def precision(
    y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0
):
    """Precision = TP/(TP + FP): of the things predicted as a class, how many really were that class.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n,)
    average: {"binary", "macro", "micro", "weighted", None}, default "binary"
        binary: only score pos_label.
        macro: plain average over the classes.
        micro: add up TP and FP over all classes, then divide.
        weighted: average weighted by how many samples each class has.
        None: return one score per class.
    pos_label:default 1
        The positive class, only used when average="binary".
    labels: array-like, optional
        Which classes to include. Defaults to all of them.
    zero_division: float, default 0.0
        Returned when nothing was predicted as the class.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) if average is None

    Raises
    ------
    ValueError
        Bad input, unknown average, or pos_label not one of the classes.
        Also raised for average="binary" when there are more than 2 classes, since the other classes would be silently ignored.
    Notes
    -----
    O(n log n) time, O(n + k) space for k classes.

    Examples
    --------
    >>> precision([0, 1, 1, 0], [1, 1, 1, 0])
    0.6666666666666666
    """
    return _prf(y_true, y_pred, average, pos_label, labels, zero_division)[0]


def recall(
    y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0
):
    """Recall = TP / (TP + FN): of the things that really are a class, how many we found.

    Parameters
    ----------
    Same as precision. zero_division is returned when the class never appears in y_true.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) if average is None

    Raises
    ------
    ValueError
        Same as precision.

    Notes
    -----
    O(n log n) time, O(n + k) space.

    Examples
    --------
    >>> recall([0, 1, 1, 0], [1, 1, 0, 0])
    0.5
    """
    return _prf(y_true, y_pred, average, pos_label, labels, zero_division)[1]


def f1(y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0):
    """F1 = 2 * precision * recall / (precision + recall).

    It's low if either precision or recall is low.
    With average="macro" this is the mean of the per-class F1 scores, not the F1 of the mean precision and recall.

    Parameters
    ----------
    Same as precision. zero_division is returned when precision + recall is 0.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) if average is None

    Raises
    ------
    ValueError
        Same as precision.

    Notes
    -----
    O(n log n) time, O(n + k) space.

    Examples
    --------
    >>> f1([0, 1, 1, 0], [1, 1, 0, 0])
    0.5
    """
    return _prf(y_true, y_pred, average, pos_label, labels, zero_division)[2]


def mse(y_true, y_pred, axis=None):
    """Mean squared error, mean((y_true - y_pred) ** 2).

    Parameters
    ----------
    y_true: array-like of shape (n,) or (n, m)
    y_pred: array-like, same shape as y_true
    axis: int or None, default None
        None gives one number. axis=0 on 2D input gives one error per column, shape (m,).

    Returns
    -------
    float, or numpy.ndarray if axis is given

    Raises
    ------
    ValueError
        Shapes don't match, empty input, NaN or infinity. NaN raises instead of being skipped, since skipping missing predictions would make the model look better than it is.
        Infinity raises because the error would just be inf (or NaN for inf - inf), which says nothing useful.
    Notes
    -----
    O(n*m) time and space.

    Examples
    --------
    >>> mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    1.3333333333333333
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("y_true and y_pred must not contain NaN")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("y_true and y_pred must not contain infinity")
    err = np.mean((y_true - y_pred) ** 2, axis=axis)
    return float(err) if axis is None else err


def roc_curve(y_true, y_score, pos_label=1):
    """ROC curve for a binary classifier.

    For each threshold t, anything with score >= t counts as positive.
    As t goes down we get more true positives and more false positives, and the curve plots the two rates against each other.

    Parameters
    ----------
    y_true: array-like of shape (n,)
        True labels, two classes only.
    y_score: array-like of shape (n,)
        Score for the positive class. Higher = more likely positive.
    pos_label: default 1
        Which label in y_true is the positive one.

    Returns
    -------
    fpr: numpy.ndarray of shape (t + 1,)
        False positive rate, from 0 up to 1.
    tpr: numpy.ndarray of shape (t + 1,)
        True positive rate, from 0 up to 1.
    thresholds: numpy.ndarray of shape (t + 1,)
        Goes down. The first one is inf (nothing predicted positive). t is the number of distinct scores.

    Raises
    ------
    ValueError
        Bad input, more than two classes, infinite scores, or y_true doesn't have both positives and negatives (one of the rates would be 0/0).
        Infinite scores raise because they would repeat the inf threshold that marks the start of the curve.
    Notes
    -----
    Sort the scores once, then a cumulative sum gives the TP count at every cut point.
    Samples with the same score can't be split by any threshold, so we only cut at the last one of each group of tied scores.
    O(n log n) time, O(n) space.

    Examples
    --------
    >>> fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    """
    y_true, y_score = _check_pair(y_true, y_score, ("y_true", "y_score"))
    y_score = y_score.astype(float)
    if np.isinf(y_score).any():
        raise ValueError("y_score must not contain infinity")

    if np.unique(y_true).size > 2:
        raise ValueError("roc_curve supports binary labels only")
    is_pos = y_true == pos_label
    n_pos = int(is_pos.sum())
    n_neg = is_pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "y_true must contain both positive and negative samples "
            f"(found {n_pos} positive, {n_neg} negative)"
        )
    order = np.argsort(-y_score, kind="stable")
    scores = y_score[order]
    pos_sorted = is_pos[order]
    # index of the last sample in each run of equal scores
    last_of_run = np.r_[np.flatnonzero(np.diff(scores) != 0), scores.size - 1]
    tps = np.cumsum(pos_sorted)[last_of_run]
    fps = (last_of_run + 1) - tps
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, scores[last_of_run]]
    return fpr, tpr, thresholds


def auc(x, y):
    """Area under a curve using the trapezoid rule.

    Works for any curve, not just ROC.

    Parameters
    ----------
    x:array-like of shape (n,)
        Must be all increasing or all decreasing.
    y:array-like of shape (n,)

    Returns
    -------
    float, never negative as long as x is monotonic

    Raises
    ------
    ValueError
        Fewer than 2 points, different lengths, or x goes up and down.

    Notes
    -----
    Adds up (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2 over all segments.
    O(n) time and space.

    Examples
    --------
    >>> auc([0, 0.5, 1], [0, 1, 1])
    0.75
    """
    x, y = _check_pair(x, y, ("x", "y"))
    x = x.astype(float)
    y = y.astype(float)
    if x.size < 2:
        raise ValueError(f"auc needs at least 2 points, got {x.size}")
    dx = np.diff(x)
    if np.all(dx >= 0):
        direction = 1.0
    elif np.all(dx <= 0):
        direction = -1.0
    else:
        raise ValueError("x must be monotonic (increasing or decreasing)")
    return float(direction * np.sum(dx * (y[1:] + y[:-1]) / 2.0))


def roc_auc_score(y_true, y_score, pos_label=1):
    """Area under the ROC curve, i.e. roc_curve followed by auc.

    0.5 is no better than guessing, 1.0 means every positive scored higher than every negative.

    Parameters
    ----------
    Same as roc_curve.

    Returns
    -------
    float between 0 and 1

    Raises
    ------
    ValueError
        Same as roc_curve.

    Notes
    -----
    O(n log n) time, O(n) space.

    Examples
    --------
    >>> roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    0.75
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    return auc(fpr, tpr)


# ---------------------------------------------------------------- streaming
def _check_window(window):
    if window is None:
        return None
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)):
        raise TypeError(f"window must be a positive integer or None, got {window!r}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return int(window)


def _check_chunk(y_true, y_pred, names=("y_true", "y_pred")):
    # like _check_pair, but an empty chunk is fine in a stream
    a = _as_1d(y_true, names[0])
    b = _as_1d(y_pred, names[1])
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same length, "
            f"got {a.shape[0]} and {b.shape[0]}"
        )
    for arr, name in ((a, names[0]), (b, names[1])):
        if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN")
    return a, b


def _append_window(old, new, window):
    # keep only the `window` most recent values
    joined = new if old is None else np.concatenate([old, new])
    return joined[-window:]


class StreamingConfusionMatrix:
    """Confusion matrix that accumulates over chunks.

    Parameters
    ----------
    labels : array-like of shape (k,), optional
        Fixed class order. Default: classes are added as they appear and kept sorted, so the matrix can grow.
        With fixed labels, samples whose true or predicted label is not listed are left out, as in confusion_matrix.
    window : int, optional
        Only the `window` most recent samples are counted (rolling window). Default: the whole stream.

    Attributes
    ----------
    labels_ : numpy.ndarray of shape (k,) or None
        Current class order of the rows and columns.
    n_samples_ : int
        Samples passed to update since the last reset.

    Notes
    -----
    Without a window only the (k, k) matrix is stored: O(k^2) memory however long the stream is.
    With a window the last `window` label pairs are stored and the matrix is recounted from them: O(window) memory.

    Examples
    --------
    >>> cm = StreamingConfusionMatrix()
    >>> cm.update([0, 1], [0, 1]).update([1, 2], [2, 2]).result()
    array([[1, 0, 0],
           [0, 1, 1],
           [0, 0, 1]])
    """

    def __init__(self, labels=None, window=None):
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim != 1 or labels.size == 0:
                raise ValueError("labels must be a non-empty 1D array")
            if np.unique(labels).size != labels.size:
                raise ValueError("labels must not contain duplicates")
        self._fixed_labels = labels
        self.window = _check_window(window)
        self.reset()

    def reset(self):
        """Forget every sample seen so far.

        Returns
        -------
        self

        Notes
        -----
        O(1).
        """
        self.labels_ = self._fixed_labels
        k = 0 if self.labels_ is None else self.labels_.size
        self._matrix = np.zeros((k, k), dtype=np.int64)
        self._true = self._pred = None
        self.n_samples_ = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of true and predicted labels.

        Parameters
        ----------
        y_true_chunk, y_pred_chunk : array-like of shape (n,)
            May be empty.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Not 1D, different lengths, or NaN in the input. Nothing is updated in that case.

        Notes
        -----
        O(n log k + k^2) time for the whole-stream matrix; O(n + window) with a window.
        """
        y_true, y_pred = _check_chunk(y_true_chunk, y_pred_chunk)
        if y_true.size == 0:
            return self
        self.n_samples_ += y_true.size
        if self.window is not None:
            self._true = _append_window(self._true, y_true, self.window)
            self._pred = _append_window(self._pred, y_pred, self.window)
            if self._fixed_labels is None:
                self.labels_ = np.unique(np.concatenate([self._true, self._pred]))
            return self
        if self._fixed_labels is None:
            seen = np.concatenate([y_true, y_pred])
            merged = seen if self.labels_ is None else np.concatenate([self.labels_, seen])
            merged = np.unique(merged)
            if self.labels_ is None or merged.size != self.labels_.size:
                # new classes: move the old counts to their new rows and columns
                grown = np.zeros((merged.size, merged.size), dtype=np.int64)
                if self.labels_ is not None and self.labels_.size:
                    pos = np.searchsorted(merged, self.labels_)
                    grown[np.ix_(pos, pos)] = self._matrix
                self._matrix, self.labels_ = grown, merged
        self._matrix += confusion_matrix(y_true, y_pred, labels=self.labels_)
        return self

    def result(self):
        """Current confusion matrix.

        Returns
        -------
        numpy.ndarray of int, shape (k, k)
            cm[i, j] = samples that are labels_[i] and were predicted as labels_[j]. Shape (0, 0) before any data.

        Notes
        -----
        O(k^2) for the whole stream; O(window log k + k^2) with a window.
        """
        if self.window is None:
            return self._matrix.copy()
        if self._true is None:
            k = 0 if self.labels_ is None else self.labels_.size
            return np.zeros((k, k), dtype=np.int64)
        return confusion_matrix(self._true, self._pred, labels=self.labels_)


class _StreamingClassificationMetric:
    """Shared base: a metric that is a function of a streaming confusion matrix.

    Subclasses implement _from_counts(cm, labels). Keeping only the confusion matrix means every
    classification metric updates in the same O(n log k) and can be derived without storing the stream.
    """

    def __init__(self, labels=None, window=None):
        self._cm = StreamingConfusionMatrix(labels=labels, window=window)

    @property
    def window(self):
        """Rolling-window length in samples, or None for the whole stream."""
        return self._cm.window

    @property
    def n_samples_(self):
        """Samples passed to update since the last reset."""
        return self._cm.n_samples_

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of true and predicted labels (see StreamingConfusionMatrix.update).

        Returns
        -------
        self
        """
        self._cm.update(y_true_chunk, y_pred_chunk)
        return self

    def reset(self):
        """Forget every sample seen so far.

        Returns
        -------
        self
        """
        self._cm.reset()
        return self

    def result(self):
        """Current value of the metric; NaN if nothing has been counted yet.

        Returns
        -------
        float (or a per-class numpy.ndarray for average=None)

        Notes
        -----
        O(k^2) from the stored confusion matrix.
        """
        cm = self._cm.result()
        if cm.sum() == 0:
            return float("nan")
        return self._from_counts(cm, self._cm.labels_)


class StreamingAccuracy(_StreamingClassificationMetric):
    """Accuracy over a stream: correct predictions / predictions.

    Parameters
    ----------
    labels : array-like, optional
        Fixed class list (see StreamingConfusionMatrix).
    window : int, optional
        Rolling window in samples. Default: cumulative accuracy over the whole stream.

    Examples
    --------
    >>> acc = StreamingAccuracy()
    >>> acc.update([0, 1], [0, 1]).update([1, 0], [0, 0]).result()
    0.75
    >>> StreamingAccuracy(window=2).update([0, 1], [0, 1]).update([1, 0], [0, 0]).result()
    0.5
    """

    def _from_counts(self, cm, labels):
        return float(np.trace(cm) / cm.sum())


class _StreamingPRF(_StreamingClassificationMetric):
    _which = 0  # 0 precision, 1 recall, 2 f1

    def __init__(self, average="binary", pos_label=1, labels=None, zero_division=0.0, window=None):
        if average not in VALID_AVERAGES:
            raise ValueError(f"average must be one of {VALID_AVERAGES}, got {average!r}")
        super().__init__(labels=labels, window=window)
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division

    def _from_counts(self, cm, labels):
        tp, pred_pos, true_pos = np.diag(cm), cm.sum(axis=0), cm.sum(axis=1)
        if self.average == "binary":
            if labels.size > 2:
                raise ValueError(
                    f"average='binary' needs at most 2 classes, got {labels.size} "
                    f"{labels.tolist()}. Use average='macro', 'micro' or 'weighted'"
                )
            hit = np.flatnonzero(labels == self.pos_label)
            if hit.size == 0:
                # the positive class has not shown up in the stream yet
                return float(self.zero_division)
            tp, pred_pos, true_pos = tp[hit], pred_pos[hit], true_pos[hit]
        return _average_prf(tp, pred_pos, true_pos, self.average, self.zero_division)[self._which]


class StreamingPrecision(_StreamingPRF):
    """Precision over a stream: TP / (TP + FP).

    Parameters
    ----------
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        Same meaning as in precision(). None returns one value per class in the order of labels_.
    pos_label : default 1
        The positive class for average="binary".
    labels : array-like, optional
        Fixed class list.
    zero_division : float, default 0.0
        Value used when a denominator is 0 (also returned by "binary" while pos_label has not appeared yet).
    window : int, optional
        Rolling window in samples. Default: the whole stream.

    Raises
    ------
    ValueError
        Unknown average; or, from result(), average="binary" on a stream with more than 2 classes.

    Examples
    --------
    >>> StreamingPrecision().update([1, 0, 1], [1, 1, 1]).result()
    0.6666666666666666
    """

    _which = 0


class StreamingRecall(_StreamingPRF):
    """Recall over a stream: TP / (TP + FN). Parameters as for StreamingPrecision.

    Examples
    --------
    >>> StreamingRecall().update([1, 0, 1], [1, 1, 0]).result()
    0.5
    """

    _which = 1


class StreamingF1(_StreamingPRF):
    """F1 score over a stream: harmonic mean of precision and recall. Parameters as for StreamingPrecision.

    Examples
    --------
    >>> round(StreamingF1(average="macro").update([0, 1, 2], [0, 1, 1]).result(), 4)
    0.5556
    """

    _which = 2


class StreamingMSE:
    """Mean squared error over a stream.

    Parameters
    ----------
    window : int, optional
        Rolling window in samples. Default: the whole stream.

    Notes
    -----
    Keeps a running mean of the squared errors, merged chunk by chunk as mean += (chunk_mean - mean) * n_chunk / n,
    which stays accurate on long streams where a running sum would keep growing. O(1) memory (O(window) with a window).

    Examples
    --------
    >>> round(StreamingMSE().update([1.0, 2.0], [1.0, 4.0]).update([0.0], [1.0]).result(), 4)
    1.6667
    """

    def __init__(self, window=None):
        self.window = _check_window(window)
        self.reset()

    def reset(self):
        """Forget every sample seen so far. Returns self."""
        self._mean = 0.0
        self._n = 0
        self._sq = None
        self.n_samples_ = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of targets and predictions.

        Parameters
        ----------
        y_true_chunk, y_pred_chunk : array-like of shape (n,)
            Numeric. May be empty.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Not 1D, different lengths, not numeric, or NaN in the input.

        Notes
        -----
        O(n).
        """
        y_true, y_pred = _check_chunk(y_true_chunk, y_pred_chunk)
        if y_true.size == 0:
            return self
        try:
            sq = (y_true.astype(float) - y_pred.astype(float)) ** 2
        except (TypeError, ValueError) as exc:
            raise ValueError(f"StreamingMSE needs numeric input ({exc})") from exc
        self.n_samples_ += sq.size
        if self.window is not None:
            self._sq = _append_window(self._sq, sq, self.window)
            return self
        self._n += sq.size
        self._mean += (float(sq.mean()) - self._mean) * sq.size / self._n
        return self

    def result(self):
        """Current mean squared error; NaN before any data.

        Returns
        -------
        float
        """
        if self.window is not None:
            return float("nan") if self._sq is None else float(self._sq.mean())
        return float("nan") if self._n == 0 else float(self._mean)


class StreamingAUC:
    """Area under the ROC curve, accumulated over a stream of scores.

    Parameters
    ----------
    pos_label : default 1
        The positive class; every other label counts as negative.
    n_bins : int, default 1000
        Number of equal-width score bins kept per class (whole-stream mode).
    score_range : (float, float), default (0.0, 1.0)
        Lowest and highest score. Scores outside are counted in the first or last bin.
    window : int, optional
        Rolling window in samples. With a window the exact AUC of the stored samples is returned. Default: the whole stream.

    Notes
    -----
    The exact AUC needs every score, which a stream cannot keep. Instead the scores of the positive and of the negative
    samples are counted into two histograms. The AUC is the chance that a random positive scores higher than a random
    negative, so from the histograms it is

        sum_b pos[b] * (negatives in lower bins + negatives in bin b / 2) / (P * N).

    Pairs that share a bin count as ties (half), exactly like tied scores in roc_auc_score. The result is exact when no
    positive and negative with different scores share a bin, and otherwise off by at most half the fraction of
    positive-negative pairs that share a bin. Memory is O(n_bins) for any stream length.

    Examples
    --------
    >>> auc_ = StreamingAUC()
    >>> auc_.update([0, 1], [0.1, 0.9]).update([0, 1], [0.4, 0.35]).result()
    0.75
    """

    def __init__(self, pos_label=1, n_bins=1000, score_range=(0.0, 1.0), window=None):
        if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)):
            raise TypeError(f"n_bins must be an integer, got {n_bins!r}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        lo, hi = (float(v) for v in score_range)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"score_range must be finite with low < high, got {score_range!r}")
        self.pos_label = pos_label
        self.n_bins = int(n_bins)
        self.score_range = (lo, hi)
        self.window = _check_window(window)
        self.reset()

    def reset(self):
        """Forget every sample seen so far. Returns self."""
        self._pos = np.zeros(self.n_bins, dtype=np.int64)
        self._neg = np.zeros(self.n_bins, dtype=np.int64)
        self._true = self._score = None
        self.n_samples_ = 0
        return self

    def update(self, y_true_chunk, y_score_chunk):
        """Add one chunk of true labels and scores.

        Parameters
        ----------
        y_true_chunk : array-like of shape (n,)
            True labels.
        y_score_chunk : array-like of shape (n,)
            Score of the positive class; higher means more positive. May be empty.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Not 1D, different lengths, NaN, or non-numeric or infinite scores.

        Notes
        -----
        O(n + n_bins) (one bincount per class).
        """
        y_true, y_score = _check_chunk(y_true_chunk, y_score_chunk, ("y_true", "y_score"))
        if y_true.size == 0:
            return self
        try:
            score = y_score.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"y_score must be numeric ({exc})") from exc
        if not np.isfinite(score).all():
            raise ValueError("y_score must be finite")
        self.n_samples_ += score.size
        is_pos = y_true == self.pos_label
        if self.window is not None:
            self._true = _append_window(self._true, is_pos, self.window)
            self._score = _append_window(self._score, score, self.window)
            return self
        lo, hi = self.score_range
        bins = np.floor((score - lo) / (hi - lo) * self.n_bins).astype(np.int64)
        bins = np.clip(bins, 0, self.n_bins - 1)
        self._pos += np.bincount(bins[is_pos], minlength=self.n_bins)
        self._neg += np.bincount(bins[~is_pos], minlength=self.n_bins)
        return self

    def result(self):
        """Current AUC; NaN until at least one positive and one negative sample have been seen.

        Returns
        -------
        float between 0 and 1

        Notes
        -----
        O(n_bins) for the whole stream; O(window log window) with a window.
        """
        if self.window is not None:
            if self._true is None or self._true.all() or not self._true.any():
                return float("nan")
            return roc_auc_score(self._true.astype(int), self._score, pos_label=1)
        n_pos, n_neg = int(self._pos.sum()), int(self._neg.sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        neg_below = np.cumsum(self._neg) - self._neg
        wins = (self._pos * (neg_below + 0.5 * self._neg)).sum()
        return float(wins / (float(n_pos) * float(n_neg)))
