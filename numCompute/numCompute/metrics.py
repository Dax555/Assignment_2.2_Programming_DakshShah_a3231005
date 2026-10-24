"""
metrics.py
Metrics for scoring a model's predictions.
Classification: accuracy, precision, recall, f1, confusion_matrix.
Regression: mse. Binary classification: roc_curve, auc, roc_auc_score.
All functions are vectorised and work with any label type, including strings.
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
