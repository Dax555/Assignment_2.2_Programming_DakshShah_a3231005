# tests for metrics.py

import numpy as np
import pytest

from numCompute.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_auc_score,
    roc_curve,
)

Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])

# accuracy


def test_accuracy_basic():
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_accuracy_all_right_and_all_wrong():
    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy([1, 2, 3], [3, 1, 2]) == 0.0


def test_accuracy_string_labels():
    assert accuracy(["cat", "dog", "cat"], ["cat", "cat", "cat"]) == pytest.approx(
        2 / 3
    )


def test_accuracy_accepts_column_vector():
    assert accuracy(np.array([[1], [0]]), [1, 1]) == 0.5


def test_accuracy_non_contiguous_input():
    X = np.arange(12).reshape(3, 4)
    # X[:, 0] is a strided view, not a contiguous array
    assert accuracy(X[:, 0], X[:, 0].copy()) == 1.0

# confusion matrix

def test_confusion_matrix_three_classes():
    expected = [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    np.testing.assert_array_equal(confusion_matrix(Y_TRUE, Y_PRED), expected)


def test_confusion_matrix_total_equals_samples():
    assert confusion_matrix(Y_TRUE, Y_PRED).sum() == len(Y_TRUE)


def test_confusion_matrix_custom_label_order():
    cm = confusion_matrix(
        ["cat", "dog", "cat"], ["cat", "cat", "cat"], labels=["dog", "cat"]
    )
    np.testing.assert_array_equal(cm, [[0, 1], [0, 2]])


def test_confusion_matrix_non_consecutive_labels():
    # labels 1, 5, 9 must not be used as row numbers directly
    cm = confusion_matrix([1, 5, 9, 9], [1, 9, 9, 5])
    np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 0, 1], [0, 1, 1]])


def test_confusion_matrix_duplicate_labels_raise():
    with pytest.raises(ValueError, match="duplicates"):
        confusion_matrix([0], [0], labels=[0, 0])

# precision / recall / f1

def test_binary_precision_counts_false_positives():
    # TP=2, FP=1 : 2/3. An early version wrongly returned 1.0 here.
    assert precision([0, 1, 1, 0], [1, 1, 1, 0]) == pytest.approx(2 / 3)


def test_binary_recall_and_f1():
    # TP=1, FN=1 : recall 0.5; precision 0.5 -> f1 0.5
    assert recall([0, 1, 1, 0], [1, 1, 0, 0]) == 0.5
    assert f1([0, 1, 1, 0], [1, 1, 0, 0]) == 0.5


def test_per_class_scores_with_average_none():
    np.testing.assert_allclose(
        precision(Y_TRUE, Y_PRED, average=None), [0.5, 2 / 3, 1.0]
    )
    np.testing.assert_allclose(recall(Y_TRUE, Y_PRED, average=None), [0.5, 1.0, 0.5])


def test_macro_average():
    assert precision(Y_TRUE, Y_PRED, average="macro") == pytest.approx(
        (0.5 + 2 / 3 + 1) / 3
    )


def test_micro_average_equals_accuracy():
    # with one label per sample, micro precision/recall/f1 all equal accuracy
    acc = accuracy(Y_TRUE, Y_PRED)
    for metric in (precision, recall, f1):
        assert metric(Y_TRUE, Y_PRED, average="micro") == pytest.approx(acc)


def test_weighted_average():
    # every class has 2 samples, so weighted equals macro here
    assert recall(Y_TRUE, Y_PRED, average="weighted") == pytest.approx(
        recall(Y_TRUE, Y_PRED, average="macro")
    )


def test_zero_division_when_class_never_predicted():
    # nothing predicted as 1 : precision is 0/0
    assert precision([1, 0], [0, 0]) == 0.0
    assert precision([1, 0], [0, 0], zero_division=1.0) == 1.0


def test_string_pos_label():
    assert precision(["n", "y", "y"], ["y", "y", "n"], pos_label="y") == 0.5


def test_unknown_pos_label_raises():
    with pytest.raises(ValueError, match="pos_label"):
        precision([0, 2], [0, 2])


def test_unknown_average_raises():
    with pytest.raises(ValueError, match="average"):
        f1([0, 1], [0, 1], average="avg")

# input checks shared by every metric

@pytest.mark.parametrize(
    "y_true, y_pred, message",
    [
        ([1, 2], [1], "same length"),
        ([], [], "empty"),
        ([1.0, np.nan], [1.0, 1.0], "NaN"),
        (np.ones((2, 2)), np.ones((2, 2)), "1D"),
    ],
)
def test_bad_inputs_raise(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        accuracy(y_true, y_pred)

# mse

def test_mse_basic():
    assert mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(4 / 3)


def test_mse_perfect_prediction_is_zero():
    assert mse([1, 2, 3], [1, 2, 3]) == 0.0


def test_mse_per_column():
    y = np.array([[0.0, 0.0], [0.0, 0.0]])
    p = np.array([[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(mse(y, p, axis=0), [1.0, 4.0])


def test_mse_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        mse([1, 2], [1, 2, 3])


def test_mse_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        mse([1.0, np.nan], [1.0, 2.0])

# roc curve / auc

def test_roc_curve_known_values():
    fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
    assert np.isinf(thr[0])


def test_roc_curve_tied_scores_move_diagonally():
    # all scores equal: no threshold can separate them, so one jump to (1, 1)
    fpr, tpr, _ = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(fpr, [0, 1])
    np.testing.assert_allclose(tpr, [0, 1])


def test_roc_auc_perfect_random_and_worst():
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert roc_auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_roc_auc_known_value():
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75


def test_roc_needs_both_classes():
    with pytest.raises(ValueError, match="both positive and negative"):
        roc_curve([1, 1], [0.2, 0.3])


def test_roc_rejects_multiclass():
    with pytest.raises(ValueError, match="binary"):
        roc_curve([0, 1, 2], [0.1, 0.2, 0.3])


def test_auc_decreasing_x_gives_positive_area():
    assert auc([1, 0.5, 0], [1, 1, 0]) == pytest.approx(0.75)


def test_auc_non_monotonic_x_raises():
    with pytest.raises(ValueError, match="monotonic"):
        auc([0, 1, 0.5], [0, 1, 1])



# binary average with more than two classes, and infinity

def test_binary_average_with_three_classes_raises():
    with pytest.raises(ValueError, match="at most 2 classes"):
        precision(Y_TRUE, Y_PRED)
    # the other averages still work on the same data
    assert 0.0 <= f1(Y_TRUE, Y_PRED, average="macro") <= 1.0


def test_mse_rejects_infinity():
    with pytest.raises(ValueError, match="infinity"):
        mse([1.0, np.inf], [1.0, 2.0])


def test_roc_curve_rejects_infinite_scores():
    with pytest.raises(ValueError, match="infinity"):
        roc_curve([0, 1, 1], [1.0, np.inf, 2.0])