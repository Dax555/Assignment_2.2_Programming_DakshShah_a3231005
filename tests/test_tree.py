# tests for tree.py: the decision tree and its buffered streaming mode
import numpy as np
import pytest

from numcompute_stream.tree import (DecisionTreeClassifier, _best_split,
                                    entropy, gini)
from numcompute_stream.utils import NotFittedError


def make_blobs(n=300, seed=0):
    """Three well-separated 2-D classes."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    y = rng.integers(0, 3, size=n)
    X = centres[y] + rng.normal(scale=0.5, size=(n, 2))
    return X, y


def brute_force_split(X, y, n_classes):
    """Loop reference: try every feature and every midpoint, keep the first best."""
    def gini_of(labels):
        if len(labels) == 0:
            return 0.0
        p = np.bincount(labels, minlength=n_classes) / len(labels)
        return 1.0 - float((p ** 2).sum())

    best = (-1, np.nan, 0.0)
    before = gini_of(y)
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for a, b in zip(values[:-1], values[1:]):
            t = a + (b - a) / 2.0
            mask = X[:, f] <= t
            after = (mask.sum() * gini_of(y[mask]) + (~mask).sum() * gini_of(y[~mask])) / len(y)
            if before - after > best[2] + 1e-12:
                best = (f, t, before - after)
    return best


# ---------------------------------------------------------------- impurity
def test_gini_and_entropy_known_values():
    assert gini([5, 5]) == pytest.approx(0.5)
    assert gini([10, 0]) == 0.0
    assert entropy([5, 5]) == pytest.approx(1.0)
    assert entropy([10, 0]) == 0.0
    assert entropy([1, 1, 1, 1]) == pytest.approx(2.0)


def test_impurity_of_empty_group_is_zero_not_nan():
    assert gini([0, 0]) == 0.0
    assert entropy([0, 0]) == 0.0


def test_impurity_is_vectorised_over_leading_axes():
    counts = np.array([[5, 5], [10, 0], [1, 3]])
    assert gini(counts).shape == (3,)
    np.testing.assert_allclose(gini(counts), [0.5, 0.0, 0.375])


# ------------------------------------------------------------ split search
def test_best_split_matches_brute_force_loop():
    rng = np.random.default_rng(1)
    for _ in range(20):
        X = rng.integers(0, 6, size=(40, 3)).astype(float)   # many ties
        y = rng.integers(0, 3, size=40)
        f, t, g = _best_split(X, y, 3)
        bf, bt, bg = brute_force_split(X, y, 3)
        assert f == bf
        assert t == pytest.approx(bt)
        assert g == pytest.approx(bg)


def test_best_split_tie_goes_to_lowest_feature():
    # both columns separate the classes perfectly: feature 0 must win
    X = np.array([[0.0, 9.0], [1.0, 8.0], [2.0, 1.0], [3.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    f, t, g = _best_split(X, y, 2)
    assert f == 0 and t == pytest.approx(1.5) and g == pytest.approx(0.5)


def test_best_split_zero_variance_returns_no_split():
    X = np.ones((10, 3))
    y = np.array([0, 1] * 5)
    assert _best_split(X, y, 2)[0] == -1


def test_best_split_can_separate_missing_from_present():
    # the class is decided by whether the feature is missing
    X = np.array([[1.0], [2.0], [3.0], [np.nan], [np.nan], [np.nan]])
    y = np.array([0, 0, 0, 1, 1, 1])
    f, t, g = _best_split(X, y, 2)
    assert f == 0 and t == 3.0 and g == pytest.approx(0.5)


def test_threshold_stays_between_adjacent_floats():
    a = 1.0
    b = np.nextafter(a, 2.0)                 # the midpoint rounds onto a or b
    X = np.array([[a], [a], [b], [b]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.predict(X).tolist() == [0, 0, 1, 1]


# --------------------------------------------------------------------- fit
def test_fit_learns_separable_data():
    X, y = make_blobs()
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert tree.score(X, y) > 0.98


def test_fit_reproduces_lecture_example():
    # twelve students, three yes/no features (classification-tree.html):
    # depth 2 asks "submitted?" then "attended?" and gets student 5 wrong
    rows = np.array([
        [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1],
        [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 0]])
    X, y = rows[:, :3].astype(float), rows[:, 3]
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert tree.feature_[0] == 0                      # submitted? asked first
    assert tree.get_n_leaves() == 3
    pred = tree.predict(X)
    assert (pred != y).sum() == 1 and pred[4] == 0    # only student 5 is missed
    for criterion in ("gini", "entropy"):             # both rulers agree here
        other = DecisionTreeClassifier(max_depth=2, criterion=criterion).fit(X, y)
        np.testing.assert_array_equal(other.predict(X), pred)


def test_max_depth_is_respected():
    X, y = make_blobs(seed=3)
    for depth in (1, 2, 4):
        tree = DecisionTreeClassifier(max_depth=depth).fit(X, y)
        assert tree.get_depth() <= depth


def test_min_samples_split_stops_small_nodes():
    X, y = make_blobs(n=60, seed=4)
    tree = DecisionTreeClassifier(max_depth=None, min_samples_split=61).fit(X, y)
    assert tree.get_n_leaves() == 1          # the root may not split


def test_single_class_gives_a_single_leaf():
    X = np.random.default_rng(0).normal(size=(20, 3))
    tree = DecisionTreeClassifier().fit(X, np.zeros(20, dtype=int))
    assert tree.n_nodes_ == 1
    assert (tree.predict(X) == 0).all()


def test_zero_variance_features_predict_majority():
    X = np.ones((9, 2))
    y = np.array([0] * 6 + [1] * 3)
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.n_nodes_ == 1
    assert (tree.predict(X) == 0).all()


def test_leaf_tie_goes_to_first_class():
    X = np.ones((4, 1))
    tree = DecisionTreeClassifier().fit(X, np.array(["b", "a", "b", "a"]))
    assert tree.predict(X).tolist() == ["a"] * 4


def test_string_labels_and_predict_proba():
    X, y = make_blobs()
    names = np.array(["cat", "dog", "emu"])[y]
    tree = DecisionTreeClassifier(max_depth=4).fit(X, names)
    assert tree.classes_.tolist() == ["cat", "dog", "emu"]
    proba = tree.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(tree.classes_[proba.argmax(axis=1)], tree.predict(X))


def test_nan_features_are_routed_right_consistently():
    rng = np.random.default_rng(5)
    X, y = make_blobs(seed=5)
    X[rng.random(X.shape) < 0.1] = np.nan
    tree = DecisionTreeClassifier(max_depth=4).fit(X, y)
    pred = tree.predict(X)                     # must not raise or produce NaN
    assert pred.shape == y.shape
    assert tree.score(X, y) > 0.8
    # a fully missing row follows the right branch at every node
    leaf = tree.apply(np.full((1, 2), np.nan))[0]
    node = 0
    while tree.feature_[node] != -1:
        node = tree.right_[node]
    assert leaf == node


def test_predict_works_on_non_contiguous_and_integer_input():
    X, y = make_blobs()
    tree = DecisionTreeClassifier().fit(X, y)
    big = np.repeat(X, 2, axis=0)
    np.testing.assert_array_equal(tree.predict(big[::2]), tree.predict(X))
    Xi = np.array([[0, 0], [5, 5], [0, 5]])
    assert tree.predict(Xi).tolist() == [0, 1, 2]


def test_max_features_is_reproducible_and_validated():
    X, y = make_blobs()
    a = DecisionTreeClassifier(max_features=1, random_state=7).fit(X, y)
    b = DecisionTreeClassifier(max_features=1, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.feature_, b.feature_)
    np.testing.assert_array_equal(a.threshold_, b.threshold_)
    for good in ("sqrt", "log2", 0.5, 2, None):
        DecisionTreeClassifier(max_features=good).fit(X, y)
    for bad in (0, 1.5, "all", True):
        with pytest.raises((ValueError, TypeError)):
            DecisionTreeClassifier(max_features=bad)


@pytest.mark.parametrize("kwargs", [
    {"max_depth": 0}, {"max_depth": 2.5}, {"min_samples_split": 1},
    {"buffer_size": 0}, {"buffer_size": "big"}, {"criterion": "mse"},
])
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises((ValueError, TypeError)):
        DecisionTreeClassifier(**kwargs)


def test_fit_rejects_bad_shapes_with_clear_messages():
    X, y = make_blobs(n=10)
    tree = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="2-D"):
        tree.fit(X[:, 0], y)
    with pytest.raises(ValueError, match="same number of samples"):
        tree.fit(X, y[:-1])
    with pytest.raises(ValueError, match="0 samples"):
        tree.fit(X[:0], y[:0])
    with pytest.raises(ValueError, match="infinity"):
        tree.fit(np.where(X > 100, X, np.inf), y)
    with pytest.raises(ValueError, match="NaN"):
        tree.fit(X, np.where(y == 0, np.nan, 1.0))
    with pytest.raises(TypeError, match="y is required"):
        tree.fit(X, None)


def test_predict_checks_fitted_and_feature_count():
    X, y = make_blobs(n=30)
    tree = DecisionTreeClassifier()
    with pytest.raises(NotFittedError):
        tree.predict(X)
    tree.fit(X, y)
    with pytest.raises(ValueError, match="expected 2 features"):
        tree.predict(np.ones((3, 5)))


# --------------------------------------------------------------- streaming
def test_partial_fit_buffers_until_buffer_size():
    X, y = make_blobs(n=100)
    tree = DecisionTreeClassifier(buffer_size=50)
    tree.partial_fit(X[:20], y[:20])
    assert not tree.is_fitted and tree.buffer_count_ == 20 and tree.n_retrains_ == 0
    with pytest.raises(NotFittedError, match="20 so far"):
        tree.predict(X)
    tree.partial_fit(X[20:40], y[20:40])
    assert not tree.is_fitted
    tree.partial_fit(X[40:60], y[40:60])              # 60 >= 50 -> retrain
    assert tree.is_fitted and tree.n_retrains_ == 1
    assert tree.buffer_count_ == 0                    # buffer cleared
    assert tree.n_samples_seen_ == 60


def test_retrain_uses_exactly_the_buffered_data():
    X, y = make_blobs(n=120)
    stream = DecisionTreeClassifier(max_depth=4, buffer_size=60)
    for start in range(0, 60, 15):
        stream.partial_fit(X[start:start + 15], y[start:start + 15])
    batch = DecisionTreeClassifier(max_depth=4).fit(X[:60], y[:60])
    np.testing.assert_array_equal(stream.feature_, batch.feature_)
    np.testing.assert_allclose(stream.threshold_, batch.threshold_, equal_nan=True)
    np.testing.assert_array_equal(stream.value_, batch.value_)


def test_predict_uses_last_tree_between_retrains():
    X, y = make_blobs(n=200)
    tree = DecisionTreeClassifier(buffer_size=50)
    tree.partial_fit(X[:50], y[:50])
    before = tree.predict(X)
    structure = tree.feature_.copy()
    # relabel everything as class 0: the tree must not change until the retrain
    tree.partial_fit(X[50:90], np.zeros(40, dtype=int))
    np.testing.assert_array_equal(tree.predict(X), before)
    np.testing.assert_array_equal(tree.feature_, structure)
    assert tree.n_retrains_ == 1 and tree.buffer_count_ == 40
    tree.partial_fit(X[90:100], np.zeros(10, dtype=int))   # buffer full -> new tree
    assert tree.n_retrains_ == 2
    assert (tree.predict(X) == 0).all()


def test_chunk_larger_than_buffer_retrains_immediately():
    X, y = make_blobs(n=100)
    tree = DecisionTreeClassifier(buffer_size=10).partial_fit(X, y)
    assert tree.n_retrains_ == 1 and tree.buffer_count_ == 0
    assert tree.n_node_samples_[0] == 100            # trained on the whole chunk


def test_empty_chunk_is_a_no_op():
    X, y = make_blobs(n=20)
    tree = DecisionTreeClassifier(buffer_size=10)
    tree.partial_fit(X[:0], y[:0])
    assert tree.buffer_count_ == 0 and tree.n_samples_seen_ == 0 and not tree.is_fitted


def test_flush_trains_on_a_partial_buffer():
    X, y = make_blobs(n=30)
    tree = DecisionTreeClassifier(buffer_size=1000).partial_fit(X, y)
    assert not tree.is_fitted
    tree.flush()
    assert tree.is_fitted and tree.n_retrains_ == 1 and tree.buffer_count_ == 0
    tree.flush()                                      # empty buffer: nothing happens
    assert tree.n_retrains_ == 1


def test_partial_fit_rejects_changed_feature_count_without_buffering():
    X, y = make_blobs(n=20)
    tree = DecisionTreeClassifier(buffer_size=100).partial_fit(X, y)
    with pytest.raises(ValueError, match="expected 2 features"):
        tree.partial_fit(np.ones((5, 3)), np.zeros(5))
    assert tree.buffer_count_ == 20


def test_buffer_is_independent_of_caller_arrays():
    X, y = make_blobs(n=40)
    Xc, yc = X.copy(), y.copy()
    tree = DecisionTreeClassifier(buffer_size=40, max_depth=3)
    tree.partial_fit(Xc[:20], yc[:20])
    Xc[:] = 0.0                                       # caller reuses its array
    yc[:] = 0
    tree.partial_fit(X[20:], y[20:])
    expected = DecisionTreeClassifier(max_depth=3).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), expected.predict(X))


def test_new_class_in_a_later_buffer_extends_classes():
    X, y = make_blobs(n=300)
    two = y < 2
    tree = DecisionTreeClassifier(buffer_size=50)
    tree.partial_fit(X[two][:50], y[two][:50])
    assert tree.classes_.tolist() == [0, 1]
    tree.partial_fit(X[:40], y[:40])                  # class 2 arrives, no retrain yet
    assert set(tree.predict(X)) <= {0, 1}
    tree.partial_fit(X[40:60], y[40:60])              # retrain
    assert tree.classes_.tolist() == [0, 1, 2]
    assert tree.predict_proba(X).shape[1] == 3


def test_classes_argument_fixes_proba_columns_up_front():
    X, y = make_blobs(n=100)
    only0 = y == 0
    tree = DecisionTreeClassifier(buffer_size=10)
    tree.partial_fit(X[only0][:10], y[only0][:10], classes=[0, 1, 2])
    proba = tree.predict_proba(X[:5])
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba[:, 0], 1.0)


def test_fit_discards_stream_state():
    X, y = make_blobs(n=60)
    tree = DecisionTreeClassifier(buffer_size=100).partial_fit(X[:30], y[:30])
    tree.fit(X, y)
    assert tree.buffer_count_ == 0 and tree.n_samples_seen_ == 0 and tree.n_retrains_ == 0


def test_export_text_names_features():
    X, y = make_blobs()
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    text = tree.export_text(["width", "height"])
    assert "<=" in text and "predict" in text
    assert "width" in text or "height" in text
    with pytest.raises(ValueError, match="feature_names"):
        tree.export_text(["only_one"])


def test_decision_tree_alias_matches_the_brief():
    from numcompute_stream import DecisionTree
    from numcompute_stream.tree import DecisionTree as from_module

    assert DecisionTree is DecisionTreeClassifier and from_module is DecisionTreeClassifier
    tree = DecisionTree(max_depth=5, buffer_size=500)
    assert tree.max_depth == 5 and tree.buffer_size == 500
