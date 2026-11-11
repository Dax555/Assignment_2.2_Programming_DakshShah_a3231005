"""Unit tests for numcompute_stream.sort_search."""
import numpy as np
import pytest

from numcompute_stream import sort_search as ss


def test_stable_sort_matches_numpy_both_directions():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10, size=100)
    np.testing.assert_array_equal(ss.stable_sort(a), np.sort(a))
    np.testing.assert_array_equal(ss.stable_sort(a, ascending=False), np.sort(a)[::-1])


def test_stable_sort_ties_keep_original_order():
    v = np.array([2, 1, 2, 1, 3, 3.0])
    np.testing.assert_array_equal(ss.stable_sort(v, True, return_indices=True), [1, 3, 0, 2, 4, 5])
    np.testing.assert_array_equal(ss.stable_sort(v, False, return_indices=True), [4, 5, 0, 2, 1, 3])


def test_stable_sort_nan_always_last():
    a = np.array([np.nan, 3.0, 1.0, np.nan, 2.0])
    np.testing.assert_array_equal(ss.stable_sort(a)[:3], [1, 2, 3])
    np.testing.assert_array_equal(ss.stable_sort(a, ascending=False)[:3], [3, 2, 1])
    assert np.isnan(ss.stable_sort(a)[-2:]).all()


def test_stable_sort_empty_all_equal_and_non_contiguous():
    assert ss.stable_sort([]).size == 0
    np.testing.assert_array_equal(ss.stable_sort([5, 5, 5], return_indices=True), [0, 1, 2])
    view = np.arange(20)[::-2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(ss.stable_sort(view), np.sort(view))


def test_stable_sort_along_axis_of_2d_array():
    a = np.array([[3.0, np.nan, 1.0], [2.0, 2.0, 5.0]])
    np.testing.assert_array_equal(ss.stable_sort(a, axis=0), np.sort(a, axis=0))
    np.testing.assert_array_equal(ss.stable_sort(a[1:], axis=1), [[2.0, 2.0, 5.0]])
    down = ss.stable_sort(a, ascending=False, axis=1)
    np.testing.assert_array_equal(down[0, :2], [3.0, 1.0])
    assert np.isnan(down[0, 2])                                # NaN still last
    np.testing.assert_array_equal(ss.stable_sort(a, False, True, axis=1)[1], [2, 0, 1])


def test_stable_sort_descending_strings_and_extreme_ints():
    np.testing.assert_array_equal(ss.stable_sort(["b", "a", "c"], ascending=False), ["c", "b", "a"])
    big = np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max])
    np.testing.assert_array_equal(ss.stable_sort(big, ascending=False), big[::-1])


def test_multi_key_sort_columns_of_2d_array():
    table = np.array([[2, 1], [1, 9], [2, 0], [1, 3]])
    np.testing.assert_array_equal(ss.multi_key_sort(table), [3, 1, 2, 0])
    np.testing.assert_array_equal(ss.multi_key_sort(table, [True, False]), [1, 3, 0, 2])


def test_stable_sort_invalid_inputs():
    with pytest.raises(ValueError):
        ss.stable_sort(5.0)                                    # 0-D
    with pytest.raises(ValueError):
        ss.stable_sort(np.ones((2, 2)), axis=2)
    with pytest.raises(TypeError):
        ss.stable_sort([1, 2], ascending="yes")


def test_multi_key_sort_matches_lexsort_and_mixed_directions():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
    np.testing.assert_array_equal(ss.multi_key_sort([a, b]), np.lexsort((b, a)))
    np.testing.assert_array_equal(
        ss.multi_key_sort([[1, 1, 2, 2], [10, 20, 30, 40]], ascending=[True, False]), [1, 0, 3, 2])


def test_multi_key_sort_ties_nan_and_invalid_inputs():
    np.testing.assert_array_equal(ss.multi_key_sort([[1, 1, 1]]), [0, 1, 2])
    perm = ss.multi_key_sort([np.array([1.0, np.nan, 1.0]), np.array([2, 1, 1])])
    np.testing.assert_array_equal(perm, [2, 0, 1])
    with pytest.raises(ValueError):
        ss.multi_key_sort([])
    with pytest.raises(ValueError):
        ss.multi_key_sort([[1, 2], [1]])
    with pytest.raises(ValueError):
        ss.multi_key_sort([[1, 2], [3, 4]], ascending=[True])


def test_topk_largest_smallest_indices_and_values():
    a = np.array([5.0, 1.0, 9.0, 3.0, 7.0])
    np.testing.assert_array_equal(ss.topk(a, 2), [2, 4])
    np.testing.assert_array_equal(ss.topk(a, 2, largest=False), [1, 3])
    np.testing.assert_array_equal(ss.topk(a, 3, return_indices=False), [9, 7, 5])


def test_topk_ties_earliest_index_wins():
    np.testing.assert_array_equal(ss.topk(np.array([5, 5, 5, 1.0]), 2), [0, 1])
    np.testing.assert_array_equal(ss.topk(np.array([2, 2, 2, 2]), 3, largest=False), [0, 1, 2])


def test_topk_extreme_k_and_empty():
    a = np.array([3.0, 1.0, 2.0])
    assert ss.topk(a, 0).size == 0
    np.testing.assert_array_equal(ss.topk(a, 100), [0, 2, 1])   # k > n clipped
    assert ss.topk([], 5).size == 0


def test_topk_nan_ignored_with_warning_and_non_contiguous():
    with pytest.warns(UserWarning, match="2 NaN"):
        idx = ss.topk(np.array([1.0, np.nan, 5.0, np.nan, 3.0]), 2)
    np.testing.assert_array_equal(idx, [2, 4])                 # indices into the ORIGINAL array
    view = np.arange(20.0)[::2]
    np.testing.assert_array_equal(ss.topk(view, 2, return_indices=False), [18.0, 16.0])


def test_topk_invalid_inputs():
    with pytest.raises(TypeError):
        ss.topk([1, 2], 1.5)
    with pytest.raises(ValueError):
        ss.topk([1, 2], -1)
    with pytest.raises(TypeError):
        ss.topk(["a", "b"], 1)


def test_quickselect_matches_sorted_for_every_k():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 20, size=40)
    s = np.sort(a)
    for k in range(40):
        assert ss.quickselect(a, k) == s[k]


def test_quickselect_all_equal_nan_and_input_unchanged():
    assert ss.quickselect([4, 4, 4, 4], 2) == 4
    with pytest.warns(UserWarning):
        assert ss.quickselect([np.nan, 3.0, 1.0, 2.0], 2) == 3.0
    a = np.array([3, 1, 2])
    ss.quickselect(a, 1)
    np.testing.assert_array_equal(a, [3, 1, 2])


def test_quickselect_invalid_inputs():
    with pytest.raises(ValueError):
        ss.quickselect([1, 2, 3], 3)                           # k out of range
    with pytest.raises(ValueError):
        ss.quickselect([], 0)
    with pytest.raises(TypeError):
        ss.quickselect([1, 2], 0.5)


def test_binary_search_found_missing_and_leftmost_duplicate():
    assert ss.binary_search([1, 2, 3, 5], 2) == (1, True)
    assert ss.binary_search([1, 2, 3, 5], 4) == (3, False)     # insertion point
    assert ss.binary_search([1, 2, 2, 2, 3], 2) == (1, True)   # leftmost
    assert ss.binary_search([1, 2, 3], 2, return_index=False) == (2, True)
    assert ss.binary_search([1, 2, 3], 7, return_index=False) == (None, False)


@pytest.mark.parametrize("method", ["loop", "searchsorted"])
def test_binary_search_agrees_with_searchsorted_empty_and_nan(method):
    a = np.sort(np.random.default_rng(4).integers(0, 30, 100))
    for x in range(-2, 33):
        assert ss.binary_search(a, x, method=method)[0] == np.searchsorted(a, x, side="left")
    assert ss.binary_search([], 1, method=method) == (0, False)
    with pytest.warns(UserWarning):
        assert ss.binary_search([1.0, 2.0, np.nan], 2.0, method=method) == (1, True)
    with pytest.warns(UserWarning):
        assert ss.binary_search([1.0, 2.0, np.nan], 9.0, method=method) == (2, False)


def test_binary_search_does_not_sort_and_check_sorted_catches_it():
    assert ss.binary_search(ss.stable_sort([3, 1, 2, 5]), 2) == (1, True)
    with pytest.raises(ValueError):
        ss.binary_search([3, 1, 2], 2, check_sorted=True)
    with pytest.raises(ValueError):
        ss.binary_search([1.0, np.nan, 2.0], 2.0, check_sorted=True)
    assert ss.binary_search([1, 2, 3], 3, check_sorted=True) == (2, True)


def test_binary_search_invalid_inputs():
    with pytest.raises(ValueError):
        ss.binary_search([1, 2], np.nan)
    with pytest.raises(ValueError):
        ss.binary_search([1, 2], [1])
    with pytest.raises(TypeError):
        ss.binary_search([1, 2], 1, return_index="yes")
    with pytest.raises(ValueError):
        ss.binary_search([1, 2], 1, method="linear")
