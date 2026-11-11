# tests for utils.py

import numpy as np
import pytest
from numcompute_stream.utils import (
    INTERPOLATIONS,
    MISSING_TOKENS,
    as_float_array,
    batch_slices,
    check_axis,
    iter_batches,
    leaky_relu,
    log_softmax,
    logsumexp,
    pairwise_distances,
    relu,
    sigmoid,
    softmax,
    softplus,
    topk_along_axis,
)

# check_axis / as_float_array

def test_check_axis_negative_becomes_positive():
    assert check_axis(-1, 3) == 2
    assert check_axis(0, 3) == 0


def test_check_axis_none_only_when_allowed():
    assert check_axis(None, 2, allow_none=True) is None
    with pytest.raises(TypeError, match="integer"):
        check_axis(None, 2)


@pytest.mark.parametrize(
    "axis, error",
    [(2, ValueError), (-3, ValueError), (True, TypeError), (1.0, TypeError)],
)
def test_check_axis_bad_values_raise(axis, error):
    with pytest.raises(error):
        check_axis(axis, 2)


def test_as_float_array_converts_ints_and_keeps_float32():
    assert as_float_array([1, 2]).dtype == np.float64
    assert as_float_array(np.array([1], dtype=np.float32)).dtype == np.float32


def test_as_float_array_checks():
    with pytest.raises(ValueError, match="numeric"):
        as_float_array(["a"])
    with pytest.raises(ValueError, match="NaN"):
        as_float_array([np.nan], allow_nan=False)
    with pytest.raises(ValueError, match="infinity"):
        as_float_array([np.inf], allow_inf=False)
    with pytest.raises(ValueError, match="dimension"):
        as_float_array([1, 2], ndim=2)


# pairwise_distances

X = np.array([[0.0, 0.0], [3.0, 4.0]])
Y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])


def test_euclidean_known_values():
    # (3, 4) is 5 away from the origin. The fast formula leaves rounding
    # error of about 1e-8 for identical points, hence the atol.
    D = pairwise_distances(X, Y)
    np.testing.assert_allclose(D, [[0, 1, 2], [5, np.sqrt(20), np.sqrt(13)]], atol=1e-7)


def test_sqeuclidean_is_euclidean_squared():
    np.testing.assert_allclose(
        pairwise_distances(X, Y, metric="sqeuclidean"), pairwise_distances(X, Y) ** 2
    )


def test_manhattan_known_values_and_batches_agree():
    expected = [[0, 1, 2], [7, 6, 5]]
    np.testing.assert_allclose(pairwise_distances(X, Y, metric="manhattan"), expected)
    np.testing.assert_allclose(
        pairwise_distances(X, Y, metric="manhattan", batch_size=1), expected
    )


def test_cosine_known_values_and_zero_vector():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    D = pairwise_distances(A, metric="cosine")
    assert D[0, 1] == pytest.approx(1.0)  # perpendicular
    assert D[0, 2] == pytest.approx(2.0)  # opposite
    assert D[3, 0] == pytest.approx(1.0)  # zero vector
    assert np.all(np.diag(D) == 0)


def test_self_distances_symmetric_with_zero_diagonal():
    A = np.random.default_rng(0).normal(size=(6, 3))
    D = pairwise_distances(A)
    np.testing.assert_allclose(D, D.T)
    assert np.all(np.diag(D) == 0)


def test_euclidean_stays_accurate_far_from_origin():
    # naive |x|^2 + |y|^2 - 2x.y loses these digits without centring
    A = np.array([[1e8, 1e8 + 1], [1e8 + 1, 1e8]])
    assert pairwise_distances(A)[0, 1] == pytest.approx(np.sqrt(2))


def test_pairwise_non_contiguous_input():
    A = np.arange(12.0).reshape(3, 4)[:, ::2]  # strided view
    np.testing.assert_allclose(pairwise_distances(A), pairwise_distances(A.copy()))


def test_pairwise_empty_input():
    assert pairwise_distances(np.empty((0, 2)), Y).shape == (0, 3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"metric": "l3"}, "metric"),
        ({"Y": np.ones((2, 3))}, "same number of features"),
    ],
)
def test_pairwise_bad_arguments_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pairwise_distances(X, **kwargs)


def test_pairwise_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        pairwise_distances([[np.nan, 0.0]])

# activations

def test_relu_and_leaky_relu():
    np.testing.assert_allclose(relu([-2, 0, 3]), [0, 0, 3])
    np.testing.assert_allclose(leaky_relu([-10, 5], alpha=0.1), [-1, 5])
    with pytest.raises(ValueError, match="alpha"):
        leaky_relu([1], alpha=-1)


def test_sigmoid_known_values_and_no_overflow():
    np.testing.assert_allclose(sigmoid([0]), [0.5])
    out = sigmoid([-1000, 1000, -np.inf, np.inf])
    np.testing.assert_allclose(out, [0, 1, 0, 1])
    assert np.isfinite(out).all()


def test_softplus_known_values_and_no_overflow():
    np.testing.assert_allclose(softplus([0]), [np.log(2)])
    np.testing.assert_allclose(softplus([-1000, 1000]), [0, 1000])


def test_activations_keep_nan():
    for f in (relu, sigmoid, softplus):
        assert np.isnan(f([np.nan])).all()

# logsumexp / softmax / log_softmax

def test_logsumexp_large_values_do_not_overflow():
    assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000 + np.log(2))


def test_logsumexp_axis_and_keepdims():
    a = np.zeros((2, 3))
    np.testing.assert_allclose(logsumexp(a, axis=1), [np.log(3)] * 2)
    assert logsumexp(a, axis=1, keepdims=True).shape == (2, 1)
    assert logsumexp(a, axis=0).shape == (3,)


def test_logsumexp_edge_cases():
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp(np.array([])) == -np.inf
    assert logsumexp([1.0, np.inf]) == np.inf
    assert np.isnan(logsumexp([1.0, np.nan]))
    assert logsumexp([0.0, np.nan], nan_policy="omit") == 0.0


def test_softmax_rows_sum_to_one_with_large_values():
    p = softmax(np.array([[1000.0, 1001.0], [0.0, 0.0]]))
    np.testing.assert_allclose(p.sum(axis=1), [1, 1])
    np.testing.assert_allclose(p[1], [0.5, 0.5])


def test_softmax_unchanged_by_adding_a_constant():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(softmax(x), softmax(x + 500))


def test_softmax_along_axis_0():
    p = softmax(np.array([[1.0, 5.0], [1.0, 5.0]]), axis=0)
    np.testing.assert_allclose(p, 0.5)


def test_softmax_nan_policy():
    assert np.isnan(softmax([0.0, np.nan])).all()
    np.testing.assert_allclose(softmax([0.0, np.nan], nan_policy="omit"), [1, 0])


def test_softmax_undefined_slice_is_nan():
    assert np.isnan(softmax([-np.inf, -np.inf])).all()


def test_log_softmax_stays_finite():
    # softmax underflows to 0 here, but its log is still -1000
    np.testing.assert_allclose(log_softmax([0.0, -1000.0]), [0, -1000])


def test_log_softmax_matches_log_of_softmax():
    x = np.array([0.5, 1.5, -2.0])
    np.testing.assert_allclose(log_softmax(x), np.log(softmax(x)))


@pytest.mark.parametrize("f", [softmax, log_softmax])
def test_softmax_family_bad_inputs(f):
    with pytest.raises(ValueError, match="1-D"):
        f(1.0)
    with pytest.raises(ValueError, match="nan_policy"):
        f([1.0], nan_policy="ignore")

# topk_along_axis

def test_topk_largest_and_smallest():
    v, i = topk_along_axis([3, 1, 4, 1, 5], 2)
    np.testing.assert_array_equal(v, [5, 4])
    np.testing.assert_array_equal(i, [4, 2])
    v, i = topk_along_axis([3, 1, 4, 1, 5], 2, largest=False)
    np.testing.assert_array_equal(v, [1, 1])
    np.testing.assert_array_equal(i, [1, 3])  # tie: earlier position first


def test_topk_ties_choose_earliest_positions():
    v, i = topk_along_axis([7, 7, 7, 7], 2)
    np.testing.assert_array_equal(i, [0, 1])


def test_topk_k_zero_and_k_larger_than_length():
    v, i = topk_along_axis([3, 1, 2], 0)
    assert v.shape == (0,) and i.shape == (0,)
    v, i = topk_along_axis([3, 1, 2], 10)
    np.testing.assert_array_equal(v, [3, 2, 1])


def test_topk_nan_ranks_last():
    v, i = topk_along_axis([np.nan, 1.0, 3.0], 2)
    np.testing.assert_array_equal(v, [3, 1])
    v, i = topk_along_axis([np.nan, 1.0], 2)
    assert v[0] == 1 and np.isnan(v[1])


def test_topk_along_each_axis_of_2d():
    a = np.array([[1, 9, 3], [8, 2, 7]])
    v, _ = topk_along_axis(a, 1, axis=1)
    np.testing.assert_array_equal(v, [[9], [8]])
    v, i = topk_along_axis(a, 1, axis=0)
    np.testing.assert_array_equal(v, [[8, 9, 7]])
    np.testing.assert_array_equal(i, [[1, 0, 1]])


def test_topk_matches_full_sort():
    a = np.random.default_rng(1).normal(size=(4, 50))
    v, _ = topk_along_axis(a, 5)
    np.testing.assert_allclose(v, -np.sort(-a, axis=1)[:, :5])


def test_topk_non_contiguous_input():
    a = np.arange(20).reshape(4, 5)[:, ::2]  # strided view
    v, _ = topk_along_axis(a, 2)
    np.testing.assert_array_equal(v, [[4, 2], [9, 7], [14, 12], [19, 17]])


@pytest.mark.parametrize(
    "args, error",
    [
        (([1, 2], -1), ValueError),
        (([1, 2], 1.0), TypeError),
        ((["a", "b"], 1), ValueError),
        ((5, 1), ValueError),
    ],
)
def test_topk_bad_arguments_raise(args, error):
    with pytest.raises(error):
        topk_along_axis(*args)

# batching

def test_batch_slices_cover_everything():
    assert batch_slices(5, 2) == [slice(0, 2), slice(2, 4), slice(4, 5)]
    assert batch_slices(5, 2, drop_last=True) == [slice(0, 2), slice(2, 4)]
    assert batch_slices(0, 3) == []


def test_batch_slices_bad_arguments_raise():
    with pytest.raises(ValueError, match="batch_size"):
        batch_slices(5, 0)
    with pytest.raises(TypeError, match="batch_size"):
        batch_slices(5, 2.0)


def test_iter_batches_shapes_and_alignment():
    Xb, yb = np.arange(10).reshape(5, 2), np.arange(5)
    batches = list(iter_batches(Xb, yb, batch_size=2))
    assert [x.shape for x, _ in batches] == [(2, 2), (2, 2), (1, 2)]
    for x, y in batches:
        np.testing.assert_array_equal(x[:, 0], y * 2)  # rows still match


def test_iter_batches_shuffle_is_reproducible_and_complete():
    Xb, yb = np.arange(10).reshape(5, 2), np.arange(5)
    first = [
        y.tolist() for _, y in iter_batches(Xb, yb, batch_size=2, shuffle=True, seed=3)
    ]
    second = [
        y.tolist() for _, y in iter_batches(Xb, yb, batch_size=2, shuffle=True, seed=3)
    ]
    assert first == second
    assert sorted(sum(first, [])) == [0, 1, 2, 3, 4]
    for x, y in iter_batches(Xb, yb, batch_size=2, shuffle=True, seed=3):
        np.testing.assert_array_equal(x[:, 0], y * 2)


def test_iter_batches_bad_arguments_raise():
    with pytest.raises(ValueError, match="at least one"):
        next(iter_batches(batch_size=1))
    with pytest.raises(ValueError, match="same length"):
        next(iter_batches([1, 2], [1], batch_size=1))


# complex input and shared constants

def test_as_float_array_rejects_complex():
    with pytest.raises(ValueError, match="complex"):
        as_float_array([1 + 2j, 3])


def test_shared_constants():
    assert "na" in MISSING_TOKENS and "" in MISSING_TOKENS
    assert INTERPOLATIONS == ("linear", "lower", "higher", "midpoint")