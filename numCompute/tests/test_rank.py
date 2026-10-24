import numpy as np

from numCompute.rank import rank, percentile_rank, percentile

X = [10, 20, 20, 30]


def test_average():
    assert np.allclose(rank(X), [1, 2.5, 2.5, 4])


def test_min():
    assert (rank(X, "min") == [1, 2, 2, 4]).all()


def test_max():
    assert (rank(X, "max") == [1, 3, 3, 4]).all()


def test_dense():
    assert (rank(X, "dense") == [1, 2, 2, 3]).all()


def test_ordinal():
    assert (rank(X, "ordinal") == [1, 2, 3, 4]).all()


def test_unsorted_input():
    assert np.allclose(rank([30, 10, 20]), [3, 1, 2])


def test_descending():
    assert np.allclose(rank(X, descending=True), [4, 2.5, 2.5, 1])
    assert (rank(X, "min", descending=True) == [4, 2, 2, 1]).all()
    assert (rank(X, "max", descending=True) == [4, 3, 3, 1]).all()
    assert (rank(X, "dense", descending=True) == [3, 2, 2, 1]).all()


def test_ordinal_descending_ties_by_position():
    assert (rank(X, "ordinal", descending=True) == [4, 2, 3, 1]).all()


def test_all_equal():
    assert np.allclose(rank([5, 5, 5]), [2, 2, 2])
    assert (rank([5, 5, 5], "dense") == [1, 1, 1]).all()


def test_single_value():
    assert np.allclose(rank([7]), [1])


def test_empty():
    assert rank([]).shape == (0,)


def test_2d_axis():
    m = np.array([[3, 1, 2],
                  [9, 8, 7]])
    assert (rank(m, axis=1) == [[3, 1, 2], [3, 2, 1]]).all()
    assert (rank(m, axis=0) == [[1, 1, 1], [2, 2, 2]]).all()


def test_shape_preserved():
    m = np.random.default_rng(0).integers(0, 5, (3, 4, 5))
    for ax in (0, 1, 2, -1):
        assert rank(m, axis=ax).shape == m.shape


def test_matches_sort_order():
    x = np.random.default_rng(1).random(50)
    assert (np.argsort(rank(x)) == np.argsort(x)).all()


def test_percentile_rank():
    assert np.allclose(percentile_rank(X), [0, 0.5, 0.5, 1])
    assert np.allclose(percentile_rank(X, descending=True), [1, 0.5, 0.5, 0])
    assert np.allclose(percentile_rank([4]), [0])


def test_errors():
    for bad in (lambda: rank([1, np.nan, 3]),
                lambda: rank(X, method="nope"),
                lambda: rank(5),
                lambda: rank(X, axis=3)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_percentile_matches_numpy():
    x = np.random.default_rng(2).random((4, 9))
    for interp in ("linear", "lower", "higher", "midpoint"):
        for q in (0, 10, 33.3, 50, 99, 100):
            got = percentile(x, q, interp, axis=1)
            assert np.allclose(got, np.percentile(x, q, axis=1, method=interp))


def test_percentile_multiple_q_and_axis():
    x = np.random.default_rng(3).random((5, 6))
    got = percentile(x, [25, 50, 75], axis=0)
    assert got.shape == (3, 6)
    assert np.allclose(got, np.percentile(x, [25, 50, 75], axis=0))


def test_percentile_ignores_nan():
    x = np.array([1.0, np.nan, 3.0, 2.0, np.nan])
    assert np.isclose(percentile(x, 50), 2.0)
    assert np.isclose(percentile(x, 100), 3.0)


def test_percentile_all_nan_and_empty():
    assert np.isnan(percentile([np.nan, np.nan], 50))
    assert np.isnan(percentile([], 50))


def test_percentile_errors():
    for bad in (lambda: percentile(X, 101),
                lambda: percentile(X, -1),
                lambda: percentile(X, 50, "nearest"),
                lambda: percentile([1, np.inf], 50),
                lambda: percentile(X, 50, axis=2)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_percentile_2d_different_nan_counts_per_row():
    x = np.array([[1.0, np.nan, 3.0, 5.0],
                  [2.0, 4.0, np.nan, np.nan],
                  [np.nan, np.nan, np.nan, np.nan]])
    got = percentile(x, 50, axis=1)
    assert np.allclose(got[:2], [3.0, 3.0]) and np.isnan(got[2])


def test_percentile_integer_input():
    assert np.isclose(percentile([1, 2, 3, 4, 5], 50), 3.0)


def test_rank_integer_methods_return_integers():
    for m in ("min", "max", "dense", "ordinal"):
        assert rank([3, 1, 2, 2], m).dtype.kind in "iu"
    assert rank([3, 1, 2, 2]).dtype.kind == "f"


def test_rank_text_values():
    assert (rank(["b", "a", "c"], "ordinal") == [2, 1, 3]).all()


def test_rank_does_not_change_input():
    x = np.array([3, 1, 2])
    rank(x)
    assert (x == [3, 1, 2]).all()

def test_percentile_agrees_with_stats_quantile():
    from numCompute.stats import quantile
    x = np.arange(26.0)
    assert percentile(x, 28, "higher") == 7.0                # 0.28 * 25 is exactly 7
    for n in (2, 26, 51):
        a = np.arange(float(n))
        for q in range(0, 101, 7):
            for interp in ("linear", "lower", "higher", "midpoint"):
                assert percentile(a, q, interp) == quantile(a, q / 100, interpolation=interp)

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\nAll {len(tests)} tests passed.")
