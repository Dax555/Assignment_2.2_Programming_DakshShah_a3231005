"""Unit tests for numcompute.stats."""
import numpy as np
import pytest

from numCompute import stats


def test_count_ignores_nan_and_supports_axis():
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    assert stats.count(a) == 3
    np.testing.assert_array_equal(stats.count(a, axis=0), [2, 1])


def test_mean_matches_numpy_and_ignores_nan():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 3))
    np.testing.assert_allclose(stats.mean(a, axis=0), a.mean(axis=0))
    assert stats.mean([1.0, np.nan, 3.0]) == 2.0


def test_mean_empty_and_all_nan_give_nan():
    assert np.isnan(stats.mean([]))
    assert np.isnan(stats.mean([np.nan, np.nan]))


def test_var_std_match_numpy_for_both_ddof():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40, 3))
    for ddof in (0, 1):
        np.testing.assert_allclose(stats.var(a, axis=0, ddof=ddof), a.var(axis=0, ddof=ddof))
        np.testing.assert_allclose(stats.std(a, axis=0, ddof=ddof), a.std(axis=0, ddof=ddof))


def test_var_edge_cases_all_equal_single_value_and_large_offset():
    assert stats.var([7.0] * 10) == 0.0                      # all-equal
    assert np.isnan(stats.var([5.0], ddof=1))                # n - ddof <= 0
    assert stats.var(1e9 + np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.25)  # stable


def test_min_max_with_nan_empty_and_axis():
    a = np.array([[1.0, 9.0], [4.0, np.nan]])
    assert stats.min([3.0, np.nan, -1.0]) == -1.0
    assert stats.max([3.0, np.nan, 8.0]) == 8.0
    np.testing.assert_array_equal(stats.max(a, axis=1), [9.0, 4.0])
    assert np.isnan(stats.min([])) and np.isnan(stats.max([np.nan]))


def test_quantile_matches_numpy_all_interpolations():
    rng = np.random.default_rng(2)
    a = rng.normal(size=101)
    for q in (0.0, 0.25, 0.9, 1.0):
        assert stats.quantile(a, q) == pytest.approx(np.quantile(a, q))
    b = np.array([1.0, 2.0, 3.0, 4.0])
    for method in ("lower", "higher", "midpoint"):
        assert stats.quantile(b, 0.5, interpolation=method) == pytest.approx(
            np.quantile(b, 0.5, method=method))


def test_quantile_nan_single_value_non_contiguous_and_sequence_q():
    assert stats.quantile([1.0, np.nan, 3.0], 0.5) == 2.0
    assert stats.quantile([4.0], 0.3) == 4.0
    assert np.isnan(stats.quantile([np.nan, np.nan], 0.5))
    a = np.arange(40, dtype=float).reshape(8, 5)[::2, ::2]   # strided view
    assert not a.flags["C_CONTIGUOUS"]
    out = stats.quantile(a, [0.25, 0.75], axis=0)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.quantile(a, [0.25, 0.75], axis=0))


def test_median_even_and_odd_length():
    assert stats.median([1.0, 3.0, 2.0]) == 2.0
    assert stats.median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_mode_ties_strings_nan_and_empty():
    assert stats.mode([1, 2, 2, 3, 3]) == (2, 2)             # tie -> smallest
    assert stats.mode(["a", "b", "b"]) == ("b", 2)
    assert stats.mode([np.nan, 2.0, np.nan]) == (2.0, 1)
    with pytest.raises(ValueError):
        stats.mode([])


def test_histogram_matches_numpy_and_density_integrates_to_one():
    rng = np.random.default_rng(3)
    a = rng.normal(size=500)
    c, e = stats.histogram(a, bins=12)
    c_np, e_np = np.histogram(a, bins=12)
    np.testing.assert_array_equal(c, c_np)
    np.testing.assert_allclose(e, e_np)
    d, e = stats.histogram(a, bins=12, density=True)
    assert np.sum(d * np.diff(e)) == pytest.approx(1.0)


def test_histogram_edge_cases():
    c, _ = stats.histogram([1, 2, 3], bins=[1, 2, 3])        # right edge in last bin
    np.testing.assert_array_equal(c, [1, 2])
    c, _ = stats.histogram([np.nan, -5.0, 0.5, 99.0], bins=2, range=(0, 2))
    assert c.sum() == 1                                      # NaN / out-of-range ignored
    c, e = stats.histogram([2.0, 2.0, 2.0], bins=3)          # all-equal
    assert c.sum() == 3
    c, e = stats.histogram([], bins=4)                       # empty
    assert c.sum() == 0 and len(e) == 5


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        stats.mean([1.0, np.inf])                            # infinity
    with pytest.raises(ValueError):
        stats.mean(["a", "b"])                               # non-numeric
    with pytest.raises(ValueError):
        stats.mean([1.0, 2.0], axis=1)                       # axis out of range
    with pytest.raises(TypeError):
        stats.var([1.0, 2.0], ddof=1.5)
    with pytest.raises(ValueError):
        stats.quantile([1.0, 2.0], 1.5)
    with pytest.raises(ValueError):
        stats.histogram([1.0], bins=0)


def test_describe_and_describe_categorical():
    a = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    d = stats.describe(a, names=["a", "b"])
    np.testing.assert_array_equal(d["count"], [3, 2])
    np.testing.assert_array_equal(d["missing"], [0, 1])
    np.testing.assert_allclose(d["50%"], [3.0, 4.0])
    cat = stats.describe_categorical(["x", "y", "x", "", "NA"])
    assert cat == {"count": 3, "missing": 2, "unique": 2, "top": "x", "freq": 2}


def test_format_summary_and_format_histogram():
    text = stats.format_summary(stats.describe(np.array([[1.0, 2.0], [3.0, 4.0]]), names=["a", "b"]))
    assert text.splitlines()[0].split() == ["stat", "a", "b"]
    c, e = stats.histogram([1, 2, 2, 3, 3, 3], bins=3)
    bars = stats.format_histogram(c, e, width=10).splitlines()
    assert len(bars) == 3 and bars[-1].count("#") == 10
    with pytest.raises(ValueError):
        stats.format_histogram(c, e[:-1])




def test_mean_does_not_overflow_and_stays_exact():
    assert stats.mean([1e308, 1e308]) == 1e308
    a = np.random.default_rng(3).normal(5, 2, 1000)
    assert stats.mean(a) == pytest.approx(a.mean(), rel=1e-15)


def test_three_dimensional_input_reduces_along_axis():
    a = np.random.default_rng(5).normal(size=(2, 3, 4))
    np.testing.assert_allclose(stats.mean(a, axis=1), a.mean(axis=1))
    np.testing.assert_allclose(stats.var(a, axis=2, ddof=1), a.var(axis=2, ddof=1))
    np.testing.assert_allclose(stats.quantile(a, 0.3, axis=0), np.quantile(a, 0.3, axis=0))
    with pytest.raises(ValueError):
        stats.describe(a)                                    # describe is 1-D/2-D only


def test_min_and_max_are_not_exported_by_star_import():
    assert "min" not in stats.__all__ and "max" not in stats.__all__
    assert stats.min([3.0, 1.0]) == 1.0                      # still callable as stats.min


def test_running_stats_matches_batch_results():
    rng = np.random.default_rng(6)
    X = rng.normal(1e6, 3.0, (500, 3))                       # large mean, small spread
    X[rng.random(X.shape) < 0.1] = np.nan
    rs = stats.RunningStats()
    for chunk in np.array_split(X, 7):
        rs.partial_fit(chunk)
    rs.update(np.empty((0, 3)))                              # an empty chunk changes nothing
    np.testing.assert_array_equal(rs.count(), stats.count(X, axis=0))
    np.testing.assert_allclose(rs.mean(), stats.mean(X, axis=0), rtol=1e-12)
    np.testing.assert_allclose(rs.var(ddof=1), stats.var(X, axis=0, ddof=1), rtol=1e-9)
    np.testing.assert_allclose(rs.std(), stats.std(X, axis=0), rtol=1e-9)
    np.testing.assert_array_equal(rs.min(), stats.min(X, axis=0))
    np.testing.assert_array_equal(rs.max(), stats.max(X, axis=0))


def test_running_stats_one_dimensional_and_errors():
    rs = stats.RunningStats()
    with pytest.raises(RuntimeError):
        rs.mean()
    rs.update([1.0, 2.0]).update([np.nan, 6.0])
    assert rs.count() == 3 and rs.mean() == 3.0
    assert np.isnan(rs.var(ddof=3))                          # n - ddof <= 0
    with pytest.raises(ValueError):
        rs.update([[1.0, 2.0]])                              # shape changed
    with pytest.raises(ValueError):
        rs.update([np.inf])
    assert np.isnan(rs.reset().update([np.nan]).mean())