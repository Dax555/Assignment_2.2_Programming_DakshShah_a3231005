import numpy as np

from numCompute.preprocessing import (StandardScaler, MinMaxScaler, Imputer,
                                      OneHotEncoder, ColumnTransformer)


def make_data(seed=0, shape=(200, 3), nan_frac=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(5, 3, shape)
    X[rng.random(shape) < nan_frac] = np.nan
    return X


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception as e:
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


# ---------------------------------------------------------------- StandardScaler
def test_standard_basic():
    out = StandardScaler().fit_transform([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(out, [[-1, -1], [1, 1]])


def test_standard_matches_numpy():
    X = make_data()
    s = StandardScaler().fit(X)
    assert np.allclose(s.mean_, np.nanmean(X, axis=0))
    assert np.allclose(s.var_, np.nanvar(X, axis=0))


def test_standard_streaming_equals_batch():
    X = make_data(1)
    batch = StandardScaler().fit(X)
    stream = StandardScaler()
    for chunk in np.array_split(X, 7):
        stream.partial_fit(chunk)
    assert np.allclose(batch.mean_, stream.mean_)
    assert np.allclose(batch.var_, stream.var_)
    assert np.allclose(batch.transform(X), stream.transform(X), equal_nan=True)


def test_standard_update_is_alias():
    a = StandardScaler().update([[1.0], [3.0]])
    assert np.allclose(a.mean_, [2.0])


def test_standard_nan_passes_through():
    out = StandardScaler().fit([[1.0], [3.0]]).transform([[np.nan], [2.0]])
    assert np.isnan(out[0, 0]) and out[1, 0] == 0.0


def test_standard_zero_variance():
    out = StandardScaler().fit_transform([[1.0, 2.0], [1.0, 3.0]])
    assert np.allclose(out[:, 0], 0.0) and np.isfinite(out).all()


def test_standard_empty_chunk_changes_nothing():
    s = StandardScaler().partial_fit([[1.0, 2.0], [3.0, 4.0]])
    mean_before = s.mean_.copy()
    s.partial_fit(np.empty((0, 2)))
    assert np.allclose(s.mean_, mean_before)


def test_standard_flags():
    X = [[1.0], [3.0]]
    assert np.allclose(StandardScaler(with_std=False).fit_transform(X), [[-1], [1]])
    assert np.allclose(StandardScaler(with_mean=False).fit_transform(X)[:, 0],
                       np.array([1.0, 3.0]) / 1.0)


def test_standard_large_mean_is_stable():
    rng = np.random.default_rng(0)
    Z = 1e9 + rng.normal(0, 1, (1000, 1))
    s = StandardScaler()
    for chunk in np.array_split(Z, 10):
        s.partial_fit(chunk)
    assert np.isclose(s.var_[0], Z.var(), rtol=1e-6)


def test_standard_errors():
    assert raises(RuntimeError, lambda: StandardScaler().transform([[1.0]]))
    assert raises(ValueError, lambda: StandardScaler().fit([[1.0, np.inf]]))
    assert raises(ValueError, lambda: StandardScaler().fit([[1.0, 2.0]]).partial_fit([[1.0]]))
    assert raises(ValueError, lambda: StandardScaler().fit([1.0, 2.0]))
    assert raises(ValueError, lambda: StandardScaler().fit([["a"], ["b"]]))


def test_standard_fit_forgets_and_reset():
    s = StandardScaler().fit([[100.0], [200.0]])
    s.fit([[1.0], [3.0]])
    assert np.allclose(s.mean_, [2.0])
    s.reset()
    assert raises(RuntimeError, lambda: s.transform([[1.0]]))


# ---------------------------------------------------------------- MinMaxScaler
def test_minmax_basic():
    out = MinMaxScaler().fit_transform([[1.0], [2.0], [3.0]])
    assert np.allclose(out[:, 0], [0, 0.5, 1])


def test_minmax_custom_range():
    out = MinMaxScaler(feature_range=(-1, 1)).fit_transform([[0.0], [10.0]])
    assert np.allclose(out[:, 0], [-1, 1])


def test_minmax_streaming_equals_batch():
    X = make_data(2)
    batch = MinMaxScaler().fit(X)
    stream = MinMaxScaler()
    for chunk in np.array_split(X, 6):
        stream.partial_fit(chunk)
    assert np.allclose(batch.min_, stream.min_) and np.allclose(batch.max_, stream.max_)


def test_minmax_out_of_range_and_clip():
    s = MinMaxScaler().fit([[0.0], [10.0]])
    assert np.allclose(s.transform([[20.0]]), [[2.0]])
    assert np.allclose(MinMaxScaler(clip=True).fit([[0.0], [10.0]]).transform([[20.0]]), [[1.0]])


def test_minmax_constant_feature_maps_to_lower_bound():
    out = MinMaxScaler().fit_transform([[5.0], [5.0]])
    assert np.allclose(out, 0.0)


def test_minmax_feature_never_seen_gives_nan():
    s = MinMaxScaler().fit([[np.nan], [np.nan]])
    assert np.isnan(s.transform([[1.0]])).all()


def test_minmax_empty_chunk_and_errors():
    s = MinMaxScaler().partial_fit(np.empty((0, 2)))
    assert np.isinf(s.min_).all()
    assert raises(ValueError, lambda: MinMaxScaler(feature_range=(1, 1)))
    assert raises(RuntimeError, lambda: MinMaxScaler().transform([[1.0]]))
    assert raises(ValueError, lambda: MinMaxScaler().fit([[np.inf]]))


# ---------------------------------------------------------------- Imputer
def test_imputer_mean():
    out = Imputer("mean").fit_transform([[1.0, np.nan], [3.0, 4.0]])
    assert np.allclose(out, [[1, 4], [3, 4]])


def test_imputer_mean_streaming():
    imp = Imputer("mean")
    imp.partial_fit([[1.0], [np.nan]])
    imp.partial_fit([[5.0]])
    assert np.allclose(imp.statistics_, [3.0])
    assert np.allclose(imp.transform([[np.nan]]), [[3.0]])


def test_imputer_constant():
    out = Imputer("constant", fill_value=-1).fit_transform([[np.nan], [2.0]])
    assert np.allclose(out, [[-1], [2]])


def test_imputer_most_frequent_text():
    imp = Imputer("most_frequent").fit([["a"], ["a"], [""], ["b"]])
    assert imp.transform([[""], ["b"]]).tolist() == [["a"], ["b"]]


def test_imputer_most_frequent_tie_picks_smallest():
    imp = Imputer("most_frequent").fit([["b"], ["a"]])
    assert imp.statistics_ == ["a"]


def test_imputer_most_frequent_streaming():
    imp = Imputer("most_frequent")
    imp.partial_fit([["a"], ["a"], [""]])
    imp.partial_fit([["b"], ["b"], ["b"]])
    assert imp.statistics_ == ["b"]


def test_imputer_unseen_column_uses_fill_value():
    imp = Imputer("mean", fill_value=7.0).fit([[np.nan]])
    assert imp.statistics_ == [7.0]


def test_imputer_errors():
    assert raises(ValueError, lambda: Imputer("median"))
    assert raises(RuntimeError, lambda: Imputer().transform([[1.0]]))


# ---------------------------------------------------------------- OneHotEncoder
def test_onehot_basic():
    enc = OneHotEncoder()
    out = enc.fit_transform([["red"], ["blue"], ["red"], ["green"]])
    assert enc.categories_[0].tolist() == ["blue", "green", "red"]
    assert out.tolist() == [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]


def test_onehot_missing_is_all_zeros():
    enc = OneHotEncoder().fit([["red"], [""], ["nan"], ["green"]])
    assert enc.categories_[0].tolist() == ["green", "red"]
    assert enc.transform([[""]]).tolist() == [[0, 0]]


def test_onehot_streaming_new_categories_append():
    enc = OneHotEncoder()
    enc.partial_fit([["r"], ["b"]])
    first = enc.categories_[0].tolist()
    enc.partial_fit([["g"]])
    assert enc.categories_[0].tolist() == first + ["g"]
    assert enc.n_output_features_ == 3


def test_onehot_freeze():
    enc = OneHotEncoder().fit([["a"], ["b"]]).freeze()
    enc.partial_fit([["c"]])
    assert enc.n_output_features_ == 2
    assert enc.transform([["c"]]).tolist() == [[0, 0]]


def test_onehot_fixed_categories_and_reset():
    enc = OneHotEncoder(categories=[["a", "b"]])
    assert enc.fit_transform([["b"], ["z"]]).tolist() == [[0, 1], [0, 0]]
    enc.reset()
    assert enc.n_output_features_ == 2


def test_onehot_unknown_error():
    enc = OneHotEncoder(handle_unknown="error").fit([["a"]])
    assert raises(ValueError, lambda: enc.transform([["b"]]))
    assert enc.transform([[""]]).tolist() == [[0]]


def test_onehot_multiple_columns_and_errors():
    enc = OneHotEncoder().fit([["a", "x"], ["b", "y"]])
    assert enc.transform([["a", "y"]]).tolist() == [[1, 0, 0, 1]]
    assert raises(ValueError, lambda: enc.transform([["a"]]))
    assert raises(ValueError, lambda: OneHotEncoder(handle_unknown="drop"))
    assert raises(RuntimeError, lambda: OneHotEncoder().transform([["a"]]))


# ---------------------------------------------------------------- ColumnTransformer
def test_column_transformer_basic():
    D = np.array([[1.0, "x"], [np.nan, "y"], [3.0, ""]], dtype=object)
    ct = ColumnTransformer([Imputer("mean"), StandardScaler()],
                           [Imputer("most_frequent"), OneHotEncoder()], [0], [1])
    out = ct.fit_transform(D)
    assert out.shape == (3, 3)
    assert np.allclose(out[:, 0], [-1.2247449, 0.0, 1.2247449])
    assert np.isfinite(out).all()


def test_column_transformer_streaming():
    D = np.array([[1.0, "x"], [2.0, "y"], [3.0, "x"], [4.0, "z"]], dtype=object)
    ct = ColumnTransformer(StandardScaler(), OneHotEncoder(), [0], [1])
    ct.partial_fit(D[:2])
    ct.partial_fit(D[2:])
    assert ct.transform(D).shape == (4, 4)


def test_column_transformer_errors():
    assert raises(ValueError, lambda: ColumnTransformer(StandardScaler(), OneHotEncoder(), [0], [0]))
    assert raises(ValueError, lambda: ColumnTransformer(None, None, [0], []))
    ct = ColumnTransformer(StandardScaler(), None, [5], [])
    assert raises(ValueError, lambda: ct.fit(np.zeros((2, 2), dtype=object)))
    assert raises(RuntimeError, lambda: ColumnTransformer(StandardScaler(), None, [0], []).transform([[1.0]]))


# ---------------------------------------------------------------- Pipeline use
def test_transformers_work_inside_pipeline_given_y():
    from numCompute.pipeline import Pipeline, Transformer
    X = make_data(seed=2, nan_frac=0.0)
    y = np.arange(X.shape[0])
    for t in (StandardScaler(), MinMaxScaler(), Imputer(), OneHotEncoder(), ColumnTransformer(StandardScaler(), None, [0], [])):
        assert isinstance(t, Transformer)
    pipe = Pipeline([("impute", Imputer()), ("scale", StandardScaler())])
    assert np.allclose(pipe.fit(X, y).transform(X), StandardScaler().fit_transform(X))


def test_imputer_most_frequent_streaming_matches_batch():
    chunks = [np.array([["b"], ["a"]], dtype=object), np.array([["b"], [""]], dtype=object),
              np.array([["a"], ["a"]], dtype=object)]
    streamed = Imputer("most_frequent")
    for c in chunks:
        streamed.update(c)
    batch = Imputer("most_frequent").fit(np.vstack(chunks))
    assert streamed.statistics_ == batch.statistics_ == ["a"]
    X = np.array([[1.0, 5.0], [1.0, np.nan], [2.0, 5.0], [2.0, 7.0]])
    assert Imputer("most_frequent").fit(X).statistics_ == [1.0, 5.0]   # tie goes to the smaller value


def test_onehot_unknown_error_names_the_right_column():
    enc = OneHotEncoder(handle_unknown="error").fit([["r", "x"], ["g", "y"]])
    assert np.array_equal(enc.transform([["g", "x"], ["", "y"]]), [[1, 0, 1, 0], [0, 0, 0, 1]])
    try:
        enc.transform([["g", "z"]])
    except ValueError as e:
        assert "column 1" in str(e) and "'z'" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\nAll {len(tests)} tests passed.")
