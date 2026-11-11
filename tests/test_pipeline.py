# tests for pipeline.py
import numpy as np
import pytest

from numcompute_stream.pipeline import Estimator, FeatureUnion, Pipeline, Transformer

# stand-in steps

class Center(Transformer):
    """Subtracts the column means learned in fit. fit takes X only, like
    the project's preprocessors."""

    def fit(self, X):
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.mean_


class Double(Transformer):
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X) * 2


class FirstColumn:
    """Doesn't inherit from Transformer and returns a 1D array."""

    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X)[:, 0]


class MeanModel(Estimator):
    """Predicts the mean of y for every row."""

    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


X_TRAIN = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
Y_TRAIN = np.array([0.0, 1.0, 1.0])

# Pipeline

def test_steps_run_in_order():
    pipe = Pipeline([("center", Center()), ("double", Double())])
    out = pipe.fit_transform(X_TRAIN)
    # column means are 3 and 20, then everything is doubled
    np.testing.assert_allclose(out, [[-4, -20], [0, 0], [4, 20]])


def test_fit_then_transform_matches_fit_transform():
    pipe = Pipeline([("center", Center()), ("double", Double())])
    a = pipe.fit_transform(X_TRAIN)
    b = pipe.fit(X_TRAIN).transform(X_TRAIN)
    np.testing.assert_allclose(a, b)


def test_transform_uses_training_statistics():
    # new data must be centred with the TRAINING mean, not its own
    pipe = Pipeline([("center", Center())]).fit(X_TRAIN)
    out = pipe.transform(np.array([[3.0, 20.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.0]])


def test_predict_with_model_as_last_step():
    pipe = Pipeline([("center", Center()), ("model", MeanModel())])
    pipe.fit(X_TRAIN, Y_TRAIN)
    np.testing.assert_allclose(pipe.predict(X_TRAIN), [2 / 3] * 3)


def test_y_is_not_passed_to_preprocessors():
    # Center.fit only accepts X, so this would crash if y reached it
    pipe = Pipeline([("center", Center()), ("model", MeanModel())])
    pipe.fit(X_TRAIN, Y_TRAIN)


def test_single_step_pipeline():
    out = Pipeline([("double", Double())]).fit_transform(X_TRAIN)
    np.testing.assert_allclose(out, X_TRAIN * 2)


def test_step_without_base_class_works():
    out = Pipeline([("center", Center()), ("first", FirstColumn())]).fit_transform(
        X_TRAIN
    )
    np.testing.assert_allclose(out, [-2, 0, 2])


def test_lookup_by_name_and_position():
    center = Center()
    pipe = Pipeline([("center", center), ("double", Double())])
    assert pipe["center"] is center
    assert pipe[0] is center
    assert len(pipe) == 2
    assert list(pipe.named_steps) == ["center", "double"]


def test_unknown_step_name_raises():
    pipe = Pipeline([("center", Center())])
    with pytest.raises(KeyError, match="no step named"):
        pipe["scale"]


def test_nested_pipeline_as_a_step():
    inner = Pipeline([("center", Center())])
    outer = Pipeline([("inner", inner), ("double", Double())])
    np.testing.assert_allclose(
        outer.fit_transform(X_TRAIN), [[-4, -20], [0, 0], [4, 20]]
    )

# Pipeline errors

def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        Pipeline([("center", Center())]).transform(X_TRAIN)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        Pipeline([("model", MeanModel())]).predict(X_TRAIN)


def test_transform_with_model_last_raises():
    pipe = Pipeline([("model", MeanModel())]).fit(X_TRAIN, Y_TRAIN)
    with pytest.raises(TypeError, match="no transform"):
        pipe.transform(X_TRAIN)


def test_predict_with_transformer_last_raises():
    pipe = Pipeline([("center", Center())]).fit(X_TRAIN)
    with pytest.raises(TypeError, match="no predict"):
        pipe.predict(X_TRAIN)


@pytest.mark.parametrize(
    "steps, error, message",
    [
        ([], ValueError, "non-empty"),
        ([Center()], TypeError, "pairs"),
        ([("", Center())], TypeError, "non-empty strings"),
        ([("a", 5)], TypeError, "no fit"),
        ([("a", Center()), ("a", Double())], ValueError, "unique"),
        ([("model", MeanModel()), ("center", Center())], TypeError, "intermediate"),
    ],
)
def test_bad_steps_raise(steps, error, message):
    with pytest.raises(error, match=message):
        Pipeline(steps)

# FeatureUnion

def test_feature_union_stacks_columns():
    union = FeatureUnion([("center", Center()), ("double", Double())])
    out = union.fit_transform(X_TRAIN)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[:, 2:], X_TRAIN * 2)


def test_feature_union_turns_1d_output_into_a_column():
    union = FeatureUnion([("double", Double()), ("first", FirstColumn())])
    assert union.fit_transform(X_TRAIN).shape == (3, 3)


def test_feature_union_fit_then_transform_matches():
    union = FeatureUnion([("center", Center()), ("double", Double())])
    np.testing.assert_allclose(
        union.fit(X_TRAIN).transform(X_TRAIN), union.fit_transform(X_TRAIN)
    )


def test_feature_union_inside_pipeline():
    pipe = Pipeline(
        [
            ("features", FeatureUnion([("center", Center()), ("double", Double())])),
            ("model", MeanModel()),
        ]
    )
    pipe.fit(X_TRAIN, Y_TRAIN)
    assert pipe.predict(X_TRAIN).shape == (3,)


def test_feature_union_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        FeatureUnion([("center", Center())]).transform(X_TRAIN)


def test_feature_union_needs_transform_method():
    with pytest.raises(TypeError, match="no transform"):
        FeatureUnion([("model", MeanModel())])

# base classes

def test_base_classes_must_be_overridden():
    with pytest.raises(NotImplementedError):
        Transformer().fit(X_TRAIN)
    with pytest.raises(NotImplementedError):
        Transformer().transform(X_TRAIN)
    with pytest.raises(NotImplementedError):
        Estimator().predict(X_TRAIN)



# y handling and streaming

class RunningSum(Transformer):
    """Keeps a running column total over every chunk it has seen."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_ = 0.0
        return self

    def fit(self, X):
        return self.reset().partial_fit(X)

    def partial_fit(self, X):
        self.total_ = self.total_ + np.asarray(X, dtype=float).sum(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) + 1


def test_y_is_ignored_when_last_step_fit_takes_x_only():
    pipe = Pipeline([("center", Center())])
    out = pipe.fit(X_TRAIN, Y_TRAIN).transform(X_TRAIN)
    np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(pipe.fit_transform(X_TRAIN, Y_TRAIN), out)


def test_transformer_fit_transform_ignores_y_for_fit_x_only():
    np.testing.assert_allclose(Double().fit_transform(X_TRAIN, Y_TRAIN), X_TRAIN * 2)


def test_partial_fit_feeds_each_step_the_previous_output():
    pipe = Pipeline([("first", RunningSum()), ("second", RunningSum())])
    pipe.partial_fit(X_TRAIN[:2]).update(X_TRAIN[2:])
    np.testing.assert_allclose(pipe["first"].total_, X_TRAIN.sum(axis=0))
    np.testing.assert_allclose(pipe["second"].total_, X_TRAIN.sum(axis=0) + 3)
    np.testing.assert_allclose(pipe.transform(X_TRAIN), X_TRAIN + 2)


def test_partial_fit_without_method_raises_and_changes_nothing():
    pipe = Pipeline([("first", RunningSum()), ("center", Center())])
    with pytest.raises(TypeError, match="partial_fit"):
        pipe.partial_fit(X_TRAIN)
    assert pipe["first"].total_ == 0.0


def test_reset_resets_steps_and_fitted_flag():
    pipe = Pipeline([("first", RunningSum()), ("double", Double())])
    pipe.fit(X_TRAIN)
    pipe.reset()
    assert pipe["first"].total_ == 0.0
    with pytest.raises(RuntimeError):
        pipe.transform(X_TRAIN)