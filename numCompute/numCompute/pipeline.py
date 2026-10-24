"""
pipeline.py
Runs preprocessing steps one after another.
Each step has fit and transform.
FeatureUnion runs steps side by side and joins their outputs.
"""

import inspect

import numpy as np

__all__ = ["Transformer", "Estimator", "Pipeline", "FeatureUnion"]


class Transformer:
    """Base class for preprocessing steps.

    Subclasses need to write fit(X) and transform(X).
    Fit should learn whatever it needs from X (e.g. the mean) and return self.
    """

    def fit(self, X, y=None):
        """Learn from X and return self. Subclasses must override this."""
        raise NotImplementedError(f"{type(self).__name__} must implement fit")

    def transform(self, X):
        """Apply the step to X. Subclasses must override this."""
        raise NotImplementedError(f"{type(self).__name__} must implement transform")

    def fit_transform(self, X, y=None):
        """fit then transform on the same X.

        y is only passed on if it's given and fit accepts it, because the preprocessors in this project define fit(X) without a y.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,), optional

        Returns
        -------
        numpy.ndarray
            Whatever transform returns.

        Notes
        -----
        Same cost as calling fit and transform separately.
        """
        return _call_fit(self.fit, X, y).transform(X)


class Estimator:
    """Base class for models: fit(X, y) and predict(X).

    The assignment doesn't ask for an actual model, this is just the interface a Pipeline expects at the end.
    """

    def fit(self, X, y=None):
        """Train on X (and y) and return self. Subclasses must override this."""
        raise NotImplementedError(f"{type(self).__name__} must implement fit")

    def predict(self, X):
        """Return predictions for X. Subclasses must override this."""
        raise NotImplementedError(f"{type(self).__name__} must implement predict")


def _validate_steps(steps, owner):
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise ValueError(f"{owner} needs a non-empty list of (name, object) pairs")
    names = []
    for item in steps:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(f"{owner} steps must be (name, object) pairs, got {item!r}")
        name, step = item
        if not isinstance(name, str) or name == "":
            raise TypeError(f"step names must be non-empty strings, got {name!r}")
        if not hasattr(step, "fit"):
            raise TypeError(f"step {name!r} ({type(step).__name__}) has no fit method")
        names.append(name)
    # collect any step names that appear more than once
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"step names must be unique, duplicated: {duplicates}")
    return [tuple(item) for item in steps]


def _accepts_y(method):
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return True
    positional = 0
    for p in params:
        if p.kind == p.VAR_POSITIONAL or p.name == "y":
            return True
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            positional += 1
    return positional >= 2


def _call_fit(method, X, y):
    # only hand y over if the method can take it, so fit(X) steps don't crash
    if y is None or not _accepts_y(method):
        return method(X)
    return method(X, y)


def _fit_transform_step(step, X):
    if hasattr(step, "fit_transform"):
        return step.fit_transform(X)
    return step.fit(X).transform(X)


def _as_2d(values, name):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(
            f"step {name!r} must return a 1D or 2D array, got shape {values.shape}"
        )
    return values


class Pipeline(Transformer):
    """Run a list of steps one after another.

    Every step except the last needs fit and transform. The last one can be a transformer (then use transform/fit_transform) or a model (then use predict).

    Parameters
    ----------
    steps : list of (str, object)
        e.g. [("scale", StandardScaler()), ("encode", OneHotEncoder())]

    Raises
    ------
    ValueError, TypeError
        If steps is badly formed (see _validate_steps), or a step before the last one has no transform method.

    Notes
    -----
    Steps can be looked up by name or position: pipe["scale"] or pipe[0].
    The total cost is just the sum of the costs of the steps.
    Steps that have partial_fit can also be trained chunk by chunk with partial_fit.
    """

    def __init__(self, steps):
        self.steps = _validate_steps(steps, "Pipeline")
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"intermediate step {name!r} ({type(step).__name__}) "
                    "has no transform method"
                )
        self._is_fitted = False

    @property
    def named_steps(self):
        """Steps as a {name: object} dict."""
        return dict(self.steps)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.named_steps:
                raise KeyError(
                    f"no step named {key!r}; steps are {list(self.named_steps)}"
                )
            return self.named_steps[key]
        return self.steps[key][1]

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet; call fit first")

    def _fit_intermediate(self, X):
        # fit each step and feed its output into the next one
        for _, step in self.steps[:-1]:
            X = _fit_transform_step(step, X)
        return X

    def _transform_intermediate(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def reset(self):
        """Reset every step that has a reset method, and mark the Pipeline as not fitted.

        Returns
        -------
        self

        Notes
        -----
        O(number of steps), plus whatever each step's reset costs.
        """
        for _, step in self.steps:
            if hasattr(step, "reset"):
                step.reset()
        self._is_fitted = False
        return self

    def fit(self, X, y=None):
        """Fit every step in order.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,), optional
            Only given to the last step, and only if its fit accepts a y. The preprocessors don't take a y, so it's ignored for them.

        Returns
        -------
        self

        Notes
        -----
        Each step's fit_transform is called once, so the cost is the sum of the steps.
        """
        X = self._fit_intermediate(X)
        final = self.steps[-1][1]
        # y only goes to the last step; the preprocessors' fit takes X only
        _call_fit(final.fit, X, y)
        self._is_fitted = True
        return self

    def partial_fit(self, X, y=None):
        """Update every step with one more chunk of data.

        Each step learns from the chunk after it has gone through the steps before it, using what they have learned so far.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            One chunk of data.
        y : array-like of shape (n_samples,), optional
            Only given to the last step, as in fit.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            A step has no partial_fit method. Nothing is updated in that case.

        Notes
        -----
        Cost is the sum of each step's partial_fit and transform on the chunk. update is an alias.
        """
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise TypeError(
                    f"step {name!r} ({type(step).__name__}) has no partial_fit method"
                )
        for _, step in self.steps[:-1]:
            step.partial_fit(X)
            X = step.transform(X)
        _call_fit(self.steps[-1][1].partial_fit, X, y)
        self._is_fitted = True
        return self

    update = partial_fit

    def transform(self, X):
        """Push X through every step using what was learned in fit.

        Use this on test data, so it gets transformed with the training data's statistics.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            Output of the last step.

        Raises
        ------
        RuntimeError
            Called before fit.
        TypeError
            The last step has no transform (it's a model, use predict).

        Notes
        -----
        Cost is the sum of each step's transform.
        """
        self._check_fitted()
        name, final = self.steps[-1]
        if not hasattr(final, "transform"):
            raise TypeError(
                f"final step {name!r} ({type(final).__name__}) has no transform method"
            )
        return final.transform(self._transform_intermediate(X))

    def fit_transform(self, X, y=None):
        """fit and transform in one pass over the steps.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,), optional
            Only given to the last step, and only if its fit accepts a y.

        Returns
        -------
        numpy.ndarray
            Output of the last step.

        Raises
        ------
        TypeError
            The last step has no transform method.

        Notes
        -----
        Same cost as fit followed by transform, but the data only goes through the steps once.
        """
        name, final = self.steps[-1]
        # check this before fitting anything, so we never end up half-fitted
        if not hasattr(final, "transform"):
            raise TypeError(
                f"final step {name!r} ({type(final).__name__}) has no transform method"
            )
        X = self._fit_intermediate(X)
        if y is None:
            X = _fit_transform_step(final, X)
        else:
            X = _call_fit(final.fit, X, y).transform(X)
        self._is_fitted = True
        return X

    def predict(self, X):
        """Transform X with every step except the last, then predict with it.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Whatever the model's predict returns.

        Raises
        ------
        RuntimeError
            Called before fit.
        TypeError
            The last step has no predict method.

        Notes
        -----
        Cost is the sum of the transforms plus the model's predict.
        """
        self._check_fitted()
        name, final = self.steps[-1]
        if not hasattr(final, "predict"):
            raise TypeError(
                f"final step {name!r} ({type(final).__name__}) has no predict method"
            )
        return final.predict(self._transform_intermediate(X))


class FeatureUnion(Transformer):
    """Run several transformers on the same X and join the results column-wise.

    Useful for mixed data, e.g. scaled numeric columns next to one-hot encoded categories.

    Parameters
    ----------
    transformer_list : list of (str, object)
        Every object needs fit and transform.

    Raises
    ------
    ValueError, TypeError
        Badly formed list, or a step with no transform method.

    Notes
    -----
    Output shape is (n_samples, total columns from all transformers).
    Cost is the sum of the transformers plus one np.hstack.
    """

    def __init__(self, transformer_list):
        self.transformer_list = _validate_steps(transformer_list, "FeatureUnion")
        for name, step in self.transformer_list:
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"step {name!r} ({type(step).__name__}) has no transform method"
                )
        self._is_fitted = False

    def _combine(self, outputs):
        # every output needs the same number of rows before we can stack them
        blocks = [_as_2d(values, name) for name, values in outputs]
        n_rows = {block.shape[0] for block in blocks}
        if len(n_rows) != 1:
            row_counts = {
                name: block.shape[0] for (name, _), block in zip(outputs, blocks)
            }
            raise ValueError(
                f"all transformers must return the same number of rows, got {row_counts}"
            )
        return np.hstack(blocks)

    def fit(self, X, y=None):
        """Fit each transformer on X. y is ignored.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: ignored

        Returns
        -------
        self

        Notes
        -----
        Cost is the sum of each transformer's fit.
        """
        for _, step in self.transformer_list:
            step.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform X with each transformer and stack the outputs side by side.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray of shape (n_samples, total columns)

        Raises
        ------
        RuntimeError
            Called before fit.
        ValueError
            The transformers returned different numbers of rows.

        Notes
        -----
        Cost is the sum of each transformer's transform, plus O(n_samples * total columns) for the hstack.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureUnion is not fitted yet; call fit first")
        return self._combine(
            [(name, step.transform(X)) for name, step in self.transformer_list]
        )

    def fit_transform(self, X, y=None):
        """fit and transform each transformer, then stack the outputs.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: ignored

        Returns
        -------
        numpy.ndarray of shape (n_samples, total columns)

        Raises
        ------
        ValueError
            The transformers returned different numbers of rows.

        Notes
        -----
        Same cost as fit followed by transform.
        """
        outputs = [
            (name, _fit_transform_step(step, X)) for name, step in self.transformer_list
        ]
        self._is_fitted = True
        return self._combine(outputs)