"""
stream.py
=========
``StreamTrainer`` runs a model or a whole ``Pipeline`` over a stream of
chunks and keeps the log: per-chunk metrics, cumulative accuracy, timing and
memory footprint.

The usual loop is *test-then-train* (prequential evaluation): each new chunk
is first used to score the current model, because the model has never seen
it, and only then used to train::

    trainer = StreamTrainer(pipe)
    for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=50):
        trainer.score_chunk(X_chunk, y_chunk)   # test on unseen data
        trainer.fit_chunk(X_chunk, y_chunk)     # then learn from it

``trainer.run(stream)`` does exactly that for every chunk of a stream.

Conventions
-----------
- A chunk is ``(X_chunk, y_chunk)`` with X of shape (n_chunk, n_features) and
  y of shape (n_chunk,).
- The model needs ``partial_fit(X, y)`` and ``predict(X)``. It may be a bare
  estimator or a Pipeline whose last step is one.
- A buffered model has no tree until its first buffer fills. Scoring during
  that warm-up is not an error: the chunk is recorded as "not scored" and
  its metric values are NaN.
- One log record is written per ``fit_chunk`` call and it carries the scores
  of the ``score_chunk`` call that preceded it.

"""

import csv
import logging
import time

import numpy as np

from .metrics import StreamingAccuracy
from .utils import NotFittedError, iter_batches

__all__ = ["StreamTrainer", "iter_chunks", "memory_footprint"]

logger = logging.getLogger(__name__)

_LOG_FIELDS = (
    "chunk", "n_samples", "samples_seen", "scored", "chunk_accuracy",
    "cumulative_accuracy", "rolling_accuracy", "error", "fit_seconds",
    "predict_seconds", "retrained", "n_retrains", "buffer_count", "memory_bytes",
)


def iter_chunks(X, y, chunk_size, shuffle=False, seed=None):
    """Split a dataset into consecutive chunks to simulate a stream.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Features.
    y : array_like, shape (n_samples,)
        Labels, aligned with the rows of X.
    chunk_size : int
        Samples per chunk (>= 1). The last chunk may be smaller.
    shuffle : bool, default False
        Visit the samples in a random order (the same for X and y). Use this
        when the file is sorted by class, otherwise the first chunks would
        contain a single class.
    seed : int, optional
        Seed for the shuffle.

    Yields
    ------
    (X_chunk, y_chunk) : tuple of np.ndarray
        Shapes (n_chunk, n_features) and (n_chunk,).

    Raises
    ------
    ValueError, TypeError
        Different lengths, or a bad `chunk_size` / `shuffle` (see
        ``utils.iter_batches``, which does the work).

    Notes
    -----
    Time: O(n) in total. Space: O(chunk) per chunk (plus O(n) when shuffling).

    Examples
    --------
    >>> import numpy as np
    >>> X, y = np.arange(10).reshape(5, 2), np.arange(5)
    >>> [len(yc) for _, yc in iter_chunks(X, y, chunk_size=2)]
    [2, 2, 1]
    """
    return iter_batches(X, y, batch_size=chunk_size, shuffle=shuffle, seed=seed)


def memory_footprint(obj):
    """Bytes held in NumPy arrays reachable from an object.

    Walks the object's attributes and any lists, tuples, sets and dicts inside
    them, and adds up ``ndarray.nbytes``. For a tree or a pipeline this is
    the learned state (tree arrays, scaler statistics) plus the buffered
    samples, i.e. the part of the model's memory that grows with the data.

    Parameters
    ----------
    obj : object
        A model, a pipeline, an array, or a container of those.

    Returns
    -------
    int
        Total number of bytes. Each array is counted once even if it is
        reachable through several attributes.

    Notes
    -----
    Python object overhead (a few hundred bytes per object) is not included,
    so the number is a lower bound that tracks the data-dependent part.
    The walk is over attributes, not over array elements.
    Time: O(number of reachable objects). Space: the same, for the visited set.
    """
    seen = set()
    total = 0
    stack = [obj]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            # a view shares its base's memory: count the base once instead
            base = item.base if isinstance(item.base, np.ndarray) else None
            if base is not None:
                stack.append(base)
            else:
                total += item.nbytes
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        elif hasattr(item, "__dict__") and not isinstance(item, (type, np.random.Generator)):
            stack.extend(vars(item).values())
    return int(total)


def _final_estimator(model):
    """The last step of a Pipeline, or the model itself.

    Parameters
    ----------
    model : object

    Returns
    -------
    object
        The object that owns the buffer and retrain counters, if any.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    steps = getattr(model, "steps", None)
    if isinstance(steps, (list, tuple)) and steps:
        return steps[-1][1]
    return model


class StreamTrainer:
    """Train and evaluate a model chunk by chunk, and log what happened.

    Parameters
    ----------
    model : object
        A model or Pipeline with ``partial_fit(X, y)`` and ``predict(X)``.
    metrics : dict of {str: metric}, optional
        Extra streaming metrics, each with ``update(y_true, y_pred)``,
        ``result()`` and ``reset()`` (see ``metrics.py``). Their current
        values are added to every log record under their names.
        Cumulative, per-chunk and rolling accuracy are always tracked.
    window : int, default 200
        Number of most recent samples used for ``rolling_accuracy``.
    name : str, optional
        Label used in log lines and plots. Default: the model's class name.
    verbose : bool, default False
        Print one line per chunk. Records are always sent to the
        ``numcompute_stream.stream`` logger at INFO level.

    Attributes
    ----------
    history : list of dict
        One record per ``fit_chunk`` call with the keys chunk, n_samples,
        samples_seen, scored, chunk_accuracy, cumulative_accuracy,
        rolling_accuracy, error (1 - chunk accuracy), fit_seconds,
        predict_seconds, retrained, n_retrains, buffer_count, memory_bytes,
        plus one key per extra metric.
    last_y_true_, last_y_pred_ : np.ndarray or None
        Labels and predictions of the most recently scored chunk.
    n_chunks_ : int
        Chunks trained on so far.
    samples_seen_ : int
        Samples trained on so far.

    Raises
    ------
    TypeError
        If the model lacks ``partial_fit`` or ``predict``, or a metric lacks
        ``update`` / ``result`` / ``reset``.
    ValueError
        If an extra metric uses a reserved record name.

    Examples
    --------
    >>> import numpy as np
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> X = np.arange(40, dtype=float).reshape(-1, 1)
    >>> y = (X[:, 0] >= 20).astype(int)
    >>> trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=20))
    >>> history = trainer.run(iter_chunks(X, y, chunk_size=10, shuffle=True, seed=0))
    >>> len(history), history[-1]["n_retrains"]
    (4, 2)
    """

    def __init__(self, model, metrics=None, window=200, name=None, verbose=False):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(
                    f"model ({type(model).__name__}) needs a {method} method to be "
                    "trained on a stream"
                )
        metrics = dict(metrics or {})
        for key, metric in metrics.items():
            if key in _LOG_FIELDS:
                raise ValueError(f"metric name {key!r} is reserved; choose another name")
            for method in ("update", "result", "reset"):
                if not callable(getattr(metric, method, None)):
                    raise TypeError(f"metric {key!r} has no {method} method")
        self.model = model
        self.metrics = metrics
        self.window = window
        self.name = name if name is not None else type(_final_estimator(model)).__name__
        self.verbose = verbose
        self._cumulative = StreamingAccuracy()
        self._rolling = StreamingAccuracy(window=window)
        self.reset_log()

    # ---------------------------------------------------------------- state
    def reset_log(self):
        """Clear the history and every metric. The model is left untouched.

        Returns
        -------
        StreamTrainer
            The same object (`self`).

        Notes
        -----
        Time: O(number of metrics). Space: O(1).
        """
        self.history = []
        self.n_chunks_ = 0
        self.samples_seen_ = 0
        self.last_y_true_ = self.last_y_pred_ = None
        self._pending = None
        self._cumulative.reset()
        self._rolling.reset()
        for metric in self.metrics.values():
            metric.reset()
        return self

    def _unscored(self):
        """Score fields of a chunk that could not be (or was not) scored."""
        out = {"scored": False, "chunk_accuracy": float("nan"),
               "cumulative_accuracy": self._cumulative.result(),
               "rolling_accuracy": self._rolling.result(),
               "error": float("nan"), "predict_seconds": 0.0}
        out.update({key: metric.result() for key, metric in self.metrics.items()})
        return out

    # -------------------------------------------------------------- scoring
    def score_chunk(self, X, y):
        """Predict a chunk with the current model and update the metrics.

        Parameters
        ----------
        X : array_like, shape (n_chunk, n_features)
        y : array_like, shape (n_chunk,)
            True labels of the chunk.

        Returns
        -------
        dict
            scored (bool), chunk_accuracy, cumulative_accuracy,
            rolling_accuracy, error, predict_seconds, plus the current value
            of every extra metric. While the model has no tree yet, or for an
            empty chunk, scored is False and the chunk values are NaN; the
            metrics are not updated.

        Raises
        ------
        ValueError
            If the model rejects X, or y does not match the predictions in
            length.

        Notes
        -----
        The result is also attached to the next ``fit_chunk`` record.
        Time: one ``predict`` plus O(n_chunk) per metric. Space: O(n_chunk).
        """
        y = np.asarray(y)
        start = time.perf_counter()
        try:
            y_pred = np.asarray(self.model.predict(X))
        except NotFittedError:
            self._pending = self._unscored()
            return dict(self._pending)
        seconds = time.perf_counter() - start
        if y_pred.shape[0] == 0:
            self._pending = self._unscored()
            return dict(self._pending)
        self._cumulative.update(y, y_pred)
        self._rolling.update(y, y_pred)
        for metric in self.metrics.values():
            metric.update(y, y_pred)
        chunk_accuracy = float(np.mean(y.ravel() == y_pred.ravel()))
        self.last_y_true_, self.last_y_pred_ = y.ravel().copy(), y_pred.ravel().copy()
        out = {"scored": True, "chunk_accuracy": chunk_accuracy,
               "cumulative_accuracy": self._cumulative.result(),
               "rolling_accuracy": self._rolling.result(),
               "error": 1.0 - chunk_accuracy, "predict_seconds": seconds}
        out.update({key: metric.result() for key, metric in self.metrics.items()})
        self._pending = out
        return dict(out)

    # ------------------------------------------------------------- training
    def fit_chunk(self, X, y):
        """Train the model on one chunk and write a log record.

        Parameters
        ----------
        X : array_like, shape (n_chunk, n_features)
        y : array_like, shape (n_chunk,)

        Returns
        -------
        dict
            The record appended to ``history`` (see the class docstring). It
            includes the scores of the preceding ``score_chunk`` call, or NaN
            if the chunk was not scored first.

        Raises
        ------
        ValueError, TypeError
            Whatever the model's ``partial_fit`` raises for a bad chunk; no
            record is written in that case.

        Notes
        -----
        ``retrained`` is True when this chunk filled the buffer and the tree
        was rebuilt; ``fit_seconds`` then includes the rebuild.
        Time: one ``partial_fit`` plus the memory walk. Space: O(1) extra.
        """
        final = _final_estimator(self.model)
        retrains_before = getattr(final, "n_retrains_", 0)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        seconds = time.perf_counter() - start

        n = int(np.asarray(y).shape[0])
        self.n_chunks_ += 1
        self.samples_seen_ += n
        n_retrains = getattr(final, "n_retrains_", 0)
        record = {"chunk": self.n_chunks_, "n_samples": n,
                  "samples_seen": self.samples_seen_}
        record.update(self._pending if self._pending is not None else self._unscored())
        record.update({
            "fit_seconds": seconds,
            "retrained": n_retrains > retrains_before,
            "n_retrains": n_retrains,
            "buffer_count": getattr(final, "buffer_count_", 0),
            "memory_bytes": memory_footprint(self.model),
        })
        self._pending = None
        self.history.append(record)
        line = self._format_record(record)
        logger.info("%s", line)
        if self.verbose:
            print(line)
        return record

    def run(self, stream):
        """Test-then-train on every chunk of a stream.

        Parameters
        ----------
        stream : iterable of (X_chunk, y_chunk)
            For example ``iter_chunks(X, y, chunk_size=50)``.

        Returns
        -------
        list of dict
            ``history`` (all records so far, including earlier runs).

        Raises
        ------
        ValueError
            If an item of the stream is not an (X, y) pair, or a chunk is
            rejected by the model.

        Notes
        -----
        Each chunk is scored before the model learns from it, so every
        accuracy in the log is measured on data the model had not seen.
        The loop is over chunks.
        """
        for item in stream:
            # a bare array would unpack row by row, so insist on a real pair
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    "every item of the stream must be an (X_chunk, y_chunk) pair, "
                    f"got {type(item).__name__}"
                )
            X_chunk, y_chunk = item
            self.score_chunk(X_chunk, y_chunk)
            self.fit_chunk(X_chunk, y_chunk)
        return self.history

    # ------------------------------------------------------------ reporting
    def get_metric(self, key):
        """One column of the log as an array, ready for plotting.

        Parameters
        ----------
        key : str
            A record key, e.g. "cumulative_accuracy" or "fit_seconds".

        Returns
        -------
        np.ndarray of float, shape (n_chunks,)
            NaN for chunks that were not scored.

        Raises
        ------
        KeyError
            If `key` is not a record key.

        Notes
        -----
        Time: O(n_chunks). Space: O(n_chunks).
        """
        if self.history and key not in self.history[0]:
            raise KeyError(f"no log column {key!r}; columns are {list(self.history[0])}")
        return np.array([record[key] for record in self.history], dtype=float)

    def summary(self):
        """Totals over the whole run.

        Returns
        -------
        dict
            name, chunks, samples, scored_chunks, cumulative_accuracy,
            rolling_accuracy, n_retrains, total_fit_seconds,
            total_predict_seconds, memory_bytes (latest), peak_memory_bytes,
            plus the current value of every extra metric.

        Notes
        -----
        Time: O(n_chunks). Space: O(1).
        """
        out = {
            "name": self.name,
            "chunks": self.n_chunks_,
            "samples": self.samples_seen_,
            "scored_chunks": sum(1 for r in self.history if r["scored"]),
            "cumulative_accuracy": self._cumulative.result(),
            "rolling_accuracy": self._rolling.result(),
            "n_retrains": self.history[-1]["n_retrains"] if self.history else 0,
            "total_fit_seconds": float(sum(r["fit_seconds"] for r in self.history)),
            "total_predict_seconds": float(sum(r["predict_seconds"] for r in self.history)),
            "memory_bytes": self.history[-1]["memory_bytes"] if self.history else 0,
            "peak_memory_bytes": max((r["memory_bytes"] for r in self.history), default=0),
        }
        out.update({key: metric.result() for key, metric in self.metrics.items()})
        return out

    def _format_record(self, record):
        """One log line for a record (used by the logger and ``verbose``)."""
        def pct(value):
            """Format a score, or n/a for a chunk that was not scored."""
            return "   n/a" if np.isnan(value) else f"{value:6.3f}"

        return (
            f"[{self.name}] chunk {record['chunk']:>3} | n={record['n_samples']:>4} "
            f"| seen={record['samples_seen']:>6} | chunk_acc={pct(record['chunk_accuracy'])} "
            f"| cum_acc={pct(record['cumulative_accuracy'])} "
            f"| roll_acc={pct(record['rolling_accuracy'])} "
            f"| fit={record['fit_seconds'] * 1e3:7.2f} ms "
            f"| mem={record['memory_bytes'] / 1024:7.1f} KiB "
            f"| retrains={record['n_retrains']}{' *' if record['retrained'] else ''}"
        )

    def format_log(self, last=None):
        """The log as text, one line per chunk (``*`` marks a retrain).

        Parameters
        ----------
        last : int, optional
            Only the `last` most recent records.

        Returns
        -------
        str

        Notes
        -----
        Time: O(n_chunks). Space: O(n_chunks).
        """
        records = self.history if last is None else self.history[-last:]
        return "\n".join(self._format_record(r) for r in records)

    def save_log(self, path):
        """Write the log to a CSV file, one row per chunk.

        Parameters
        ----------
        path : str or path-like
            Output file; overwritten if it exists.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        OSError
            If the file cannot be written.

        Notes
        -----
        Time: O(n_chunks). Space: O(1).
        """
        fields = list(_LOG_FIELDS) + list(self.metrics)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)
        return str(path)
