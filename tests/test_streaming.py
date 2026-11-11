# tests for the streaming layer: metrics, statistics, preprocessing, pipeline
# and StreamTrainer. The recurring check is "chunk by chunk == all at once".
import csv
import logging

import numpy as np
import pytest

from numcompute_stream import metrics, stats
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import (Imputer, MinMaxScaler,
                                             OneHotEncoder, StandardScaler)
from numcompute_stream.stream import StreamTrainer, iter_chunks, memory_footprint
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.utils import NotFittedError


def labelled_stream(n=600, k=3, seed=0, noise=0.2):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, k, size=n)
    flip = rng.random(n) < noise
    y_pred = np.where(flip, rng.integers(0, k, size=n), y_true)
    return y_true, y_pred


def chunks_of(*arrays, size):
    n = len(arrays[0])
    for start in range(0, n, size):
        yield tuple(a[start:start + size] for a in arrays)


def make_blobs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    y = rng.integers(0, 3, size=n)
    return centres[y] + rng.normal(scale=0.6, size=(n, 2)), y


# ------------------------------------------------------ streaming metrics
@pytest.mark.parametrize("size", [1, 7, 100, 600])
def test_streaming_classification_metrics_equal_batch(size):
    y_true, y_pred = labelled_stream()
    acc, cm = metrics.StreamingAccuracy(), metrics.StreamingConfusionMatrix()
    prec = metrics.StreamingPrecision(average="macro")
    rec = metrics.StreamingRecall(average="weighted")
    f1 = metrics.StreamingF1(average="micro")
    for t, p in chunks_of(y_true, y_pred, size=size):
        for m in (acc, cm, prec, rec, f1):
            m.update(t, p)
    assert acc.result() == pytest.approx(metrics.accuracy(y_true, y_pred))
    np.testing.assert_array_equal(cm.result(), metrics.confusion_matrix(y_true, y_pred))
    assert prec.result() == pytest.approx(metrics.precision(y_true, y_pred, average="macro"))
    assert rec.result() == pytest.approx(metrics.recall(y_true, y_pred, average="weighted"))
    assert f1.result() == pytest.approx(metrics.f1(y_true, y_pred, average="micro"))


def test_streaming_binary_prf_and_per_class():
    y_true, y_pred = labelled_stream(k=2, seed=3)
    p, r, f = metrics.StreamingPrecision(), metrics.StreamingRecall(), metrics.StreamingF1()
    per_class = metrics.StreamingF1(average=None)
    for t, q in chunks_of(y_true, y_pred, size=50):
        for m in (p, r, f, per_class):
            m.update(t, q)
    assert p.result() == pytest.approx(metrics.precision(y_true, y_pred))
    assert r.result() == pytest.approx(metrics.recall(y_true, y_pred))
    assert f.result() == pytest.approx(metrics.f1(y_true, y_pred))
    np.testing.assert_allclose(per_class.result(), metrics.f1(y_true, y_pred, average=None))


def test_confusion_matrix_grows_when_new_classes_arrive():
    cm = metrics.StreamingConfusionMatrix()
    cm.update(["b", "b"], ["b", "d"])
    assert cm.labels_.tolist() == ["b", "d"]
    cm.update(["a", "c"], ["a", "b"])           # two new classes, in the middle too
    assert cm.labels_.tolist() == ["a", "b", "c", "d"]
    expected = metrics.confusion_matrix(["b", "b", "a", "c"], ["b", "d", "a", "b"])
    np.testing.assert_array_equal(cm.result(), expected)


def test_confusion_matrix_fixed_labels_keep_shape():
    cm = metrics.StreamingConfusionMatrix(labels=[0, 1, 2])
    cm.update([0, 0], [0, 1])
    assert cm.result().shape == (3, 3)
    assert cm.result().sum() == 2
    with pytest.raises(ValueError, match="duplicates"):
        metrics.StreamingConfusionMatrix(labels=[0, 0])


def test_rolling_window_metrics_use_only_recent_samples():
    y_true, y_pred = labelled_stream(n=500, seed=4)
    window = 120
    acc = metrics.StreamingAccuracy(window=window)
    f1 = metrics.StreamingF1(average="macro", window=window)
    cm = metrics.StreamingConfusionMatrix(window=window)
    for t, p in chunks_of(y_true, y_pred, size=33):
        acc.update(t, p), f1.update(t, p), cm.update(t, p)
    tail_t, tail_p = y_true[-window:], y_pred[-window:]
    assert acc.result() == pytest.approx(metrics.accuracy(tail_t, tail_p))
    assert f1.result() == pytest.approx(metrics.f1(tail_t, tail_p, average="macro"))
    np.testing.assert_array_equal(cm.result(), metrics.confusion_matrix(tail_t, tail_p))
    assert cm.result().sum() == window


def test_rolling_window_reacts_to_a_change_cumulative_lags():
    cumulative, rolling = metrics.StreamingAccuracy(), metrics.StreamingAccuracy(window=50)
    good = np.zeros(500, dtype=int)
    for m in (cumulative, rolling):
        m.update(good, good)                    # 500 correct
        m.update(good[:50], good[:50] + 1)      # then 50 wrong
    assert rolling.result() == 0.0
    assert cumulative.result() == pytest.approx(500 / 550)


def test_metric_reset_and_empty_state():
    acc = metrics.StreamingAccuracy()
    assert np.isnan(acc.result())               # nothing seen: NaN, not an error
    acc.update([], [])                          # empty chunk is allowed
    assert np.isnan(acc.result()) and acc.n_samples_ == 0
    acc.update([1, 1], [1, 0])
    assert acc.result() == 0.5
    assert np.isnan(acc.reset().result()) and acc.n_samples_ == 0
    assert metrics.StreamingConfusionMatrix().result().shape == (0, 0)


def test_metric_update_validates_and_leaves_state_unchanged():
    acc = metrics.StreamingAccuracy().update([0, 1], [0, 1])
    with pytest.raises(ValueError, match="same length"):
        acc.update([0, 1, 1], [0, 1])
    with pytest.raises(ValueError, match="NaN"):
        acc.update([0.0, np.nan], [0.0, 1.0])
    with pytest.raises(ValueError, match="1D"):
        acc.update(np.zeros((2, 2)), np.zeros((2, 2)))
    assert acc.result() == 1.0 and acc.n_samples_ == 2
    for bad in (0, -3, 2.5, True):
        with pytest.raises((ValueError, TypeError)):
            metrics.StreamingAccuracy(window=bad)


def test_binary_prf_before_positive_class_appears():
    rec = metrics.StreamingRecall(pos_label=1, zero_division=0.0)
    rec.update([0, 0], [0, 0])
    assert rec.result() == 0.0
    rec.update([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        rec.result()                            # three classes: binary is undefined


def test_streaming_mse_equals_batch_and_window():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=1000) + 1e6, rng.normal(size=1000) + 1e6
    whole, rolling = metrics.StreamingMSE(), metrics.StreamingMSE(window=64)
    for t, p in chunks_of(a, b, size=37):
        whole.update(t, p), rolling.update(t, p)
    assert whole.result() == pytest.approx(metrics.mse(a, b), rel=1e-12)
    assert rolling.result() == pytest.approx(metrics.mse(a[-64:], b[-64:]), rel=1e-12)
    assert np.isnan(metrics.StreamingMSE().result())


def test_streaming_auc_accumulates_and_matches_exact():
    rng = np.random.default_rng(1)
    score = rng.integers(0, 101, size=3000) / 100.0      # scores on a 0.01 grid
    y = (rng.random(3000) < score).astype(int)
    auc = metrics.StreamingAUC(n_bins=1000)
    rolling = metrics.StreamingAUC(window=400)
    for t, s in chunks_of(y, score, size=250):
        auc.update(t, s), rolling.update(t, s)
    # on a grid the histogram loses nothing, so the result is exact
    assert auc.result() == pytest.approx(metrics.roc_auc_score(y, score), abs=1e-12)
    assert rolling.result() == pytest.approx(metrics.roc_auc_score(y[-400:], score[-400:]))


def test_streaming_auc_edge_cases():
    auc = metrics.StreamingAUC()
    assert np.isnan(auc.result())
    auc.update([1, 1], [0.2, 0.9])              # one class only: undefined
    assert np.isnan(auc.result())
    auc.update([0], [0.5])
    assert auc.result() == pytest.approx(0.5)
    tied = metrics.StreamingAUC().update([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert tied.result() == pytest.approx(0.5)  # all tied -> chance level
    with pytest.raises(ValueError, match="finite"):
        auc.update([0, 1], [0.1, np.inf])


# ---------------------------------------------------- streaming statistics
@pytest.mark.parametrize("size", [1, 13, 500])
def test_running_stats_equal_batch_for_any_chunking(size):
    rng = np.random.default_rng(2)
    data = rng.normal(loc=1e8, scale=3.0, size=(500, 3))  # large mean: stability
    data[rng.random(data.shape) < 0.05] = np.nan
    rs = stats.RunningStats()
    for (chunk,) in chunks_of(data, size=size):
        rs.update_stats(chunk)
    np.testing.assert_allclose(rs.mean(), stats.mean(data, axis=0), rtol=1e-12)
    np.testing.assert_allclose(rs.var(ddof=1), stats.var(data, axis=0, ddof=1), rtol=1e-8)
    np.testing.assert_allclose(rs.min(), np.nanmin(data, axis=0))
    np.testing.assert_allclose(rs.max(), np.nanmax(data, axis=0))


def test_running_stats_zero_variance_and_all_nan_chunks():
    rs = stats.RunningStats()
    rs.partial_fit(np.full((10, 2), 7.0))
    rs.partial_fit(np.full((5, 2), np.nan))     # contributes nothing
    rs.partial_fit(np.empty((0, 2)))            # neither does an empty chunk
    np.testing.assert_array_equal(rs.var(), [0.0, 0.0])
    np.testing.assert_array_equal(rs.mean(), [7.0, 7.0])
    np.testing.assert_array_equal(rs.count(), [10, 10])


def test_streaming_quantile_exact_below_capacity():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(900, 2))
    data[5, 0] = np.nan
    sq = stats.StreamingQuantile(capacity=1000)
    for (chunk,) in chunks_of(data, size=64):
        sq.update(chunk)
    expected = stats.quantile(data, [0.1, 0.5, 0.9], axis=0)
    np.testing.assert_allclose(sq.result([0.1, 0.5, 0.9]), expected)
    assert sq.result(0.5).shape == (2,)


def test_streaming_quantile_reservoir_is_bounded_and_close():
    rng = np.random.default_rng(4)
    data = rng.exponential(size=50_000)
    sq = stats.StreamingQuantile(capacity=2000, seed=1)
    for (chunk,) in chunks_of(data, size=777):
        sq.update(chunk)
    assert sq._rows.shape[0] == 2000 and sq.n_seen_ == 50_000
    qs = np.array([0.25, 0.5, 0.75])
    assert np.abs(sq.result(qs) - np.quantile(data, qs)).max() < 0.08
    again = stats.StreamingQuantile(capacity=2000, seed=1)
    for (chunk,) in chunks_of(data, size=777):
        again.update(chunk)
    np.testing.assert_array_equal(again.result(qs), sq.result(qs))   # seeded


def test_streaming_quantile_sliding_window_is_exact():
    data = np.arange(1000, dtype=float)
    sq = stats.StreamingQuantile(window=100)
    for (chunk,) in chunks_of(data, size=30):
        sq.update(chunk)
    assert sq.result(0.5) == pytest.approx(np.median(data[-100:]))
    with pytest.raises(RuntimeError):
        stats.StreamingQuantile().result()


def test_streaming_histogram_equals_batch_and_counts_outliers():
    rng = np.random.default_rng(5)
    data = rng.normal(size=5000)
    data[:7] = np.nan
    hist = stats.StreamingHistogram(bins=12, range=(-2.0, 2.0))
    for (chunk,) in chunks_of(data, size=333):
        hist.update(chunk)
    counts, edges = hist.result()
    ref_counts, ref_edges = stats.histogram(data, bins=12, range=(-2.0, 2.0))
    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_allclose(edges, ref_edges)
    valid = data[~np.isnan(data)]
    assert hist.n_below_ == (valid < -2).sum() and hist.n_above_ == (valid > 2).sum()
    assert hist.n_nan_ == 7
    density, _ = hist.result(density=True)
    assert (density * np.diff(edges)).sum() == pytest.approx(1.0)


def test_streaming_histogram_sliding_window_and_right_edge():
    hist = stats.StreamingHistogram(bins=[0.0, 1.0, 2.0], window=4)
    hist.update([0.5, 0.5, 0.5]).update([1.5, 2.0, 1.0])     # 2.0 is in the last bin
    counts, _ = hist.result()
    assert counts.tolist() == [1, 3]                          # last 4 values only
    with pytest.raises(ValueError, match="range"):
        stats.StreamingHistogram(bins=5)                      # edges must be fixed
    with pytest.raises(ValueError, match="increasing"):
        stats.StreamingHistogram(bins=[0.0, 0.0, 1.0])


def test_update_stats_api_function_and_class():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(300, 2))
    state = None
    for (chunk,) in chunks_of(data, size=40):
        state = stats.update_stats(chunk, state)
    summary = state.result()
    np.testing.assert_allclose(summary["mean"], data.mean(axis=0))
    np.testing.assert_allclose(summary["std"], data.std(axis=0))
    np.testing.assert_allclose(summary["quantiles"][0.5], np.median(data, axis=0))
    with_hist = stats.StreamStats(hist_bins=4, hist_range=(-3, 3)).update_stats(data)
    assert len(with_hist.result()["histogram"]) == 2
    with pytest.raises(ValueError, match="expected"):
        state.update_stats(np.ones((3, 5)))                   # column count changed
    with pytest.raises(TypeError):
        stats.update_stats(data, state="not a state")
    with pytest.raises(ValueError, match="infinity"):
        stats.update_stats([1.0, np.inf])


# -------------------------------------------------- streaming preprocessing
def test_scalers_partial_fit_equal_fit_and_handle_zero_variance():
    rng = np.random.default_rng(7)
    X = rng.normal(loc=50.0, scale=4.0, size=(240, 3))
    X[:, 2] = 3.0                                             # zero-variance feature
    X[4, 0] = np.nan
    for cls in (StandardScaler, MinMaxScaler):
        stream, batch = cls(), cls().fit(X)
        for (chunk,) in chunks_of(X, size=25):
            stream.partial_fit(chunk)
        out = stream.transform(X)
        np.testing.assert_allclose(out, batch.transform(X), rtol=1e-10, atol=1e-10)
        assert np.isfinite(out[:, 2]).all()                   # no division by zero


def test_imputer_updates_fill_value_on_the_fly():
    imp = Imputer(strategy="mean")
    imp.partial_fit(np.array([[1.0], [3.0]]))
    assert imp.transform(np.array([[np.nan]]))[0, 0] == pytest.approx(2.0)
    imp.partial_fit(np.array([[np.nan], [8.0]]))              # NaN is ignored
    assert imp.transform(np.array([[np.nan]]))[0, 0] == pytest.approx(4.0)


def test_one_hot_encoder_expands_categories_incrementally():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([["red"], ["blue"]]))
    first = enc.transform(np.array([["red"], ["blue"]]))
    enc.partial_fit(np.array([["green"], ["red"]]))           # a new category
    second = enc.transform(np.array([["red"], ["blue"], ["green"]]))
    assert first.shape[1] == 2 and second.shape[1] == 3
    np.testing.assert_array_equal(second[:2, :2], first)      # old columns never move


# --------------------------------------------------------- pipeline stream
def test_pipeline_partial_fit_chains_transformers_into_the_tree():
    X, y = make_blobs()
    X = X * np.array([1.0, 1000.0])
    X[3, 1] = np.nan
    pipe = Pipeline([
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=100)),
    ])
    with pytest.raises(NotFittedError):
        pipe.predict(X)                                       # nothing seen yet
    pipe.partial_fit(X[:50], y[:50])
    with pytest.raises(NotFittedError):
        pipe.predict(X)                                       # tree buffer still filling
    for Xc, yc in chunks_of(X[50:], y[50:], size=50):
        pipe.partial_fit(Xc, yc)
    assert pipe["model"].n_retrains_ == 4
    assert pipe["scale"].n_.max() == len(X)                   # scaler saw every row
    assert np.mean(pipe.predict(X) == y) > 0.95


def test_pipeline_partial_fit_requires_streaming_steps():
    class BatchOnly:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    pipe = Pipeline([("batch", BatchOnly()), ("model", DecisionTreeClassifier())])
    with pytest.raises(TypeError, match="no partial_fit"):
        pipe.partial_fit(np.ones((2, 2)), [0, 1])


# ------------------------------------------------------------ StreamTrainer
def test_stream_trainer_logs_one_record_per_chunk():
    X, y = make_blobs(n=300)
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=4, buffer_size=60), window=100)
    history = trainer.run(iter_chunks(X, y, chunk_size=30))
    assert len(history) == 10 and trainer.samples_seen_ == 300
    assert [r["chunk"] for r in history] == list(range(1, 11))
    # warm-up: nothing can be scored until the first retrain (chunk 2)
    assert [r["scored"] for r in history[:3]] == [False, False, True]
    assert np.isnan(history[0]["chunk_accuracy"])
    assert [r["retrained"] for r in history] == [False, True] * 5
    assert history[-1]["n_retrains"] == 5
    assert all(r["memory_bytes"] > 0 and r["fit_seconds"] >= 0 for r in history)
    # the buffer empties on a retrain, so the footprint drops there
    assert history[1]["memory_bytes"] < history[0]["memory_bytes"] * 2
    assert trainer.summary()["cumulative_accuracy"] > 0.9


def test_stream_trainer_scores_before_training():
    # prequential: the cumulative accuracy must equal test-then-train by hand
    X, y = make_blobs(n=200, seed=9)
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=40))
    trainer.run(iter_chunks(X, y, chunk_size=20))
    tree = DecisionTreeClassifier(buffer_size=40)
    hits = total = 0
    for Xc, yc in iter_chunks(X, y, chunk_size=20):
        if tree.is_fitted:
            hits += int((tree.predict(Xc) == yc).sum())
            total += len(yc)
        tree.partial_fit(Xc, yc)
    assert trainer.summary()["cumulative_accuracy"] == pytest.approx(hits / total)
    np.testing.assert_array_equal(trainer.last_y_true_, y[-20:])


def test_stream_trainer_with_pipeline_extra_metrics_and_csv(tmp_path):
    X, y = make_blobs(n=240)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=4, buffer_size=40))])
    trainer = StreamTrainer(pipe, metrics={"macro_f1": metrics.StreamingF1(average="macro")},
                            window=80, name="scaled tree")
    trainer.run(iter_chunks(X, y, chunk_size=40, shuffle=True, seed=1))
    assert trainer.get_metric("cumulative_accuracy").shape == (6,)
    assert 0.0 <= trainer.history[-1]["macro_f1"] <= 1.0
    assert "scaled tree" in trainer.format_log(last=1)
    path = trainer.save_log(tmp_path / "log.csv")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and "macro_f1" in rows[0] and rows[-1]["n_retrains"] == "6"
    with pytest.raises(KeyError):
        trainer.get_metric("no_such_column")


def test_stream_trainer_emits_log_records(caplog):
    X, y = make_blobs(n=40)
    with caplog.at_level(logging.INFO, logger="numcompute_stream.stream"):
        StreamTrainer(DecisionTreeClassifier(buffer_size=20)).run(iter_chunks(X, y, 20))
    assert len(caplog.records) == 2 and "retrains=1 *" in caplog.records[0].getMessage()


def test_stream_trainer_validates_inputs():
    with pytest.raises(TypeError, match="partial_fit"):
        StreamTrainer(object())
    with pytest.raises(TypeError, match="update"):
        StreamTrainer(DecisionTreeClassifier(), metrics={"bad": object()})
    with pytest.raises(ValueError, match="reserved"):
        StreamTrainer(DecisionTreeClassifier(), metrics={"chunk": metrics.StreamingAccuracy()})
    trainer = StreamTrainer(DecisionTreeClassifier())
    with pytest.raises(ValueError, match="pair"):
        trainer.run([np.ones((2, 2))])
    with pytest.raises(ValueError):
        trainer.fit_chunk(np.ones((3, 2)), [0, 1])            # lengths differ
    assert trainer.history == []                              # nothing was logged


def test_memory_footprint_counts_each_array_once():
    a = np.zeros(1000)                                        # 8000 bytes
    assert memory_footprint(a) == 8000
    assert memory_footprint({"x": a, "y": [a, a[:10]]}) == 8000   # views share memory
    tree = DecisionTreeClassifier(buffer_size=10_000)
    empty = memory_footprint(tree)
    tree.partial_fit(np.zeros((500, 4)), np.zeros(500, dtype=np.int64))
    assert memory_footprint(tree) - empty == 500 * 4 * 8 + 500 * 8
