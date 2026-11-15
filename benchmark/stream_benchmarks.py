"""
stream_benchmarks.py
====================
Streaming benchmarks for NumCompute-Stream. Two parts:

(a) loop vs vectorised
    The same streaming computation written with plain Python loops and with
    the package's vectorised code: the tree's split search, tree prediction,
    a whole buffered-tree stream, the streaming confusion matrix, running
    mean/variance and the streaming histogram. Both versions run on the same
    seeded data, their results are checked for equality first, then timed.

(b) buffer_size
    One stream, many trees that differ only in ``buffer_size``. For each the
    script reports prequential (test-then-train) accuracy, the number of
    retrains, the retraining time and the peak memory. The stream changes
    its labelling rule half way (concept drift), so the table also shows how
    fast each buffer size recovers.

Run from the repository root::

    python benchmark/stream_benchmarks.py            # full run (about a minute)
    python benchmark/stream_benchmarks.py --quick    # small sizes, a few seconds
    python benchmark/stream_benchmarks.py --part a   # only loop vs vectorised
    python benchmark/stream_benchmarks.py --no-plots

Results are printed and written to ``benchmark/results/``:
``stream_loop_vs_vectorised.csv/.md``, ``buffer_size_results.csv/.md`` and
three PNG plots. Exit status is 0 only if every loop/vectorised pair agreed.

Conventions
-----------
- Data comes from ``numpy.random.default_rng(seed)``: the same seed gives the
  same data and the same accuracies on every machine; timings differ.
- Times are medians over ``--repeat`` runs (part a) or wall-clock totals of
  one pass over the stream (part b).

"""

import argparse
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from numcompute_stream import benchmarking as bench  # noqa: E402
from numcompute_stream import metrics, stats  # noqa: E402
from numcompute_stream.stream import StreamTrainer, iter_chunks  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier, _best_split  # noqa: E402

CHUNK = 100


# --------------------------------------------------------------------------
# (a) pure-Python loop baselines
# --------------------------------------------------------------------------
def loop_gini(labels, n_classes):
    """Gini impurity of a list of class indices, with Python loops."""
    if not labels:
        return 0.0
    counts = [0] * n_classes
    for c in labels:
        counts[c] += 1
    total = float(len(labels))
    return 1.0 - sum((c / total) ** 2 for c in counts)


def loop_best_split(X, y, n_classes):
    """Best (feature, threshold, gain): every feature, every midpoint, by loop.

    X is a list of rows, y a list of class indices. Mirrors ``_best_split``
    (same gain, same tie rule: first feature, then lowest threshold).
    """
    n, d = len(X), len(X[0])
    before = loop_gini(y, n_classes)
    best_f, best_t, best_gain = -1, float("nan"), 0.0
    for f in range(d):
        values = sorted({row[f] for row in X})
        for a, b in zip(values[:-1], values[1:]):
            t = a + (b - a) / 2.0
            left = [y[i] for i in range(n) if X[i][f] <= t]
            right = [y[i] for i in range(n) if X[i][f] > t]
            after = (len(left) * loop_gini(left, n_classes)
                     + len(right) * loop_gini(right, n_classes)) / n
            if before - after > best_gain + 1e-12:
                best_f, best_t, best_gain = f, t, before - after
    return best_f, best_t, best_gain


def loop_build(X, y, n_classes, max_depth, depth=0):
    """Recursive loop-based tree: nested dicts, majority label at the leaves."""
    counts = [0] * n_classes
    for c in y:
        counts[c] += 1
    node = {"label": counts.index(max(counts))}
    if depth >= max_depth or len(y) < 2 or max(counts) == len(y):
        return node
    f, t, _ = loop_best_split(X, y, n_classes)
    if f == -1:
        return node
    left = [i for i in range(len(y)) if X[i][f] <= t]
    right = [i for i in range(len(y)) if X[i][f] > t]
    node.update(feature=f, threshold=t,
                left=loop_build([X[i] for i in left], [y[i] for i in left],
                                n_classes, max_depth, depth + 1),
                right=loop_build([X[i] for i in right], [y[i] for i in right],
                                 n_classes, max_depth, depth + 1))
    return node


def loop_predict_one(node, row):
    """Walk one sample down a nested-dict tree."""
    while "feature" in node:
        node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
    return node["label"]


def loop_predict_arrays(tree, X):
    """Predict one sample at a time from the fitted tree's arrays."""
    feature, threshold = tree.feature_.tolist(), tree.threshold_.tolist()
    left, right = tree.left_.tolist(), tree.right_.tolist()
    labels = np.argmax(tree.value_, axis=1).tolist()
    out = []
    for row in X:
        node = 0
        while feature[node] != -1:
            node = left[node] if row[feature[node]] <= threshold[node] else right[node]
        out.append(labels[node])
    return out


def loop_buffered_stream(chunks, n_classes, max_depth, buffer_size):
    """A whole buffered-tree stream with loops: buffer, retrain, predict."""
    buf_X, buf_y, tree, predictions = [], [], None, []
    for Xc, yc in chunks:
        if tree is not None:
            predictions.extend(loop_predict_one(tree, row) for row in Xc)
        buf_X.extend(Xc)
        buf_y.extend(yc)
        if len(buf_y) >= buffer_size:
            tree = loop_build(buf_X, buf_y, n_classes, max_depth)
            buf_X, buf_y = [], []
    return predictions


def vec_buffered_stream(chunks, n_classes, max_depth, buffer_size):
    """The same stream with DecisionTreeClassifier.partial_fit / predict."""
    tree = DecisionTreeClassifier(max_depth=max_depth, buffer_size=buffer_size)
    predictions = []
    for Xc, yc in chunks:
        if tree.is_fitted:
            predictions.extend(tree.predict(Xc).tolist())
        tree.partial_fit(Xc, yc)
    return predictions


def loop_confusion(chunks, k):
    """Streaming confusion matrix: one increment per sample."""
    cm = [[0] * k for _ in range(k)]
    for y_true, y_pred in chunks:
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1
    return cm


def vec_confusion(chunks, k):
    cm = metrics.StreamingConfusionMatrix(labels=np.arange(k))
    for y_true, y_pred in chunks:
        cm.update(y_true, y_pred)
    return cm.result()


def loop_welford(chunks):
    """Running mean and variance: Welford's update, one value at a time."""
    n, mean, m2 = 0, 0.0, 0.0
    for chunk in chunks:
        for x in chunk:
            n += 1
            delta = x - mean
            mean += delta / n
            m2 += delta * (x - mean)
    return mean, m2 / n


def vec_welford(chunks):
    rs = stats.RunningStats()
    for chunk in chunks:
        rs.partial_fit(chunk)
    return float(rs.mean()), float(rs.var())


def loop_histogram(chunks, low, high, bins):
    """Streaming histogram: one bin lookup per value."""
    counts = [0] * bins
    width = (high - low) / bins
    for chunk in chunks:
        for x in chunk:
            if low <= x <= high:
                counts[min(int((x - low) / width), bins - 1)] += 1
    return counts


def vec_histogram(chunks, low, high, bins):
    hist = stats.StreamingHistogram(bins=bins, range=(low, high))
    for chunk in chunks:
        hist.update(chunk)
    return hist.result()[0]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def make_classification(n, d, rng, n_classes=3):
    """Gaussian blobs on a grid of values (so thresholds have ties)."""
    y = rng.integers(0, n_classes, size=n)
    centres = rng.normal(scale=2.0, size=(n_classes, d))
    X = np.round(centres[y] + rng.normal(size=(n, d)), 1)
    return X, y


def make_drift_stream(n, rng, d=10, noise=0.05):
    """A stream whose labelling rule changes abruptly half way.

    First half: class = quadrant of (x0, x1). Second half: the same rule on
    (x2, x3), so a tree trained before the drift is close to random after it.
    A fraction `noise` of the labels is replaced at random.
    """
    X = rng.normal(size=(n, d))
    half = n // 2
    y = np.empty(n, dtype=np.int64)
    y[:half] = (X[:half, 0] > 0).astype(int) * 2 + (X[:half, 1] > 0).astype(int)
    y[half:] = (X[half:, 2] > 0).astype(int) * 2 + (X[half:, 3] > 0).astype(int)
    flip = rng.random(n) < noise
    y[flip] = rng.integers(0, 4, size=int(flip.sum()))
    return X, y


def as_chunks(*arrays, size=CHUNK, to_list=False):
    """Pre-split arrays into chunk tuples (lists for the loop baselines)."""
    out = []
    for start in range(0, len(arrays[0]), size):
        part = tuple(a[start:start + size] for a in arrays)
        out.append(tuple(p.tolist() for p in part) if to_list else part)
    return out


# --------------------------------------------------------------------------
# (a) loop vs vectorised
# --------------------------------------------------------------------------
def run_loop_vs_vectorised(quick, repeat, seed):
    """Time every loop/vectorised pair. Returns the list of result rows."""
    rows = []

    def add(name, n, loop_fn, vec_fn, same, baseline):
        row = bench.compare(name, n, loop_fn, vec_fn, (), same,
                            repeat=repeat, baseline=baseline)
        rows.append(row)
        print(f"  {name:<44} n={n:>8,}  loop {row['loop_ms']:>10.3f} ms  "
              f"vec {row['vec_ms']:>9.3f} ms  {row['speedup']:>7.1f}x  "
              f"{'ok' if row['ok'] else 'MISMATCH'}", flush=True)

    def same_split(a, b):
        return a[0] == b[0] and np.isclose(a[1], b[1]) and np.isclose(a[2], b[2])

    for n in ((200,) if quick else (200, 1000, 4000)):
        rng = np.random.default_rng(seed)
        X, y = make_classification(n, 8, rng)
        Xl, yl = X.tolist(), y.tolist()
        add("tree: best split of one node (d=8)", n,
            lambda: loop_best_split(Xl, yl, 3), lambda: _best_split(X, y, 3),
            same_split, "loop over features x thresholds x rows")

    for n in ((2000,) if quick else (2000, 20000, 200000)):
        rng = np.random.default_rng(seed)
        X, y = make_classification(n, 8, rng)
        tree = DecisionTreeClassifier(max_depth=6).fit(X[:2000], y[:2000])
        Xl = X.tolist()
        add("tree: predict (depth 6)", n,
            lambda: loop_predict_arrays(tree, Xl),
            lambda: np.argmax(tree.value_[tree.apply(X)], axis=1),
            lambda a, b: np.array_equal(a, b), "one root-to-leaf walk per row")

    for n in ((1000,) if quick else (1000, 4000)):
        rng = np.random.default_rng(seed)
        X, y = make_classification(n, 6, rng)
        arr_chunks, list_chunks = as_chunks(X, y), as_chunks(X, y, to_list=True)
        add("tree: buffered stream (buffer 500, depth 4)", n,
            lambda: loop_buffered_stream(list_chunks, 3, 4, 500),
            lambda: vec_buffered_stream(arr_chunks, 3, 4, 500),
            lambda a, b: a == b, "loop tree, per-row predict")

    # per-chunk NumPy calls have a fixed overhead, so the chunk size decides
    # who wins: both a small and a large chunk are timed on the same data
    n = 20000 if quick else 1000000
    for chunk in (100, 10000):
        rng = np.random.default_rng(seed)
        y_true, y_pred = rng.integers(0, 5, size=n), rng.integers(0, 5, size=n)
        arr_chunks = as_chunks(y_true, y_pred, size=chunk)
        list_chunks = as_chunks(y_true, y_pred, size=chunk, to_list=True)
        add(f"metrics: confusion matrix, chunks of {chunk:,}", n,
            lambda: loop_confusion(list_chunks, 5), lambda: vec_confusion(arr_chunks, 5),
            lambda a, b: np.array_equal(a, b), "one increment per sample")

        data = rng.normal(loc=100.0, scale=5.0, size=n)
        arr_chunks = [c[0] for c in as_chunks(data, size=chunk)]
        list_chunks = [c.tolist() for c in arr_chunks]
        add(f"stats: running mean/variance, chunks of {chunk:,}", n,
            lambda: loop_welford(list_chunks), lambda: vec_welford(arr_chunks),
            lambda a, b: np.allclose(a, b, rtol=1e-9), "Welford, one value at a time")
        add(f"stats: histogram (20 bins), chunks of {chunk:,}", n,
            lambda: loop_histogram(list_chunks, 80.0, 120.0, 20),
            lambda: vec_histogram(arr_chunks, 80.0, 120.0, 20),
            lambda a, b: np.array_equal(a, b), "one bin lookup per value")
    return rows


# --------------------------------------------------------------------------
# (b) buffer size
# --------------------------------------------------------------------------
def run_buffer_sizes(quick, seed):
    """Run one stream through trees with different buffer sizes.

    Returns (rows, curves): one summary dict per buffer size, and the
    rolling-accuracy curve of each for plotting.
    """
    n = 6000 if quick else 40000
    sizes = (100, 500, 2000) if quick else (100, 250, 500, 1000, 2000, 5000, 10000)
    X, y = make_drift_stream(n, np.random.default_rng(seed))
    half_chunk = (n // 2) // CHUNK
    rows, curves = [], {}
    for size in sizes:
        tree = DecisionTreeClassifier(max_depth=6, buffer_size=size)
        trainer = StreamTrainer(tree, window=1000, name=f"buffer {size}")
        trainer.run(iter_chunks(X, y, chunk_size=CHUNK))
        acc = trainer.get_metric("chunk_accuracy")
        fit = trainer.get_metric("fit_seconds")
        retrained = np.array([r["retrained"] for r in trainer.history])
        summary = trainer.summary()
        # chunks after the drift until rolling accuracy is back above 0.8
        rolling = trainer.get_metric("rolling_accuracy")
        after = np.flatnonzero(rolling[half_chunk + 10:] >= 0.8)
        recovery = int(after[0] + 10) * CHUNK if after.size else -1
        rows.append({
            "buffer_size": size,
            "retrains": summary["n_retrains"],
            "accuracy": summary["cumulative_accuracy"],
            "acc_before_drift": float(np.nanmean(acc[:half_chunk])),
            "acc_after_drift": float(np.nanmean(acc[half_chunk:])),
            "samples_to_recover": recovery,
            "mean_retrain_ms": float(fit[retrained].mean() * 1e3) if retrained.any() else 0.0,
            "total_retrain_ms": float(fit[retrained].sum() * 1e3),
            "predict_ms": summary["total_predict_seconds"] * 1e3,
            "peak_memory_kib": summary["peak_memory_bytes"] / 1024,
        })
        curves[size] = rolling
        r = rows[-1]
        print(f"  buffer_size={size:>6}  retrains={r['retrains']:>4}  "
              f"acc={r['accuracy']:.3f}  after drift={r['acc_after_drift']:.3f}  "
              f"mean retrain={r['mean_retrain_ms']:>8.2f} ms  "
              f"total={r['total_retrain_ms']:>9.1f} ms  "
              f"peak mem={r['peak_memory_kib']:>8.1f} KiB", flush=True)
    return rows, curves, n


def format_buffer_table(rows, env, n):
    """Markdown table for the buffer-size results."""
    lines = ["**Environment**"]
    lines.extend(f"- {key}: {value}" for key, value in env.items())
    lines += [
        f"- stream: {n:,} samples, 10 features, 4 classes, chunks of {CHUNK}, "
        "labelling rule changes at the half-way point, 5% label noise",
        "- tree: max_depth=6, gini; accuracy is prequential (test-then-train)",
        "- samples to recover: samples after the drift until the rolling accuracy "
        "(last 1,000 samples) is back at 0.8 or more, checked from 1,000 samples "
        "after the drift (when the window holds only post-drift data)",
        "",
        "| buffer_size | retrains | accuracy | before drift | after drift | "
        "samples to recover | mean retrain (ms) | total retrain (ms) | peak memory (KiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        recover = f"{r['samples_to_recover']:,}" if r["samples_to_recover"] >= 0 else "never"
        lines.append(
            f"| {r['buffer_size']:,} | {r['retrains']} | {r['accuracy']:.3f} | "
            f"{r['acc_before_drift']:.3f} | {r['acc_after_drift']:.3f} | {recover} | "
            f"{r['mean_retrain_ms']:.2f} | {r['total_retrain_ms']:.1f} | "
            f"{r['peak_memory_kib']:.1f} |"
        )
    return "\n".join(lines)


def save_plots(rows, curves, out):
    """Write the three buffer-size plots with the package's own plotting module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numcompute_stream import visualise

    sizes = [r["buffer_size"] for r in rows]
    fig, ax = visualise.plot_metric_over_time(
        [r["accuracy"] for r in rows], "Accuracy vs buffer size",
        "Prequential accuracy", xlabel="buffer_size (log scale)", x=sizes, xscale="log",
        show=False)
    fig.savefig(os.path.join(out, "buffer_size_accuracy.png"), dpi=150)
    fig, ax = visualise.plot_metric_over_time(
        [r["mean_retrain_ms"] for r in rows], "Retraining time vs buffer size",
        "Mean time per retrain (ms)", xlabel="buffer_size (log scale)", x=sizes, xscale="log",
        show=False)
    fig.savefig(os.path.join(out, "buffer_size_retrain_time.png"), dpi=150)
    small, large = sizes[0], sizes[-1]
    visualise.compare_models(
        curves[small], curves[large],
        labels=(f"buffer_size={small}", f"buffer_size={large}"),
        title="Rolling accuracy around a concept drift (window 1,000)",
        ylabel="Rolling accuracy", xlabel=f"Chunk ({CHUNK} samples each)",
        save_path=os.path.join(out, "buffer_size_drift.png"), show=False)
    plt.close("all")


# --------------------------------------------------------------------------
def _parse_args(argv):
    p = argparse.ArgumentParser(description="NumCompute-Stream streaming benchmarks")
    p.add_argument("--quick", action="store_true", help="small sizes, a few seconds")
    p.add_argument("--part", choices=("a", "b", "both"), default="both",
                   help="a: loop vs vectorised, b: buffer sizes (default both)")
    p.add_argument("--repeat", type=int, default=5, help="timed repeats in part a")
    p.add_argument("--seed", type=int, default=0, help="data seed")
    p.add_argument("--out", default=os.path.join(_HERE, "results"),
                   help="output folder (default benchmark/results)")
    p.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    return p.parse_args(argv)


def main(argv=None):
    """Run the selected parts, print the tables and save the results.

    Returns
    -------
    int
        0 if every loop/vectorised pair agreed, otherwise 1.
    """
    args = _parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env = bench.environment_info(args.seed)
    failed = False

    if args.part in ("a", "both"):
        print("(a) loop vs vectorised, streaming scenarios")
        rows = run_loop_vs_vectorised(args.quick, args.repeat, args.seed)
        table = bench.format_table(rows, env)
        print("\n" + table + "\n")
        bench.write_csv(rows, os.path.join(args.out, "stream_loop_vs_vectorised.csv"))
        with open(os.path.join(args.out, "stream_loop_vs_vectorised.md"), "w",
                  encoding="utf-8") as handle:
            handle.write(table + "\n")
        failed = not all(r["ok"] for r in rows)

    if args.part in ("b", "both"):
        print("(b) buffer_size: accuracy and retraining time")
        rows, curves, n = run_buffer_sizes(args.quick, args.seed)
        table = format_buffer_table(rows, env, n)
        print("\n" + table + "\n")
        with open(os.path.join(args.out, "buffer_size_results.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(args.out, "buffer_size_results.md"), "w",
                  encoding="utf-8") as handle:
            handle.write(table + "\n")
        if not args.no_plots:
            save_plots(rows, curves, args.out)

    print(f"saved results to {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
