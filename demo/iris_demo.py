"""
iris_demo.py
============
End-to-end check of NumCompute-Stream on the Iris dataset, runnable from a
terminal. It exercises every module of the streaming framework and finishes
with a list of self-checks, so one run tells you whether everything works.

Run from the repository root (no installation needed)::

    python demo/iris_demo.py                 # saves plots to demo/outputs/
    python demo/iris_demo.py --show          # also opens the plot windows
    python demo/iris_demo.py --chunk-size 15 --buffer-size 45 --seed 3

What it does
------------
1. loads ``demo/data/iris.csv`` with ``io.load_mixed_csv``
2. blanks a few cells (to exercise the Imputer) and splits the rows into
   shuffled chunks that play the role of a stream
3. keeps streaming statistics with ``stats.update_stats`` and compares them
   with the batch values
4. trains ``Imputer -> StandardScaler -> DecisionTreeClassifier`` with
   ``Pipeline.partial_fit`` through a ``StreamTrainer`` (test-then-train),
   printing one log line per chunk
5. trains a second pipeline with a larger ``buffer_size`` for comparison
6. reports streaming metrics (accuracy, macro F1, confusion matrix)
7. draws the plots with ``visualise`` and writes the chunk log to CSV
8. runs self-checks and prints PASS/FAIL for each

Only NumPy, matplotlib and the standard library are used.

"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:               # lets the demo run from a plain checkout
    sys.path.insert(0, _ROOT)

from numcompute_stream import io, metrics, stats  # noqa: E402
from numcompute_stream.pipeline import Pipeline  # noqa: E402
from numcompute_stream.preprocessing import Imputer, StandardScaler  # noqa: E402
from numcompute_stream.stream import StreamTrainer, iter_chunks  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier  # noqa: E402

DATA = os.path.join(_HERE, "data", "iris.csv")
OUT = os.path.join(_HERE, "outputs")


def heading(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def make_pipeline(buffer_size, max_depth=4):
    """Imputer -> StandardScaler -> buffered decision tree."""
    return Pipeline([
        ("impute", Imputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=max_depth, buffer_size=buffer_size)),
    ])


def run_stream(X, y, buffer_size, chunk_size, seed, verbose):
    """Test-then-train one pipeline over the shuffled stream."""
    trainer = StreamTrainer(
        make_pipeline(buffer_size),
        metrics={"macro_f1": metrics.StreamingF1(average="macro"),
                 "confusion": metrics.StreamingConfusionMatrix()},
        window=50, name=f"buffer_size={buffer_size}", verbose=verbose)
    trainer.run(iter_chunks(X, y, chunk_size=chunk_size, shuffle=True, seed=seed))
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description="NumCompute-Stream Iris demo")
    parser.add_argument("--chunk-size", type=int, default=10, help="samples per chunk")
    parser.add_argument("--buffer-size", type=int, default=30,
                        help="buffer_size of the main model (the comparison model uses twice this)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the shuffle and the gaps")
    parser.add_argument("--show", action="store_true", help="open the plot windows")
    parser.add_argument("--no-plots", action="store_true", help="skip plotting")
    args = parser.parse_args(argv)
    checks = []

    # ------------------------------------------------------------ 1. load
    heading("1. Load the CSV with io.load_mixed_csv")
    X, text, names = io.load_mixed_csv(DATA, numeric_cols=(0, 1, 2, 3), header=True)
    y = text[:, 0]
    feature_names = names[:4]
    classes, counts = np.unique(y, return_counts=True)
    print(f"file      : {os.path.relpath(DATA, _ROOT)}")
    print(f"X shape   : {X.shape}   features: {feature_names}")
    print(f"y classes : {dict(zip(classes.tolist(), counts.tolist()))}")
    checks.append(("CSV loaded as 150 x 4 with 3 classes",
                   X.shape == (150, 4) and classes.size == 3))

    # ---------------------------------------------------------- 2. stream
    heading("2. Simulate a stream: blank a few cells, shuffle, cut into chunks")
    rng = np.random.default_rng(args.seed)
    X = X.copy()
    gaps = rng.random(X.shape) < 0.04
    X[gaps] = np.nan
    n_chunks = int(np.ceil(len(X) / args.chunk_size))
    print(f"missing cells injected : {int(gaps.sum())} of {X.size} (handled by the Imputer)")
    print(f"chunks                 : {n_chunks} chunks of {args.chunk_size} samples, "
          f"shuffled with seed {args.seed}")

    # ----------------------------------------------------------- 3. stats
    heading("3. Streaming statistics with stats.update_stats(X_chunk)")
    state = None
    for X_chunk, _ in iter_chunks(X, y, chunk_size=args.chunk_size, shuffle=True,
                                  seed=args.seed):
        state = stats.update_stats(X_chunk, state)
    summary = state.result()
    print(f"{'feature':<14}{'count':>7}{'mean':>9}{'std':>9}{'min':>7}{'median':>9}{'max':>7}")
    for j, name in enumerate(feature_names):
        print(f"{name:<14}{int(summary['count'][j]):>7}{summary['mean'][j]:>9.3f}"
              f"{summary['std'][j]:>9.3f}{summary['min'][j]:>7.1f}"
              f"{summary['quantiles'][0.5][j]:>9.2f}{summary['max'][j]:>7.1f}")
    checks.append(("streaming mean/std equal the batch values",
                   np.allclose(summary["mean"], np.nanmean(X, axis=0))
                   and np.allclose(summary["std"], np.nanstd(X, axis=0))))
    checks.append(("streaming median equals the batch median",
                   np.allclose(summary["quantiles"][0.5], np.nanmedian(X, axis=0))))

    # -------------------------------------------------------- 4. training
    heading(f"4. Pipeline.partial_fit through StreamTrainer (buffer_size={args.buffer_size})")
    print("each chunk is scored first (unseen data) and then used for training;")
    print("'n/a' = no tree yet (buffer still filling), '*' = the tree was retrained\n")
    main_run = run_stream(X, y, args.buffer_size, args.chunk_size, args.seed, verbose=True)
    tree = main_run.model["model"]
    info = main_run.summary()
    expected_retrains = len(X) // args.buffer_size if args.buffer_size % args.chunk_size == 0 else None
    checks.append(("one log record per chunk", len(main_run.history) == n_chunks))
    if expected_retrains is not None:
        checks.append((f"tree retrained every {args.buffer_size} samples "
                       f"({expected_retrains} times)", info["n_retrains"] == expected_retrains))
    checks.append(("buffer is cleared after a retrain",
                   all(r["buffer_count"] == 0 for r in main_run.history if r["retrained"])))
    checks.append(("imputer filled every gap, so the scaler saw all 150 rows per feature",
                   bool(np.all(main_run.model["scale"].n_ == len(X)))))

    # ------------------------------------------------------ 5. comparison
    other_size = args.buffer_size * 2
    heading(f"5. Second model for comparison (buffer_size={other_size})")
    other_run = run_stream(X, y, other_size, args.chunk_size, args.seed, verbose=False)
    print(other_run.format_log(last=3))
    for run in (main_run, other_run):
        s = run.summary()
        print(f"{run.name:<16} retrains={s['n_retrains']:<3} "
              f"cumulative accuracy={s['cumulative_accuracy']:.3f}  "
              f"total fit={s['total_fit_seconds'] * 1e3:6.2f} ms  "
              f"peak memory={s['peak_memory_bytes'] / 1024:5.1f} KiB")

    # --------------------------------------------------------- 6. metrics
    heading("6. Streaming metrics and the current tree")
    cm = main_run.metrics["confusion"]
    print(f"cumulative accuracy : {info['cumulative_accuracy']:.3f}")
    print(f"rolling accuracy    : {info['rolling_accuracy']:.3f}  (last 50 samples)")
    print(f"macro F1            : {info['macro_f1']:.3f}")
    print("confusion matrix (rows = truth, columns = prediction):")
    print(f"{'':<12}" + "".join(f"{str(c):>12}" for c in cm.labels_))
    for label, row in zip(cm.labels_, cm.result()):
        print(f"{str(label):<12}" + "".join(f"{int(v):>12}" for v in row))
    print("\ntree after the last retrain (inputs are the scaled features):")
    print(tree.export_text(feature_names))
    scored = main_run.get_metric("cumulative_accuracy")
    checks.append(("confusion matrix counts every scored sample",
                   int(cm.result().sum()) == cm.n_samples_))
    checks.append(("streaming accuracy equals the confusion-matrix trace",
                   np.isclose(info["cumulative_accuracy"],
                              np.trace(cm.result()) / cm.result().sum())))
    checks.append(("final cumulative accuracy above 0.80",
                   info["cumulative_accuracy"] > 0.80))
    checks.append(("warm-up chunks are logged as not scored (NaN), not as zero",
                   np.isnan(scored[0]) and not main_run.history[0]["scored"]))

    # ----------------------------------------------------------- 7. plots
    log_path = os.path.join(OUT, "stream_log.csv")
    os.makedirs(OUT, exist_ok=True)
    main_run.save_log(log_path)
    saved = [log_path]
    if not args.no_plots:
        heading("7. Plots with visualise.py")
        if not args.show:
            import matplotlib
            matplotlib.use("Agg")       # draw to files only, no window needed
        from numcompute_stream import visualise

        plots = {
            "accuracy_over_time.png": lambda p: visualise.plot_metric_over_time(
                main_run.get_metric("cumulative_accuracy"),
                title=f"Cumulative accuracy over time (buffer_size={args.buffer_size})",
                ylabel="Cumulative accuracy", save_path=p, show=args.show),
            "error_over_time.png": lambda p: visualise.plot_metric_over_time(
                main_run.get_metric("error"),
                title="Error rate of each chunk", ylabel="Chunk error rate",
                save_path=p, show=args.show),
            "compare_buffer_sizes.png": lambda p: visualise.compare_models(
                main_run.get_metric("cumulative_accuracy"),
                other_run.get_metric("cumulative_accuracy"),
                labels=(main_run.name, other_run.name),
                title="Two buffer sizes on the same stream",
                ylabel="Cumulative accuracy", save_path=p, show=args.show),
            "predictions_vs_truth.png": lambda p: visualise.plot_predictions_vs_ground_truth(
                main_run.last_y_true_, main_run.last_y_pred_, save_path=p, show=args.show),
        }
        for filename, draw in plots.items():
            path = os.path.join(OUT, filename)
            draw(path)
            saved.append(path)
        checks.append(("all four plots were written",
                       all(os.path.getsize(p) > 1000 for p in saved[1:])))
    print("files written:")
    for path in saved:
        print(f"  {os.path.relpath(path, _ROOT)}")

    # ---------------------------------------------------------- 8. checks
    heading("8. Self-checks")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = [label for label, ok in checks if not ok]
    print("\n" + ("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
