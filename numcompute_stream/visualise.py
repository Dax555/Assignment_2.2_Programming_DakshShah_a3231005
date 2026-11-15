"""
visualise.py
============
Reusable matplotlib plots for streaming experiments. Every function works
from plain arrays, so it can be called from a script, a notebook or straight
from a ``StreamTrainer`` log.

Required plots
--------------
- ``plot_metric_over_time(metric_values, title, ylabel)``
      one metric (e.g. accuracy) across chunks
- ``compare_models(metric1, metric2, labels)``
      two models on the same streaming metric (e.g. two buffer sizes)
- ``plot_predictions_vs_ground_truth(y_true, y_pred)``
      predictions against the true labels of the latest chunk

Shared options (all functions)
------------------------------
- ``save_path`` : write the figure to this file (PNG, PDF, SVG, ... chosen
  by the extension). Missing folders are created.
- ``show`` : display the figure (inline in a notebook, a window in a
  script). Use ``show=False`` in scripts and tests that only save.
- ``ax`` : draw into an existing axes instead of creating a figure, so plots
  can be combined into one figure.
- Each function returns ``(fig, ax)`` for further customisation.

Conventions
-----------
- Inputs are 1-D. NaN in a metric means "no value for this chunk" (a model
  that had not been trained yet) and is drawn as a gap, not as zero.
- Colour is never the only cue: series differ in line style and marker as
  well, and every figure with two series has a legend.

Only matplotlib and NumPy are used.

"""

import os

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]

# colour-blind-safe pair for two series, plus neutral inks for text and grid
_SERIES = ("#2a78d6", "#eb6834")
_WRONG = "#d03b3b"
_INK = "#0b0b0b"
_INK_SOFT = "#52514e"
_GRID = "#e4e3df"
_SURFACE = "#fcfcfb"
_LINE_STYLES = ("-", "--")
_MARKERS = ("o", "s")


# ------------------------------------------------------------------ helpers
def _as_series(values, name):
    """Validate one metric series.

    Parameters
    ----------
    values : array_like, shape (n,)
        Numeric values; NaN allowed.
    name : str
        Argument name for error messages.

    Returns
    -------
    np.ndarray of float, shape (n,)

    Raises
    ------
    ValueError
        If `values` is not numeric, not 1-D, or empty.

    Notes
    -----
    Time: O(n). Space: O(n).
    """
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric ({exc})") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty; there is nothing to plot")
    return arr


def _new_axes(ax, figsize):
    """Return (fig, ax), creating a styled figure when `ax` is None.

    Notes
    -----
    Time: O(1). Space: O(1).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=_SURFACE)
    else:
        fig = ax.figure
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_SOFT, labelsize=9)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _label(ax, title, xlabel, ylabel):
    """Set the title and axis labels in the shared text style."""
    ax.set_title(title, color=_INK, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, color=_INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=_INK_SOFT, fontsize=10)


def _finish(fig, save_path, show):
    """Save and/or show a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    save_path : str, path-like or None
        File to write; its folder is created if needed.
    show : bool
        Whether to call ``plt.show()``.

    Raises
    ------
    OSError
        If the file cannot be written.

    Notes
    -----
    The figure stays open after saving so the caller can keep editing it;
    close it with ``matplotlib.pyplot.close(fig)`` when making many plots.
    """
    fig.tight_layout()
    if save_path is not None:
        folder = os.path.dirname(os.fspath(save_path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    if show:
        plt.show()


def _draw_series(ax, x, values, index, label, nudge=0):
    """Draw one metric line with its style, and label its last value.

    Parameters
    ----------
    nudge : int, default 0
        Vertical offset of the end label in points, used to keep the end
        labels of two series apart when their last values are close.

    Notes
    -----
    Markers are only drawn for short series, where single chunks matter.
    The horizontal axis always starts at the first chunk, so warm-up chunks
    without a value show as an empty stretch instead of being cut off. On a
    log axis the margin is a factor, so the limit never reaches zero.
    """
    marker = _MARKERS[index] if values.size <= 40 else None
    ax.plot(x, values, color=_SERIES[index], linestyle=_LINE_STYLES[index],
            linewidth=2, marker=marker, markersize=5, label=label)
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size:
        last = valid[-1]
        ax.annotate(f"{values[last]:.3f}", (x[last], values[last]),
                    textcoords="offset points", xytext=(6, nudge), va="center",
                    fontsize=9, color=_INK)
    if ax.get_xscale() == "log":
        left = x[0] / (x[-1] / x[0]) ** 0.03 if x[-1] > x[0] else x[0] / 2
    else:
        left = x[0] - 0.02 * (x[-1] - x[0] + 1)
    ax.set_xlim(left=min(ax.get_xlim()[0], left))


# -------------------------------------------------------------------- plots
def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Value",
                          xlabel="Chunk", x=None, xscale="linear", save_path=None,
                          show=True, ax=None):
    """Plot one metric across the chunks of a stream.

    Parameters
    ----------
    metric_values : array_like, shape (n_chunks,)
        Metric value after each chunk; NaN leaves a gap.
    title : str, default "Metric over time"
        Figure title.
    ylabel : str, default "Value"
        Name of the metric, e.g. "Cumulative accuracy".
    xlabel : str, default "Chunk"
        Label of the horizontal axis.
    x : array_like, shape (n_chunks,), optional
        Horizontal positions, e.g. samples seen. Default: 1, 2, ..., n_chunks.
    xscale : {"linear", "log"}, default "linear"
        Scale of the horizontal axis. Use "log" when `x` spans several orders
        of magnitude, e.g. buffer sizes from 100 to 10,000.
    save_path : str or path-like, optional
        Save the figure to this file.
    show : bool, default True
        Display the figure.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a new figure.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)

    Raises
    ------
    ValueError
        If `metric_values` is not numeric, not 1-D or empty, `x` has a
        different length, `xscale` is unknown, or `xscale="log"` is used with
        an `x` that is not positive and increasing.

    Notes
    -----
    Time: O(n_chunks). Space: O(n_chunks).

    Examples
    --------
    >>> fig, ax = plot_metric_over_time([0.5, 0.7, 0.8], "Accuracy", "Accuracy",
    ...                                 show=False)
    """
    values = _as_series(metric_values, "metric_values")
    x = np.arange(1, values.size + 1) if x is None else _as_series(x, "x")
    if x.size != values.size:
        raise ValueError(
            f"x and metric_values must have the same length, got {x.size} and {values.size}"
        )
    if xscale not in ("linear", "log"):
        raise ValueError(f"xscale must be 'linear' or 'log', got {xscale!r}")
    if xscale == "log" and not (np.all(x > 0) and np.all(np.diff(x) > 0)):
        raise ValueError("xscale='log' needs x values that are positive and increasing")
    fig, ax = _new_axes(ax, (7.5, 4.0))
    ax.set_xscale(xscale)          # before drawing, so the axis margin suits the scale
    if xscale == "log":
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:,.0f}" for v in x])
        ax.minorticks_off()
    _draw_series(ax, x, values, 0, None)
    _label(ax, title, xlabel, ylabel)
    _finish(fig, save_path, show)
    return fig, ax


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   title="Model comparison", ylabel="Value", xlabel="Chunk",
                   save_path=None, show=True, ax=None):
    """Compare two models on the same streaming metric.

    Parameters
    ----------
    metric1, metric2 : array_like, shape (n1,) and (n2,)
        The metric after each chunk for each model, e.g. the cumulative
        accuracy of two trees with different ``buffer_size``. NaN leaves a
        gap. The lengths may differ; each series is drawn against its own
        chunk numbers.
    labels : sequence of two str, default ("Model 1", "Model 2")
        Legend names of the two models.
    title : str, default "Model comparison"
    ylabel : str, default "Value"
    xlabel : str, default "Chunk"
    save_path : str or path-like, optional
        Save the figure to this file.
    show : bool, default True
        Display the figure.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a new figure.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)

    Raises
    ------
    ValueError
        If a series is not numeric, not 1-D or empty, or `labels` does not
        hold exactly two names.

    Notes
    -----
    Both series share one axis, so they must be the same metric. The models
    differ in colour, line style and marker, and a legend is always drawn.
    Time: O(n1 + n2). Space: O(n1 + n2).

    Examples
    --------
    >>> fig, ax = compare_models([0.5, 0.6], [0.4, 0.7], ("small", "large"),
    ...                          show=False)
    """
    if isinstance(labels, str) or len(labels) != 2:
        raise ValueError(f"labels must hold exactly two names, got {labels!r}")
    series = (_as_series(metric1, "metric1"), _as_series(metric2, "metric2"))
    fig, ax = _new_axes(ax, (7.5, 4.0))
    # if the two lines end close together, push their end labels apart
    last = [v[~np.isnan(v)][-1] if (~np.isnan(v)).any() else np.nan for v in series]
    both = np.concatenate(series)
    spread = np.nanmax(both) - np.nanmin(both) if (~np.isnan(both)).any() else 0.0
    close = not np.isnan(last).any() and abs(last[0] - last[1]) <= 0.08 * spread
    nudges = (0, 0) if not close else ((6, -6) if last[0] >= last[1] else (-6, 6))
    for index, (values, label) in enumerate(zip(series, labels)):
        _draw_series(ax, np.arange(1, values.size + 1), values, index, str(label),
                     nudge=nudges[index])
    _label(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)
    _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, title=None, save_path=None,
                                     show=True, ax=None):
    """Show predictions against the true labels of one chunk, sample by sample.

    Each sample is one position on the horizontal axis. Its true class is an
    open circle and its predicted class a filled dot, so a correct prediction
    is a dot inside a circle and a wrong one is a red cross away from its
    circle.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels of the chunk (any type).
    y_pred : array_like, shape (n,)
        Predicted labels.
    title : str, optional
        Default: "Predictions vs ground truth" with the chunk accuracy.
    save_path : str or path-like, optional
        Save the figure to this file.
    show : bool, default True
        Display the figure.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a new figure.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)

    Raises
    ------
    ValueError
        If the inputs are not 1-D, are empty, or differ in length.

    Notes
    -----
    Classes are placed on the vertical axis in sorted order, so string
    labels work. Meant for one chunk (tens to a few hundred samples).
    Time: O(n log n) for the class lookup. Space: O(n).

    Examples
    --------
    >>> fig, ax = plot_predictions_vs_ground_truth(["a", "b", "b"], ["a", "b", "a"],
    ...                                            show=False)
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_true.size == 0:
        raise ValueError("y_true is empty; there is nothing to plot")
    if y_true.size != y_pred.size:
        raise ValueError(
            f"y_true and y_pred must have the same length, got {y_true.size} and {y_pred.size}"
        )
    classes, codes = np.unique(np.concatenate([y_true, y_pred]), return_inverse=True)
    true_pos, pred_pos = codes[:y_true.size], codes[y_true.size:]
    correct = true_pos == pred_pos
    index = np.arange(1, y_true.size + 1)
    n_wrong = int((~correct).sum())

    fig, ax = _new_axes(ax, (8.0, 1.6 + 0.6 * classes.size))
    ax.scatter(index, true_pos, s=90, facecolors="none", edgecolors=_INK_SOFT,
               linewidths=1.2, label="ground truth", zorder=2)
    ax.scatter(index[correct], pred_pos[correct], s=22, color=_SERIES[0],
               label=f"correct prediction ({int(correct.sum())})", zorder=3)
    if n_wrong:
        ax.scatter(index[~correct], pred_pos[~correct], s=55, color=_WRONG, marker="x",
                   linewidths=2, label=f"wrong prediction ({n_wrong})", zorder=3)
    ax.set_yticks(np.arange(classes.size))
    ax.set_yticklabels([str(c) for c in classes])
    ax.set_ylim(-0.6, classes.size - 0.4)
    if title is None:
        title = f"Predictions vs ground truth (accuracy {correct.mean():.3f})"
    _label(ax, title, "Sample in chunk", "Class")
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=3)
    _finish(fig, save_path, show)
    return fig, ax
