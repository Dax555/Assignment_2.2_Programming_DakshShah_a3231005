# tests for visualise.py (drawn off-screen with the Agg backend)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from numcompute_stream import visualise  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_metric_over_time_draws_values_and_labels():
    values = [0.5, 0.7, np.nan, 0.9]            # NaN = chunk without a score
    fig, ax = visualise.plot_metric_over_time(values, "Accuracy", "Acc", show=False)
    line = ax.get_lines()[0]
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3, 4])
    np.testing.assert_array_equal(line.get_ydata(), values)
    assert ax.get_title(loc="left") == "Accuracy" and ax.get_ylabel() == "Acc"
    assert ax.get_legend() is None              # a single series needs no legend


def test_plot_metric_over_time_custom_x_and_validation():
    fig, ax = visualise.plot_metric_over_time([1.0, 2.0], x=[100, 200], show=False)
    np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), [100, 200])
    with pytest.raises(ValueError, match="empty"):
        visualise.plot_metric_over_time([], show=False)
    with pytest.raises(ValueError, match="1-D"):
        visualise.plot_metric_over_time(np.ones((2, 2)), show=False)
    with pytest.raises(ValueError, match="same length"):
        visualise.plot_metric_over_time([1.0, 2.0], x=[1], show=False)
    with pytest.raises(ValueError, match="numeric"):
        visualise.plot_metric_over_time(["a", "b"], show=False)


def test_compare_models_has_two_series_and_a_legend():
    fig, ax = visualise.compare_models([0.5, 0.6, 0.7], [0.4, 0.8],
                                       labels=("buffer 50", "buffer 500"), show=False)
    assert len(ax.get_lines()) == 2             # different lengths are fine
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ["buffer 50", "buffer 500"]
    styles = {line.get_linestyle() for line in ax.get_lines()}
    assert len(styles) == 2                     # not told apart by colour alone
    with pytest.raises(ValueError, match="two names"):
        visualise.compare_models([1.0], [2.0], labels=("only one",), show=False)


def test_predictions_vs_ground_truth_marks_errors():
    y_true = np.array(["a", "b", "b", "c", "a"])
    y_pred = np.array(["a", "b", "c", "c", "b"])
    fig, ax = visualise.plot_predictions_vs_ground_truth(y_true, y_pred, show=False)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "correct prediction (3)" in labels and "wrong prediction (2)" in labels
    assert "0.600" in ax.get_title(loc="left")
    with pytest.raises(ValueError, match="same length"):
        visualise.plot_predictions_vs_ground_truth([1, 2], [1], show=False)
    with pytest.raises(ValueError, match="empty"):
        visualise.plot_predictions_vs_ground_truth([], [], show=False)


def test_save_to_file_creates_folders(tmp_path):
    target = tmp_path / "plots" / "nested" / "acc.png"
    visualise.plot_metric_over_time([0.1, 0.2], save_path=target, show=False)
    assert target.exists() and target.stat().st_size > 1000
    pdf = tmp_path / "cmp.pdf"
    visualise.compare_models([0.1], [0.2], save_path=pdf, show=False)
    assert pdf.read_bytes().startswith(b"%PDF")


def test_plots_can_share_one_figure():
    fig, axes = plt.subplots(1, 2)
    out_fig, out_ax = visualise.plot_metric_over_time([1.0, 2.0], ax=axes[0], show=False)
    visualise.compare_models([1.0], [2.0], ax=axes[1], show=False)
    assert out_fig is fig and out_ax is axes[0]
    assert len(axes[1].get_lines()) == 2
