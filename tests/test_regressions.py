# Regression tests for the issues raised in the Assignment 2.1 feedback:
#   1. pairwise_distances lost close pairs that are far from the centre
#   2. OneHotEncoder treated the integer 1 and the float 1.0 as different
#   3. the package must import from a clean checkout (no credentials, no
#      hidden dependencies) and install from the repository root
import os
import subprocess
import sys

import numpy as np
import pytest

from numcompute_stream.preprocessing import OneHotEncoder
from numcompute_stream.utils import pairwise_distances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "numcompute_stream")


def exact_distances(X, Y):
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))


# ----------------------------------------------------------------- distances
def test_distance_with_widely_separated_reference_points():
    # the grader's case: the centre of Y is near 0, far from both points
    D = pairwise_distances([[1e9]], [[1e9 + 1], [-1e9]])
    np.testing.assert_allclose(D, [[1.0, 2e9]], rtol=1e-12)
    D2 = pairwise_distances([[1e9]], [[1e9 + 1], [-1e9]], metric="sqeuclidean")
    np.testing.assert_allclose(D2, [[1.0, 4e18]], rtol=1e-12)


def test_distance_close_pairs_inside_far_apart_clusters():
    rng = np.random.default_rng(0)
    offsets = np.array([[1e8, -1e8, 1e8], [-1e8, 1e8, 1e8], [0.0, 0.0, -3e8]])
    X = offsets[rng.integers(0, 3, size=40)] + rng.normal(size=(40, 3))
    Y = offsets[rng.integers(0, 3, size=30)] + rng.normal(size=(30, 3))
    np.testing.assert_allclose(pairwise_distances(X, Y), exact_distances(X, Y), rtol=1e-9)
    np.testing.assert_allclose(pairwise_distances(X), exact_distances(X, X),
                               rtol=1e-9, atol=1e-12)


def test_distance_identical_far_points_are_exactly_zero():
    X = np.array([[1e12, -1e12], [3.0, 4.0]])
    D = pairwise_distances(X, X.copy())
    assert D[0, 0] == 0.0 and D[1, 1] == 0.0
    assert pairwise_distances(X)[0, 0] == 0.0


def test_distance_ordinary_data_unchanged():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(50, 4)), rng.normal(size=(20, 4))
    np.testing.assert_allclose(pairwise_distances(X, Y), exact_distances(X, Y), rtol=1e-10)


# ------------------------------------------------------------ one-hot dtypes
def test_one_hot_integer_fit_float_transform():
    # the grader's case: fit on ints 1 and 2, transform floats 1.0 and 2.0
    enc = OneHotEncoder().fit(np.array([[1], [2]]))
    np.testing.assert_array_equal(enc.transform(np.array([[1.0], [2.0]])),
                                  [[1.0, 0.0], [0.0, 1.0]])


def test_one_hot_float_fit_integer_transform_and_streaming_dtype_change():
    enc = OneHotEncoder().fit(np.array([[1.0], [2.0]]))
    np.testing.assert_array_equal(enc.transform(np.array([[2], [1]])),
                                  [[0.0, 1.0], [1.0, 0.0]])
    stream = OneHotEncoder()
    stream.partial_fit(np.array([[1], [2]]))             # chunk 1 arrives as int
    stream.partial_fit(np.array([[2.0], [3.0]]))         # chunk 2 arrives as float
    assert stream.categories_[0].tolist() == ["1", "2", "3"]   # no duplicate "2.0"


def test_one_hot_numeric_edge_values():
    enc = OneHotEncoder().fit(np.array([[1.0], [2.5], [-3.0], [np.nan]]))
    assert enc.categories_[0].tolist() == ["-3", "1", "2.5"]   # NaN is not a category
    out = enc.transform(np.array([[2.5], [np.nan], [-3]], dtype=float))
    np.testing.assert_array_equal(out, [[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    mixed = OneHotEncoder(categories=[[1, 2]]).transform(np.array([[1.0], [2.0]]))
    np.testing.assert_array_equal(mixed, np.eye(2))
    objects = OneHotEncoder().fit(np.array([[1], [2.0], ["x"]], dtype=object))
    assert objects.categories_[0].tolist() == ["1", "2", "x"]


def test_one_hot_text_codes_stay_distinct():
    # documented: text is compared as text, so "1" and "1.0" are two labels
    enc = OneHotEncoder().fit(np.array([["1"], ["1.0"]]))
    assert enc.n_output_features_ == 2


# ------------------------------------------------- clean import and install
def test_clean_import_smoke_check(tmp_path):
    # a fresh interpreter, started outside the repository, with only the
    # repository root on the path: every module must import and run
    code = (
        "import numpy as np, numcompute_stream as ncs;"
        "from numcompute_stream import (io, preprocessing, sort_search, rank, stats,"
        " metrics, optim, pipeline, utils, benchmarking, tree, stream, visualise);"
        "t = ncs.DecisionTreeClassifier(buffer_size=2)"
        ".partial_fit(np.array([[0.0], [1.0]]), [0, 1]);"
        "print(ncs.__version__, t.predict([[1.0]])[0])"
    )
    env = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND="Agg")
    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["0.2.0", "1"]


def test_only_numpy_and_matplotlib_are_imported():
    banned = ("sklearn", "pandas", "scipy", "torch", "tensorflow", "seaborn")
    for name in sorted(os.listdir(PACKAGE)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(PACKAGE, name)) as handle:
            for number, line in enumerate(handle, 1):
                words = line.strip().split()
                if words[:1] in (["import"], ["from"]):
                    module = words[1].split(".")[0]
                    assert module not in banned, f"{name}:{number} imports {module}"


def test_packaging_installs_from_the_repository_root():
    # the 2.1 README told readers to install from the root while
    # pyproject.toml sat one folder down; keep both in the same place
    pyproject = os.path.join(ROOT, "pyproject.toml")
    assert os.path.isfile(pyproject), "pyproject.toml must be at the repository root"
    text = open(pyproject).read()
    assert 'name = "numcompute-stream"' in text
    assert "numcompute_stream" in text
    assert os.path.isfile(os.path.join(PACKAGE, "__init__.py"))
    for module in ("io", "preprocessing", "stats", "metrics", "pipeline",
                   "tree", "stream", "visualise"):
        assert os.path.isfile(os.path.join(PACKAGE, module + ".py"))


def test_demo_assets_exist_for_a_local_run():
    # the demo must run from a local checkout, without GitHub credentials
    assert os.path.isfile(os.path.join(ROOT, "demo", "data", "iris.csv"))
    notebook = os.path.join(ROOT, "demo", "stream_demo.ipynb")
    if not os.path.isfile(notebook):
        pytest.skip("notebook not present in this checkout")
    text = open(notebook).read()
    assert "git clone" not in text and "getpass" not in text and "TOKEN" not in text


# ------------------------------------------------------- benchmark results
def test_write_csv_writes_the_rows_not_only_the_header(tmp_path):
    # the 2.1 results file contained a header and no data
    from numcompute_stream import benchmarking

    row = benchmarking.compare("sum", 100, lambda a: sum(a.tolist()), lambda a: a.sum(),
                               (np.arange(100.0),), lambda x, y: np.isclose(x, y), repeat=2)
    assert row["ok"] and row["loop_ms"] > 0 and row["vec_ms"] > 0
    path = tmp_path / "results.csv"
    benchmarking.write_csv([row, row], path)
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("case,n,baseline")
    assert lines[1].startswith("sum,100,")


def test_committed_benchmark_results_are_populated():
    results = os.path.join(ROOT, "benchmark", "results")
    for name in ("benchmark_results.csv", "stream_loop_vs_vectorised.csv",
                 "buffer_size_results.csv"):
        path = os.path.join(results, name)
        assert os.path.isfile(path), f"{name} is missing; run the benchmark scripts"
        with open(path) as handle:
            assert len(handle.read().splitlines()) > 1, f"{name} has no data rows"
