# tests for io.py

import numpy as np
import pytest

from numcompute_stream.io import iter_csv, load_csv, load_mixed_csv

@pytest.fixture
def write_csv(tmp_path):
    """Write text to a temp file and return its path."""

    def _write(text, name="data.csv", encoding="utf-8"):
        path = tmp_path / name
        path.write_text(text, encoding=encoding)
        return path

    return _write

# basic loading

def test_loads_clean_numeric_file(write_csv):
    path = write_csv("1,2,3\n4,5,6\n")
    X = load_csv(path)
    assert X.shape == (2, 3)
    assert X.dtype == np.float64
    np.testing.assert_array_equal(X, [[1, 2, 3], [4, 5, 6]])


def test_tab_delimiter(write_csv):
    path = write_csv("1\t2\n3\t4\n", name="data.tsv")
    np.testing.assert_array_equal(load_csv(path, delimiter="\t"), [[1, 2], [3, 4]])


def test_whitespace_delimiter_none(write_csv):
    path = write_csv("1  2\n3 \t 4\n")
    np.testing.assert_array_equal(load_csv(path, delimiter=None), [[1, 2], [3, 4]])


def test_header_returns_names_and_excludes_them_from_data(write_csv):
    path = write_csv("a, b\n1,2\n3,4\n")
    X, names = load_csv(path, header=True)
    assert names == ["a", "b"]
    np.testing.assert_array_equal(X, [[1, 2], [3, 4]])


def test_usecols_selects_columns_and_names(write_csv):
    path = write_csv("a,b,c\n1,2,3\n4,5,6\n")
    X, names = load_csv(path, header=True, usecols=[0, 2])
    assert names == ["a", "c"]
    np.testing.assert_array_equal(X, [[1, 3], [4, 6]])


def test_text_column_with_usecols(write_csv):
    path = write_csv("Adelaide,1\nSydney,2\n")
    cities = load_csv(path, dtype=str, usecols=[0])
    assert cities.shape == (2, 1)
    assert cities[:, 0].tolist() == ["Adelaide", "Sydney"]


# always 2D


def test_single_column_is_column_vector(write_csv):
    assert load_csv(write_csv("1\n2\n3\n")).shape == (3, 1)


def test_single_row_is_row_vector(write_csv):
    assert load_csv(write_csv("1,2,3\n")).shape == (1, 3)


def test_single_value_is_1x1(write_csv):
    X = load_csv(write_csv("7\n"))
    assert X.shape == (1, 1) and X[0, 0] == 7


def test_header_only_file_returns_empty_with_columns(write_csv):
    X, names = load_csv(write_csv("a,b,c\n"), header=True)
    assert X.shape == (0, 3)
    assert names == ["a", "b", "c"]

# missing values: fill

def test_empty_fields_become_nan(write_csv):
    X = load_csv(write_csv("1,\n,4\n"))
    assert np.isnan(X[0, 1]) and np.isnan(X[1, 0])
    assert X[0, 0] == 1 and X[1, 1] == 4


def test_default_markers_recognised_in_every_column(write_csv):
    # markers have to work in any column, not just one column each
    X = load_csv(write_csv("?,NA\nNA,?\nN/A,nan\n"))
    assert np.isnan(X).all()


def test_custom_missing_values(write_csv):
    X = load_csv(write_csv("1,-\n-,4\n"), missing_values=["-"])
    assert np.isnan(X[0, 1]) and np.isnan(X[1, 0])


def test_custom_fill_value(write_csv):
    X = load_csv(write_csv("1,NA\n3,4\n"), fill_value=-99.0)
    np.testing.assert_array_equal(X, [[1, -99], [3, 4]])


def test_all_values_missing(write_csv):
    X = load_csv(write_csv(",\n,\n"))
    assert X.shape == (2, 2) and np.isnan(X).all()


def test_text_gaps_default_to_empty_string(write_csv):
    X = load_csv(write_csv("a,b\nc,\n"), dtype=str)
    assert X[1, 1] == ""


def test_text_fill_value_is_not_truncated(write_csv):
    # longest value is 1 character, "missing" must not be cut to "m"
    X = load_csv(write_csv("a,b\nc,\n"), dtype=str, fill_value="missing")
    assert X[1, 1] == "missing"


def test_integer_dtype_without_gaps(write_csv):
    X = load_csv(write_csv("1,2\n3,4\n"), dtype=int)
    assert np.issubdtype(X.dtype, np.integer)
    np.testing.assert_array_equal(X, [[1, 2], [3, 4]])


def test_integer_gap_with_integer_fill_value(write_csv):
    X = load_csv(write_csv("1,2\n3,\n"), dtype=int, fill_value=-1)
    np.testing.assert_array_equal(X, [[1, 2], [3, -1]])


def test_skip_drops_only_rows_with_gaps(write_csv):
    X = load_csv(write_csv("1,2\n3,NA\n?,6\n7,8\n"), missing="skip")
    np.testing.assert_array_equal(X, [[1, 2], [7, 8]])


def test_skip_every_row_gives_empty_2d(write_csv):
    assert load_csv(write_csv("1,\n,2\n"), missing="skip").shape == (0, 2)


def test_skip_integer_column_with_gap(write_csv):
    X = load_csv(write_csv("1,2\n3,\n"), dtype=int, missing="skip")
    np.testing.assert_array_equal(X, [[1, 2]])

# awkward content

def test_hash_is_not_treated_as_comment(write_csv):
    X = load_csv(write_csv("x,2#note\n"), dtype=str)
    assert X[0, 1] == "2#note"


def test_windows_line_endings(write_csv):
    X, names = load_csv(write_csv("a,b\r\n1,2\r\n"), header=True)
    assert names == ["a", "b"]
    np.testing.assert_array_equal(X, [[1, 2]])


def test_negative_and_scientific_numbers(write_csv):
    X = load_csv(write_csv("-1.5,2e3\n0,-0.001\n"))
    np.testing.assert_allclose(X, [[-1.5, 2000.0], [0.0, -0.001]])


def test_accepts_pathlib_path(write_csv):
    assert load_csv(write_csv("1,2\n")).shape == (1, 2)

# errors

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file"):
        load_csv(tmp_path / "nope.csv")


def test_empty_file_raises(write_csv):
    with pytest.raises(ValueError, match="empty"):
        load_csv(write_csv(""))


def test_blank_lines_only_file_raises(write_csv):
    with pytest.raises(ValueError, match="empty"):
        load_csv(write_csv("\n\n  \n"))


def test_invalid_strategy_raises(write_csv):
    with pytest.raises(ValueError, match="fill"):
        load_csv(write_csv("1,2\n"), missing="drop")


def test_empty_delimiter_raises(write_csv):
    with pytest.raises(ValueError, match="empty string"):
        load_csv(write_csv("1,2\n"), delimiter="")


def test_non_string_delimiter_raises(write_csv):
    with pytest.raises(TypeError, match="delimiter"):
        load_csv(write_csv("1,2\n"), delimiter=5)


def test_wrong_column_count_raises(write_csv):
    with pytest.raises(ValueError, match="Could not parse"):
        load_csv(write_csv("1,2\n3,4,5\n"))


def test_non_numeric_text_raises_with_location(write_csv):
    with pytest.raises(ValueError, match=r"line 2, column 1"):
        load_csv(write_csv("1,2\n3,abc\n"))


def test_non_numeric_line_number_counts_header(write_csv):
    with pytest.raises(ValueError, match=r"line 3, column 1"):
        load_csv(write_csv("a,b\n1,2\n3,abc\n"), header=True)


def test_integer_gap_without_fill_value_raises(write_csv):
    with pytest.raises(ValueError, match="integer data cannot hold NaN"):
        load_csv(write_csv("1,2\n3,\n"), dtype=int)


def test_integer_dtype_with_nan_fill_raises(write_csv):
    with pytest.raises(ValueError, match="Cannot fill integer data with NaN"):
        load_csv(write_csv("1,2\n"), dtype=int, fill_value=np.nan)



# blank lines at the top

def test_blank_lines_before_header_are_skipped(write_csv):
    data, names = load_csv(write_csv("\n\na,b\n1,2\n"), header=True)
    assert names == ["a", "b"]
    np.testing.assert_array_equal(data, [[1.0, 2.0]])


def test_error_line_number_counts_leading_blank_lines(write_csv):
    with pytest.raises(ValueError, match=r"line 4, column 1"):
        load_csv(write_csv("\na,b\n1,2\n3,x\n"), header=True)

# iter_csv

@pytest.mark.parametrize("chunk_size", [1, 2, 3, 10])
def test_iter_csv_chunks_join_back_to_load_csv(write_csv, chunk_size):
    path = write_csv("1,2\n3,\n5,6\n7,8\n9,10\n")
    chunks = list(iter_csv(path, chunk_size=chunk_size))
    assert all(chunk.shape[0] <= chunk_size for chunk in chunks)
    np.testing.assert_array_equal(np.vstack(chunks), load_csv(path))


def test_iter_csv_gives_names_with_every_chunk(write_csv):
    path = write_csv("x,y\n1,2\n3,4\n5,6\n")
    out = list(iter_csv(path, chunk_size=2, header=True))
    assert len(out) == 2
    assert all(names == ["x", "y"] for _, names in out)
    np.testing.assert_array_equal(out[1][0], [[5.0, 6.0]])


def test_iter_csv_does_not_yield_chunks_with_every_row_skipped(write_csv):
    path = write_csv("1,2\n,4\n5,\n7,8\n")
    chunks = list(iter_csv(path, chunk_size=2, missing="skip"))
    assert len(chunks) == 2
    np.testing.assert_array_equal(np.vstack(chunks), [[1, 2], [7, 8]])


def test_iter_csv_bad_chunk_size_raises(write_csv):
    path = write_csv("1,2\n")
    with pytest.raises(ValueError, match="at least 1"):
        next(iter_csv(path, chunk_size=0))
    with pytest.raises(TypeError, match="chunk_size"):
        next(iter_csv(path, chunk_size=2.5))


def test_iter_csv_column_count_change_between_chunks_raises(write_csv):
    path = write_csv("1,2\n3,4\n5,6,7\n")
    with pytest.raises(ValueError, match="Expected 2 columns"):
        list(iter_csv(path, chunk_size=2))


def test_iter_csv_empty_file_raises(write_csv):
    with pytest.raises(ValueError, match="empty"):
        next(iter_csv(write_csv("\n\n")))

# load_mixed_csv

def test_load_mixed_csv_splits_numbers_and_text(write_csv):
    path = write_csv("name,age,city\nann,30,adl\nbob,,syd\n")
    numbers, text, names = load_mixed_csv(path, [1], header=True)
    assert names == ["name", "age", "city"]
    np.testing.assert_array_equal(numbers, [[30.0], [np.nan]])
    np.testing.assert_array_equal(text, [["ann", "adl"], ["bob", "syd"]])


def test_load_mixed_csv_skip_keeps_rows_lined_up(write_csv):
    path = write_csv("ann,30,adl\nbob,,syd\n,40,mel\ncal,50,per\n")
    numbers, text = load_mixed_csv(path, [1], missing="skip")
    np.testing.assert_array_equal(numbers, [[30.0], [50.0]])
    np.testing.assert_array_equal(text, [["ann", "adl"], ["cal", "per"]])


def test_load_mixed_csv_all_numeric_gives_empty_text(write_csv):
    numbers, text = load_mixed_csv(write_csv("1,2\n3,4\n"), [0, 1])
    assert numbers.shape == (2, 2) and text.shape == (2, 0)


@pytest.mark.parametrize("cols, error", [([3], ValueError), ([0, 0], ValueError), (["a"], TypeError)])
def test_load_mixed_csv_bad_numeric_cols_raise(write_csv, cols, error):
    with pytest.raises(error):
        load_mixed_csv(write_csv("1,2\n3,4\n"), cols)