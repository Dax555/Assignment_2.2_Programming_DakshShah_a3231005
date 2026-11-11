"""
io.py
Reads CSV files and converts them into NumPy arrays.
load_csv reads a delimited text file (comma, tab or whitespace) and always returns a 2D array of shape (n_rows, n_cols).
Missing values can be filled in or the rows containing them skipped.
iter_csv reads the same files in chunks, so large files don't have to fit in memory.
load_mixed_csv loads number columns and text columns from one file in a single read.
Scaling and encoding are left to preprocessing.py.
"""

import os
from itertools import islice

import numpy as np

__all__ = ["load_csv", "iter_csv", "load_mixed_csv"]
DEFAULT_MISSING = ("", "NA", "N/A", "NaN", "nan", "?")
VALID_STRATEGIES = ("fill", "skip")


def _validate_args(path, delimiter, missing):
    if missing not in VALID_STRATEGIES:
        raise ValueError(f"Missing must be one of {VALID_STRATEGIES} got {missing!r}")
    if delimiter is not None and not isinstance(delimiter, str):
        raise TypeError(
            f"delimiter must be a string or None, got {type(delimiter).__name__}"
        )
    if delimiter == "":
        raise ValueError("delimiter cannot be an empty string")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {os.fspath(path)!r}")


def _check_chunk_size(chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError(
            f"chunk_size must be an integer, got {type(chunk_size).__name__}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")


def _resolve_dtype(dtype, fill_value):
    dtype = np.dtype(dtype)
    if fill_value is None:
        if np.issubdtype(dtype, np.floating):
            fill_value = np.nan
        elif np.issubdtype(dtype, np.str_):
            fill_value = ""
    is_nan = isinstance(fill_value, float) and np.isnan(fill_value)
    if np.issubdtype(dtype, np.integer) and is_nan:
        raise ValueError(
            "Cannot fill integer data with NaN, use dtype = float or give an integer fill_value"
        )
    return dtype, fill_value


def _split_header(line, delimiter, usecols):
    text = line.rstrip("\r\n")
    names = text.split() if delimiter is None else text.split(delimiter)
    names = [name.strip() for name in names]
    if usecols is not None:
        names = [names[i] for i in usecols]
    return names


def _take_header(lines, path, delimiter, header, usecols):
    # skip blank lines at the top, so a header after them is still found
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        raise ValueError(f"file is empty {os.fspath(path)!r}")
    names = None
    if header:
        names = _split_header(lines[first], delimiter, usecols)
        first += 1
    return lines[first:], names, first + 1


def _count_cols(names, usecols):
    if names is not None:
        return len(names)
    if usecols is not None:
        return len(usecols)
    return None


def _parse_lines(
    lines,
    *,
    delimiter,
    missing,
    fill_value,
    markers,
    dtype,
    usecols,
    n_cols,
    start_line,
    return_mask=False,
):
    if not any(line.strip() for line in lines):
        empty = np.empty((0, n_cols or 0), dtype=dtype)
        return (empty, np.zeros(empty.shape, dtype=bool)) if return_mask else empty
    try:
        parsed = np.genfromtxt(
            lines,
            delimiter=delimiter,
            dtype=dtype,
            usecols=usecols,
            # a plain list gets read as one marker per column,the {None: ...} form applies them to every column
            missing_values={None: list(markers)},
            usemask=True,  # tells us where the gaps were, for any dtype
            ndmin=2,  # one-row/one-column files would come back 1D otherwise
            comments=None,  # don't cut values at '#'
            encoding="utf-8",
        )
    except ValueError as err:
        raise ValueError(
            f"Could not parse data starting at file line {start_line}:{err}"
        ) from err
    is_missing = np.ma.getmaskarray(parsed)
    values = np.ma.getdata(parsed)
    # genfromtxt turns text like "abc" into NaN without marking it as missing,so any NaN that isn't a real gap means the file has bad data
    if np.issubdtype(dtype, np.floating):
        invalid = np.isnan(values) & ~is_missing
        if invalid.any():
            row, col = np.argwhere(invalid)[0]
            raise ValueError(
                f"Non-numeric value at file line {start_line+row}, column {col}"
            )
    if missing == "skip":
        row_has_gap = is_missing.any(axis=1)
        if return_mask:
            return values[~row_has_gap], is_missing[~row_has_gap]
        return values[~row_has_gap]
    if is_missing.any():
        if fill_value is None:
            row, col = np.argwhere(is_missing)[0]
            raise ValueError(
                f"Missing value at file line {start_line+row}, column {col}, but integer data cannot hold NaN. Use dtype=float, missing='skip', or give an integer fill_value"
            )

        # string arrays have a fixed width, so make room for the fill value or it gets cut off (numpy uses 4 bytes per character)
        if np.issubdtype(values.dtype, np.str_):
            width = max(values.dtype.itemsize // 4, len(str(fill_value)))
            values = values.astype(f"<U{width}")
        else:
            values = values.copy()
        values[is_missing] = fill_value

    if return_mask:
        return values, is_missing
    return values


def load_csv(
    path,
    delimiter=",",
    header=False,
    missing="fill",
    fill_value=None,
    missing_values=None,
    dtype=float,
    usecols=None,
):
    """Load a CSV (or other delimited) file into a 2D NumPy array.

    Parameters
    ----------
    path: str or path-like
        File to read.
    delimiter: str or None, default ","
        Use "\\t" for tab-separated files, or None for any whitespace.
    header: bool, default False
        Set to True if the first line has column names.
    missing: {"fill", "skip"}, default "fill"
        "fill" puts fill_value in each gap. "skip" drops any row with a gap.
    fill_value: scalar, optional
        Defaults to NaN for floats and "" for strings. There's no default for ints, see Raises.
    missing_values: list of str, optional
        Strings to treat as missing. Defaults to DEFAULT_MISSING.
    dtype: data-type, default float
    usecols: list of int, optional
        Only load these columns. Handy for loading the number columns and the text columns separately.

    Returns
    -------
    data: numpy.ndarray of shape (n_rows, n_cols)
        Always 2D, even for a single row or column. Has 0 rows if the file only has a header or every row was skipped.
    names: list of str
        Only returned when header=True, as (data, names).

    Raises
    ------
    FileNotFoundError
        File doesn't exist.
    TypeError
        Delimiter isn't a string or None.
    ValueError
        Bad missing option, empty file, a row with the wrong number of columns, text in a number column, or a gap in an int column with no int fill_value.

    Notes
    -----
    Blank lines before the header (or before the first row) are skipped.
    The whole file is read into memory, use iter_csv for files that don't fit.
    O(R*C) time and space for R rows and C columns.

    Examples
    --------
    >>> X= load_csv("data.csv")  # doctest: +SKIP
    >>> X, names= load_csv("data.csv", header=True)  # doctest: +SKIP
    >>> X= load_csv("data.tsv", delimiter="\\t", missing="skip")  # doctest: +SKIP
    """
    _validate_args(path, delimiter, missing)
    dtype, fill_value = _resolve_dtype(dtype, fill_value)
    markers = DEFAULT_MISSING if missing_values is None else missing_values
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines, names, start_line = _take_header(lines, path, delimiter, header, usecols)
    n_cols = _count_cols(names, usecols)

    data = _parse_lines(
        lines,
        delimiter=delimiter,
        missing=missing,
        fill_value=fill_value,
        markers=markers,
        dtype=dtype,
        usecols=usecols,
        n_cols=n_cols,
        start_line=start_line,
    )
    if header:
        return data, names
    return data


def iter_csv(
    path,
    chunk_size=1000,
    delimiter=",",
    header=False,
    missing="fill",
    fill_value=None,
    missing_values=None,
    dtype=float,
    usecols=None,
):
    """Read a CSV file in chunks of rows instead of all at once.

    Takes the same options as load_csv. Each chunk is parsed exactly like load_csv would parse it.

    Parameters
    ----------
    path: str or path-like
        File to read.
    chunk_size: int, default 1000
        Number of lines read per chunk. A chunk can have fewer rows if some were skipped or it is the last one.
    delimiter, header, missing, fill_value, missing_values, dtype, usecols:
        Same as load_csv.

    Yields
    ------
    chunk: numpy.ndarray of shape (n_rows, n_cols)
        Always 2D. Chunks with no rows left (every row skipped) are not yielded.
    names: list of str
        Only when header=True, as (chunk, names). The same list comes with every chunk.

    Raises
    ------
    FileNotFoundError, TypeError, ValueError
        Same as load_csv, plus a bad chunk_size, or chunks with different numbers of columns.
        Since this is a generator, nothing is checked until the first chunk is asked for.

    Notes
    -----
    Only one chunk is in memory at a time: O(chunk_size*C) space, O(R*C) time in total.

    Examples
    --------
    >>> for chunk in iter_csv("big.csv", chunk_size=10000):  # doctest: +SKIP
    ...     scaler.partial_fit(chunk)
    """
    _validate_args(path, delimiter, missing)
    _check_chunk_size(chunk_size)
    dtype, fill_value = _resolve_dtype(dtype, fill_value)
    markers = DEFAULT_MISSING if missing_values is None else missing_values
    with open(path, "r", encoding="utf-8") as f:
        # find the first line that isn't blank; it's either the header or the first row
        line_no = 0
        for first in f:
            line_no += 1
            if first.strip():
                break
        else:
            raise ValueError(f"file is empty {os.fspath(path)!r}")
        names = _split_header(first, delimiter, usecols) if header else None
        pending = [] if header else [first]
        start_line = line_no + 1 if header else line_no
        n_cols = _count_cols(names, usecols)
        expected = None
        while True:
            lines = pending + list(islice(f, chunk_size - len(pending)))
            pending = []
            if not lines:
                return
            chunk = _parse_lines(
                lines,
                delimiter=delimiter,
                missing=missing,
                fill_value=fill_value,
                markers=markers,
                dtype=dtype,
                usecols=usecols,
                n_cols=n_cols,
                start_line=start_line,
            )
            start_line += len(lines)
            if chunk.shape[0] == 0:
                continue
            # each chunk is parsed on its own, so check the width matches the first one
            if expected is None:
                expected = chunk.shape[1]
            elif chunk.shape[1] != expected:
                raise ValueError(
                    f"Expected {expected} columns but the chunk starting at file line "
                    f"{start_line - len(lines)} has {chunk.shape[1]}"
                )
            yield (chunk, names) if header else chunk


def load_mixed_csv(
    path,
    numeric_cols,
    delimiter=",",
    header=False,
    missing="fill",
    missing_values=None,
):
    """Load a file that has both number and text columns.

    The number columns come back as a float array and every other column as a string array, with the rows lined up.

    Parameters
    ----------
    path: str or path-like
        File to read.
    numeric_cols: list of int
        Indexes of the number columns. All the other columns are loaded as text.
    delimiter, header, missing_values:
        Same as load_csv.
    missing: {"fill", "skip"}, default "fill"
        "fill" puts NaN in number gaps and "" in text gaps. "skip" drops any row with a gap in either part.

    Returns
    -------
    numbers: numpy.ndarray of float, shape (n_rows, len(numeric_cols))
    text: numpy.ndarray of str, shape (n_rows, n_cols - len(numeric_cols))
    names: list of str
        Only returned when header=True, as (numbers, text, names). Has every column name, in file order.

    Raises
    ------
    FileNotFoundError, TypeError, ValueError
        Same as load_csv, plus numeric_cols that aren't distinct column indexes.

    Notes
    -----
    The file is read once and parsed twice, once per part. O(R*C) time and space.

    Examples
    --------
    >>> X_num, X_text, names = load_mixed_csv("people.csv", [1, 2], header=True)  # doctest: +SKIP
    """
    _validate_args(path, delimiter, missing)
    markers = DEFAULT_MISSING if missing_values is None else missing_values
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines, names, start_line = _take_header(lines, path, delimiter, header, None)
    if names is not None:
        n_total = len(names)
    else:
        first = next((line for line in lines if line.strip()), "")
        n_total = len(_split_header(first, delimiter, None))
    numeric_cols = list(numeric_cols)
    for c in numeric_cols:
        if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
            raise TypeError(f"numeric_cols must hold integers, got {type(c).__name__}")
        if not 0 <= c < n_total:
            raise ValueError(f"column {c} is out of range for {n_total} columns")
    if len(set(numeric_cols)) != len(numeric_cols):
        raise ValueError("numeric_cols must not contain duplicates")
    text_cols = [c for c in range(n_total) if c not in set(numeric_cols)]

    def parse(cols, dtype, fill_value):
        """Read the given columns as one array, or return None if there are none."""
        if not cols:
            return None
        # always fill here; rows are skipped below, using both parts' gaps
        return _parse_lines(
            lines,
            delimiter=delimiter,
            missing="fill",
            fill_value=fill_value,
            markers=markers,
            dtype=dtype,
            usecols=cols,
            n_cols=len(cols),
            start_line=start_line,
            return_mask=True,
        )

    num_part = parse(numeric_cols, float, np.nan)
    text_part = parse(text_cols, str, "")
    # a part with no columns still needs the right number of rows
    n_rows = (num_part or text_part)[0].shape[0]
    if num_part is None:
        num_part = np.empty((n_rows, 0)), np.zeros((n_rows, 0), dtype=bool)
    if text_part is None:
        text_part = np.empty((n_rows, 0), dtype=str), np.zeros((n_rows, 0), dtype=bool)
    numbers, num_gaps = num_part
    text, text_gaps = text_part
    if missing == "skip":
        keep = ~(num_gaps.any(axis=1) | text_gaps.any(axis=1))
        numbers, text = numbers[keep], text[keep]
    if header:
        return numbers, text, names
    return numbers, text