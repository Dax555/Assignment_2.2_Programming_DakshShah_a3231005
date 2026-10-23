import os
import numpy as np

__all__ = ["load_csv"]
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
):
    if not any(line.strip() for line in lines):
        return np.empty((0, n_cols or 0), dtype=dtype)
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
    if not any(line.strip() for line in lines):
        raise ValueError(f"file is empty {os.fspath(path)!r}")
    names = None
    if header:
        names = _split_header(lines[0], delimiter, usecols)
        lines = lines[1:]
    if names is not None:
        n_cols = len(names)
    elif usecols is not None:
        n_cols = len(usecols)
    else:
        n_cols = None

    data = _parse_lines(
        lines,
        delimiter=delimiter,
        missing=missing,
        fill_value=fill_value,
        markers=markers,
        dtype=dtype,
        usecols=usecols,
        n_cols=n_cols,
        start_line=2 if header else 1,
    )
    if header:
        return data, names
    return data
