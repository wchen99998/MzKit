"""The direct sparse reads return exactly what scipy's slicing returns."""
import numpy as np
import scipy.sparse as sp

from core.utils.sparse_reads import (csc_col_entries, csc_rows_above, csc_rows_dense, csr_row_dense,
                                     csr_rows_dense)


def _lanes(rng, n_rows, n_cols, density):
    arr = sp.random(n_rows, n_cols, density=density, format='csr', random_state=rng, dtype=np.float32)
    arr.sum_duplicates()
    return arr


def test_csr_rows_dense_matches_scipy_short_and_long_rows():
    rng = np.random.default_rng(0)
    for n_cols, density in [(50, 0.3), (20_000, 0.5)]:  # short rows (one pass) and rows far longer than the window
        arr = _lanes(rng, 200, n_cols, density)
        for _ in range(50):
            rows = rng.choice(200, size=rng.integers(1, 40), replace=False)
            start = int(rng.integers(0, n_cols - 1))
            stop = int(rng.integers(start, min(n_cols, start + 40) + 1))
            expected = arr[rows, start:stop].toarray()
            got = csr_rows_dense(arr, rows, start, stop)
            assert got.dtype == expected.dtype and np.array_equal(got, expected)


def test_csr_row_dense_and_csc_col_entries_match_scipy():
    rng = np.random.default_rng(1)
    arr = _lanes(rng, 100, 500, 0.2)
    for row in range(0, 100, 7):
        assert np.array_equal(csr_row_dense(arr, row), arr[row].toarray().ravel())
        assert np.array_equal(csr_row_dense(arr, row, 40, 90), arr[row, 40:90].toarray().ravel())
    csc = arr.tocsc()
    for col in range(0, 500, 37):
        rows, vals = csc_col_entries(csc, col)
        dense = csc[:, col].toarray().ravel()
        assert np.array_equal(rows, np.flatnonzero(dense)) and np.array_equal(vals, dense[dense != 0])


def test_csc_reads_match_scipy():
    rng = np.random.default_rng(2)
    arr = _lanes(rng, 300, 400, 0.1)
    arr.data *= 1000
    csc = arr.tocsc()
    for _ in range(40):
        start = int(rng.integers(0, 390))
        stop = int(rng.integers(start, min(400, start + 30) + 1))
        rows = rng.choice(300, size=rng.integers(1, 60), replace=False)
        assert np.array_equal(csc_rows_dense(csc, rows, start, stop), arr[rows, start:stop].toarray())
        cols = np.arange(start, stop) if rng.random() < 0.5 else np.sort(rng.choice(400, size=12, replace=False))
        for threshold in (0.0, 300.0):
            expected = np.where(csc[:, cols].max(axis=1).toarray().ravel() > threshold)[0]
            assert np.array_equal(csc_rows_above(csc, cols, threshold), expected)
    assert csc_rows_dense(csc, np.array([3, 3]), 0, 5) is None  # repeated rows: caller falls back to CSR
