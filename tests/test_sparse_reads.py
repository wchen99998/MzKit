"""The direct sparse reads return exactly what scipy's slicing returns."""
import numpy as np
import scipy.sparse as sp

from core.utils.sparse_reads import csc_col_entries, csr_row_dense, csr_rows_dense


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
