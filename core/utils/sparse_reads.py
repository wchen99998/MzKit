"""
Direct reads of sparse (lane x scan) arrays.

scipy's sparse indexing costs tens of microseconds per call, and ensemble
extraction reads lanes and columns hundreds of thousands of times. These read
the index arrays of canonical CSR / CSC arrays directly and return exactly what
the equivalent scipy slice + toarray() returns (falling back to scipy otherwise).
"""
from typing import Optional

import numpy as np

def csr_row_dense(arr, row: int, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
    """``arr[row, start:stop].toarray().flatten()`` for a CSR array."""
    stop = arr.shape[1] if stop is None else stop
    if arr.format != 'csr' or not arr.has_canonical_format:
        return arr[row, start:stop].toarray().flatten()
    out = np.zeros(max(stop - start, 0), dtype=arr.dtype)
    lo, hi = arr.indptr[row], arr.indptr[row + 1]
    cols = arr.indices[lo:hi]
    a, b = np.searchsorted(cols, start), np.searchsorted(cols, stop)
    out[cols[a:b] - start] = arr.data[lo + a:lo + b]
    return out


def csr_rows_dense(arr, rows: np.ndarray, start: int, stop: int) -> np.ndarray:
    """``arr[rows, start:stop].toarray()`` for a CSR array."""
    rows = np.asarray(rows, dtype=np.int64)
    if arr.format != 'csr' or not arr.has_canonical_format:
        return arr[rows, start:stop].toarray()
    out = np.zeros((rows.size, max(stop - start, 0)), dtype=arr.dtype)
    lo = arr.indptr[rows]
    n = arr.indptr[rows + 1] - lo
    total = int(n.sum())
    if total == 0:
        return out
    if total > 4 * out.size + 1024:
        # Rows much longer than the window (long runs: a lane can span 100k scans): binary-search each row's
        # window, as csr_row_dense does, instead of reading every entry of every row
        for i, (a0, m) in enumerate(zip(lo.tolist(), n.tolist())):
            cols = arr.indices[a0:a0 + m]
            a, b = np.searchsorted(cols, start), np.searchsorted(cols, stop)
            out[i, cols[a:b] - start] = arr.data[a0 + a:a0 + b]
        return out
    which = np.repeat(np.arange(rows.size), n)
    pos = np.repeat(lo - (np.cumsum(n) - n), n) + np.arange(total)
    cols = arr.indices[pos]
    keep = (cols >= start) & (cols < stop)
    out[which[keep], cols[keep] - start] = arr.data[pos[keep]]
    return out


def csc_rows_dense(csc, rows: np.ndarray, start: int, stop: int) -> Optional[np.ndarray]:
    """``arr[rows, start:stop].toarray()`` from the canonical CSC copy ``csc`` of a CSR array ``arr``: reads only the
    entries of columns start..stop-1, so the cost follows the window, not the rows' full length. None (use
    csr_rows_dense) when ``csc`` is not canonical CSC or ``rows`` repeats a row."""
    rows = np.asarray(rows, dtype=np.int64)
    if csc.format != 'csc' or not csc.has_canonical_format:
        return None
    order = np.argsort(rows, kind='stable')
    sorted_rows = rows[order]
    if sorted_rows.size > 1 and (sorted_rows[1:] == sorted_rows[:-1]).any():
        return None
    out = np.zeros((rows.size, max(stop - start, 0)), dtype=csc.dtype)
    a, b = csc.indptr[start], csc.indptr[max(stop, start)]
    if rows.size == 0 or a == b:
        return out
    entry_rows = csc.indices[a:b]
    entry_cols = np.repeat(np.arange(stop - start), np.diff(csc.indptr[start:stop + 1]))
    pos = np.minimum(np.searchsorted(sorted_rows, entry_rows), rows.size - 1)
    hit = sorted_rows[pos] == entry_rows
    out[order[pos[hit]], entry_cols[hit]] = csc.data[a:b][hit]
    return out


def csc_col_entries(arr, col: int) -> tuple[np.ndarray, np.ndarray]:
    """Stored (row indices, values) of one column of a CSC array, rows ascending."""
    if arr.format != 'csc' or not arr.has_canonical_format:
        dense = arr[:, [col]].toarray().flatten()
        rows = np.flatnonzero(dense)
        return rows, dense[rows]
    lo, hi = arr.indptr[col], arr.indptr[col + 1]
    return arr.indices[lo:hi], arr.data[lo:hi]
