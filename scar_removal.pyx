# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fmin, fmax

cnp.import_array()

ctypedef cnp.float64_t DTYPE_t


# ─────────────────────────────────────────────
# Internal helpers (cdef = C-only, no Python overhead)
# ─────────────────────────────────────────────

cdef void _mark_if_positive_scar(
    Py_ssize_t row,
    Py_ssize_t col,
    double stddev,
    DTYPE_t[:, ::1] img,
    DTYPE_t[:, ::1] marked,
    double threshold_low,
    int max_scar_width,
) noexcept nogil:

    cdef:
        Py_ssize_t k, nrows
        double min_scar_value, max_border_value, val

    nrows = img.shape[0]
    min_scar_value = img[row + 1, col]
    max_border_value = img[row, col]

    if min_scar_value - max_border_value > threshold_low * stddev:
        for k in range(1, max_scar_width + 1):
            if row + k + 1 >= nrows:
                break
            if img[row + k, col] < min_scar_value:
                min_scar_value = img[row + k, col]
            if img[row + k + 1, col] > img[row, col]:
                max_border_value = img[row + k + 1, col]
            else:
                max_border_value = img[row, col]
            if min_scar_value - max_border_value > threshold_low * stddev:
                while k > 0:
                    val = (img[row + k, col] - max_border_value) / stddev
                    marked[row + k, col] = val
                    k -= 1


cdef void _mark_if_negative_scar(
    Py_ssize_t row,
    Py_ssize_t col,
    double stddev,
    DTYPE_t[:, ::1] img,
    DTYPE_t[:, ::1] marked,
    double threshold_low,
    int max_scar_width,
) noexcept nogil:

    cdef:
        Py_ssize_t k, nrows
        double min_scar_value, max_border_value, val

    nrows = img.shape[0]
    min_scar_value = img[row + 1, col]
    max_border_value = img[row, col]

    if min_scar_value - max_border_value < -threshold_low * stddev:
        for k in range(1, max_scar_width + 1):
            if row + k + 1 >= nrows:
                break
            if img[row + k, col] > min_scar_value:
                min_scar_value = img[row + k, col]
            if img[row + k + 1, col] < img[row, col]:
                max_border_value = img[row + k + 1, col]
            else:
                max_border_value = img[row, col]
            if min_scar_value - max_border_value < -threshold_low * stddev:
                while k > 0:
                    val = (max_border_value - img[row + k, col]) / stddev
                    marked[row + k, col] = val
                    k -= 1


cdef void _spread_scars(
    DTYPE_t[:, ::1] marked,
    double threshold_low,
    double threshold_high,
) noexcept nogil:

    cdef Py_ssize_t row, col, nrows, ncols
    nrows = marked.shape[0]
    ncols = marked.shape[1]

    for row in range(nrows):
        for col in range(1, ncols):
            if marked[row, col] >= threshold_low and marked[row, col - 1] >= threshold_high:
                marked[row, col] = threshold_high
        for col in range(ncols - 1, 0, -1):
            if marked[row, col - 1] >= threshold_low and marked[row, col] >= threshold_high:
                marked[row, col - 1] = threshold_high


cdef void _remove_short_scars(
    DTYPE_t[:, ::1] marked,
    double threshold_high,
    int min_scar_length,
) noexcept nogil:

    cdef:
        Py_ssize_t row, col, nrows, ncols
        int k

    nrows = marked.shape[0]
    ncols = marked.shape[1]

    for row in range(nrows):
        k = 0
        for col in range(ncols):
            if marked[row, col] >= threshold_high:
                marked[row, col] = 1.0
                k += 1
                continue
            if k > 0 and k < min_scar_length:
                while k > 0:
                    marked[row, col - k] = 0.0
                    k -= 1
            marked[row, col] = 0.0
            k = 0
        if k > 0 and k < min_scar_length:
            while k > 0:
                marked[row, col - k] = 0.0
                k -= 1


# ─────────────────────────────────────────────
# Public-facing Python functions
# ─────────────────────────────────────────────

def _mark_scars(
    cnp.ndarray[DTYPE_t, ndim=2] img,
    str direction,
    double threshold_low,
    double threshold_high,
    int max_scar_width,
    int min_scar_length,
):
    cdef:
        DTYPE_t[:, ::1] image_view
        DTYPE_t[:, ::1] marked_view
        double stddev
        Py_ssize_t row, col, nrows, ncols

    image = np.ascontiguousarray(img, dtype=np.float64)
    marked_np = np.zeros(image.shape, dtype=np.float64)

    image_view = image
    marked_view = marked_np

    stddev = float(np.std(image))
    nrows = image.shape[0]
    ncols = image.shape[1]

    if direction == "positive":
        with nogil:
            for row in range(nrows - 1):
                for col in range(ncols):
                    _mark_if_positive_scar(
                        row, col, stddev, image_view, marked_view,
                        threshold_low, max_scar_width
                    )
    elif direction == "negative":
        with nogil:
            for row in range(nrows - 1):
                for col in range(ncols):
                    _mark_if_negative_scar(
                        row, col, stddev, image_view, marked_view,
                        threshold_low, max_scar_width
                    )
    else:
        raise ValueError(f"direction '{direction}' invalid.")

    with nogil:
        _spread_scars(marked_view, threshold_low, threshold_high)
        _remove_short_scars(marked_view, threshold_high, min_scar_length)

    return marked_np


def _remove_marked_scars(
    cnp.ndarray[DTYPE_t, ndim=2] img,
    cnp.ndarray[DTYPE_t, ndim=2] scar_mask,
):
    cdef:
        DTYPE_t[:, ::1] img_view = np.ascontiguousarray(img, dtype=np.float64)
        DTYPE_t[:, ::1] mask_view = np.ascontiguousarray(scar_mask, dtype=np.float64)
        Py_ssize_t row, col, nrows, ncols, scar_width, k
        double above, below, interp_val

    nrows = img_view.shape[0]
    ncols = img_view.shape[1]

    with nogil:
        for row in range(nrows):
            for col in range(ncols):
                if mask_view[row, col] == 1.0:
                    scar_width = 1
                    while row + scar_width < nrows and mask_view[row + scar_width, col] == 1.0:
                        scar_width += 1
                    above = img_view[row - 1, col]
                    below = img_view[row + scar_width, col]
                    k = scar_width
                    while k > 0:
                        interp_val = (
                            (<double>k / <double>(scar_width + 1)) * below
                            + (1.0 - (<double>k / <double>(scar_width + 1))) * above
                        )
                        img_view[row + k - 1, col] = interp_val
                        mask_view[row + k - 1, col] = 0.0
                        k -= 1


def remove_scars(
    cnp.ndarray[DTYPE_t, ndim=2] img,
    int removal_iterations=2,
    double threshold_low=0.250,
    double threshold_high=0.666,
    int max_scar_width=4,
    int min_scar_length=16,
):
    cdef int i
    cdef cnp.ndarray[DTYPE_t, ndim=2] marked_positive, marked_negative
    cdef cnp.ndarray marked_both

    for i in range(removal_iterations):
        marked_positive = _mark_scars(
            img, "positive", threshold_low, threshold_high,
            max_scar_width, min_scar_length
        )
        marked_negative = _mark_scars(
            img, "negative", threshold_low, threshold_high,
            max_scar_width, min_scar_length
        )
        marked_both = np.bitwise_or(
            marked_positive.astype(bool),
            marked_negative.astype(bool)
        ).astype(np.float64)

        _remove_marked_scars(img, np.copy(marked_both))

    return img
