# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from numpy cimport ndarray, int64_t

# Cython routines for window indexers


def calculate_variable_window_bounds(
    int64_t num_values,
    int64_t window_size,
    object step_size_obj,
    object min_periods_obj,
    object center,  # unused but here to match get_window_bounds signature
    object closed,
    const int64_t[:] index
):
    """
    Calculate window boundaries for rolling windows from a time offset.

    Parameters
    ----------
    num_values : int64
        total number of values

    window_size : int64
        window size calculated from the offset

    step_size : Optional[int], default None
        the window step size

    min_periods : object
        Minimum data points in each window.

    center : object
        ignored, exists for compatibility

    closed : str
        string of side of the window that should be closed

    index : ndarray[int64]
        time series index to roll over

    Returns
    -------
    (ndarray[int64], ndarray[int64])
    """
    cdef:
        bint left_open = False
        bint right_open = False
        int idx_scalar = 1
        ndarray[int64_t, ndim=1] start, end
        int64_t step_size, min_periods
        int64_t index_i, index_si, index_ei,
        int64_t index_window_i, index_step_i
        int64_t index_window_max, index_step_max
        int64_t window_i = 0
        int64_t next_index_si = 0
        int64_t next_index_ei = 0
        Py_ssize_t i, j

    if closed is None:
        closed = 'left'

    if closed not in ['right', 'both']:
        right_open = True

    if closed not in ['left', 'both']:
        left_open = True

    # Assume index is monotonic increasing or decreasing. If decreasing (WHY??) negate values.
    if index[num_values - 1] < index[0]:
        idx_scalar = -1

    # Minimum "observations".
    min_periods = min_periods_obj if min_periods_obj is not None else 0
    step_size = step_size_obj if step_size_obj is not None else 1

    start = np.empty(num_values, dtype='int64')
    start.fill(-1)
    end = np.empty(num_values, dtype='int64')

    if num_values < 1:
        return start, end

    # Indexing into indices: index_si index_ei (index start/end)
    # Indexing into start/end arrays: window_i
    # This will find closed intervals [start, end]

    window_i = 0
    next_index_si = 0
    next_index_ei = 0

    with nogil:
        while next_index_ei < num_values:
            index_si = next_index_si

            start[window_i] = index_si

            index_window_max = index[index_si] + idx_scalar*(window_size - 1)
            index_step_max = index[index_si] + idx_scalar*(step_size - 1)

            # Find end of step.
            index_step_i = num_values - 1
            for index_i in range(index_si + 1, num_values):
                # Outside of step?
                if idx_scalar*index[index_i] > idx_scalar*index_step_max:
                    index_step_i = index_i - 1
                    break

            # Find end of window.
            index_window_i = num_values - 1
            for index_i in range(next_index_ei + 1, num_values):
                # Outside of window?
                if idx_scalar*index[index_i] > idx_scalar*index_window_max:
                    index_window_i = index_i - 1
                    break

            next_index_si = index_step_i + 1
            next_index_ei = next_index_si if next_index_si > index_window_i + 1 else index_window_i + 1

            end[window_i] = index_window_i
            window_i += 1

    # Remove excess slots.
    valid_idx = (start >= 0) & (start <= end)

    # And windows without enough data.
    if min_periods is not None:
        valid_idx &= (end - start + 1) >= min_periods

    # Update open boundaries.
    if left_open:
        start -= 1
    if right_open:
        end += 1

    return start[valid_idx], end[valid_idx]
