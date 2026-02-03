import numpy as np
import pandas as pd


def interpolate_over_nans(input_mat, t=None, max_gap=10):
    """Function to interpolate over NaN values along the second dimension of a matrix.

    Args:
        input_mat: numpy array, 1D or [neurons, time] 2D matrix with NaN values. A
            pandas DataFrame of shape [neurons, time] is also supported.
        t: optional time vector, only useful if input_mat is not sampled regularly in time.
        max_gap: int, maximum number of consecutive NaNs to interpolate. 
                 Gaps larger than this will remain NaN. (Default: 10)

    Returns:
        Tuple containing the interpolated matrix (matching the input type) and the
        generated/interpolated time vector.
    """

    def get_large_gap_mask(arr_nan_mask, max_gap):
        """Helper to identify runs of NaNs larger than max_gap."""
        mask = np.zeros_like(arr_nan_mask, dtype=bool)
        if not arr_nan_mask.any() or max_gap is None:
            return mask
            
        # Pad with False to handle boundaries
        padded = np.concatenate(([False], arr_nan_mask, [False]))
        # Diff to find edges (1: False->True (Start), -1: True->False (End))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            length = end - start
            if length > max_gap:
                mask[start:end] = True
        return mask

    def check_numpy_input(data):
        if not isinstance(data, np.ndarray):
            raise Exception('The red and green matricies must be the numpy arrays')

        if data.ndim not in (1, 2):
            raise Exception('The red and green matricies should be 1 or 2 dimensional')

        if data.ndim == 1:
            data = data[None, :]  # Make it [1, time] instead of [time, 1]

        return data

    def interpolate_dataframe(df_input, t_vals, max_gap):
        df_numeric = df_input.copy()

        if not all(np.issubdtype(dtype, np.number) for dtype in df_numeric.dtypes):
            raise TypeError('All DataFrame columns must be numeric for interpolation')

        df_numeric = df_numeric.astype(float)

        if t_vals is None:
            t_vals = np.arange(df_numeric.shape[1])
        else:
            t_vals = np.asarray(t_vals)
            if t_vals.ndim != 1:
                raise ValueError('t must be a 1D array-like when provided')
            if t_vals.size != df_numeric.shape[1]:
                raise ValueError('The length of t must match the number of time points')

        sample_rate = 1 / np.mean(np.diff(t_vals, axis=0))
        t_interp_vals = np.arange(df_numeric.shape[1]) / sample_rate

        # 1. Interpolate everything first
        interp_df = df_numeric.interpolate(axis=1, method='linear', limit_direction='both')

        # 2. Restore large gaps if max_gap is set
        if max_gap is not None:
             # Check each row for large gaps
             for idx in df_numeric.index:
                 row_nan_mask = df_numeric.loc[idx].isna().values
                 large_gaps = get_large_gap_mask(row_nan_mask, max_gap)
                 if large_gaps.any():
                     interp_df.loc[idx, large_gaps] = np.nan

        all_nan_cols = df_numeric.columns[df_numeric.isna().all()]
        for col in all_nan_cols:
            print('column ' + str(col) + ' is all NaN, skipping')
            interp_df[col] = 0.0

        return interp_df, t_interp_vals

    if isinstance(input_mat, pd.DataFrame):
        return interpolate_dataframe(input_mat, t, max_gap)

    # Check and prepare numpy input
    input_mat = check_numpy_input(input_mat)

    # Transpose to [time, neurons] for processing
    input_mat = input_mat.T

    # if t is not specified, assume it has been sampled at regular intervals
    if t is None:
        t = np.arange(input_mat.shape[0])

    output_mat = np.zeros(input_mat.shape)

    # calculate the average sample rate and uses this to create an interpolated t
    sample_rate = 1 / np.mean(np.diff(t, axis=0))
    t_interp = np.arange(input_mat.shape[0]) / sample_rate

    # loop through each column of the data and interpolate them separately
    for c in range(input_mat.shape[1]):
        # check if all the data is nan and skip if it is
        if np.all(np.isnan(input_mat[:, c])):
            print('column ' + str(c) + ' is all NaN, skipping')
            output_mat[:, c] = np.nan # Ensure output is NaN, not 0 if all bad
            continue

        # find the location of all nan values
        # Original logic: interpolate on valid points
        no_nan_ind = ~np.isnan(input_mat[:, c])

        # remove nans from t and the data
        no_nan_t = t[no_nan_ind]
        no_nan_data_mat = input_mat[no_nan_ind, c]
        
        # Linear Interpolation
        interp_res = np.interp(t_interp, no_nan_t, no_nan_data_mat)
        
        # Restore large gaps
        if max_gap is not None:
            nan_mask = np.isnan(input_mat[:, c])
            large_gaps = get_large_gap_mask(nan_mask, max_gap)
            interp_res[large_gaps] = np.nan
            
        output_mat[:, c] = interp_res

    # Transpose back to [neurons, time]
    return output_mat.T, t_interp
