import numpy as np
import pandas as pd


def interpolate_over_nans(input_mat, t=None):
    """Function to interpolate over NaN values along the second dimension of a matrix.

    Args:
        input_mat: numpy array, 1D or [neurons, time] 2D matrix with NaN values. A
            pandas DataFrame of shape [neurons, time] is also supported.
        t: optional time vector, only useful if input_mat is not sampled regularly in time.

    Returns:
        Tuple containing the interpolated matrix (matching the input type) and the
        generated/interpolated time vector.
    """

    def check_numpy_input(data):
        if not isinstance(data, np.ndarray):
            raise Exception('The red and green matricies must be the numpy arrays')

        if data.ndim not in (1, 2):
            raise Exception('The red and green matricies should be 1 or 2 dimensional')

        if data.ndim == 1:
            data = data[None, :]  # Make it [1, time] instead of [time, 1]

        return data

    def interpolate_dataframe(df_input, t_vals):
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

        interp_df = df_numeric.interpolate(axis=1, method='linear', limit_direction='both')

        all_nan_cols = df_numeric.columns[df_numeric.isna().all()]
        for col in all_nan_cols:
            print('column ' + str(col) + ' is all NaN, skipping')
            interp_df[col] = 0.0

        return interp_df, t_interp_vals

    if isinstance(input_mat, pd.DataFrame):
        return interpolate_dataframe(input_mat, t)

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
            continue

        # find the location of all nan values
        no_nan_ind = ~np.isnan(input_mat[:, c])

        # remove nans from t and the data
        no_nan_t = t[no_nan_ind]
        no_nan_data_mat = input_mat[no_nan_ind, c]

        # use numpy.interp for linear interpolation (modern alternative to interp1d)
        output_mat[:, c] = np.interp(t_interp, no_nan_t, no_nan_data_mat)

    # Transpose back to [neurons, time]
    return output_mat.T, t_interp
