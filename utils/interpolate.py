import numpy as np
def interpolate_over_nans(input_mat, t=None):
    """ Function to interpolate over NaN values along the second dimension of a matrix

    Args:
        input_mat: numpy array, 1D or [neurons, time] 2D matrix with NaN values
        t: optional time vector, only useful if input_mat is not sampled regularly in time

    Returns: Interpolated input_mat, interpolated time
    """
    def check_input_format(data):
        if type(data) is not np.ndarray:
            raise Exception('The red and green matricies must be the numpy arrays')

        if data.ndim != 1 and data.ndim != 2:
            raise Exception('The red and green matricies should be 1 or 2 dimensional')

        if data.ndim == 1:
            data = data[None, :] # Make it [1, time] instead of [time, 1]

        return data

    # Check and prepare input
    input_mat = check_input_format(input_mat)
    
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
