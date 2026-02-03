import numpy as np
import pandas as pd

def detect_and_mask_step_drops(intensity_input, diff_threshold=None, window=10, drop_ratio=0.3):
    """
    Detect sudden irreversible intensity drops (focus loss, displacement) and mask 
    the affected post-drop data as NaN.
    
    Parameters:
    - intensity_input: pd.DataFrame or np.ndarray (Neurons x Time)
    - diff_threshold: float, optional. Threshold for the first derivative to identify potential drops.
                      If None, uses 5 * MAD of the diff.
    - window: int. Window size for comparing pre- and post-drop means.
    - drop_ratio: float (0.0 to 1.0). The required relative drop magnitude to confirm a step artifact.
                  e.g., 0.3 means the post-drop mean must be < 70% of the pre-drop mean.
                  
    Returns:
    - masked_data: Same type as input, with drops and subsequent data masked as NaN.
    """
    # Handle Input Type
    is_df = isinstance(intensity_input, pd.DataFrame)
    if is_df:
        data = intensity_input.values.astype(float).copy() # (n_neurons, n_time)
        columns = intensity_input.columns
        index = intensity_input.index
    else:
        data = np.array(intensity_input, dtype=float, copy=True)
        
    # Ensure 2D
    original_shape = data.shape
    if data.ndim == 1:
        data = data[np.newaxis, :]
        
    n_neurons, n_time = data.shape
    
    for i in range(n_neurons):
        trace = data[i, :]
        
        # Handle existing NaNs (if any, though usually raw intensity shouldn't have them yet)
        # We compute diff ignoring NaNs or just on the raw trace.
        # np.diff on NaNs results in NaNs, which is fine for threshold comparison (NaN < thresh is False)
        
        # 1. Calculate Diff
        diff = np.diff(trace, prepend=trace[0])
        
        # 2. Determine Threshold
        if diff_threshold is None:
            # Dynamic threshold: 5 * MAD
            # Use valid diffs only
            valid_diff = diff[np.isfinite(diff)]
            if len(valid_diff) == 0:
                continue
                
            median_diff = np.median(valid_diff)
            abs_dev = np.abs(valid_diff - median_diff)
            mad = np.median(abs_dev)
            
            if mad == 0:
                # Fallback if signal is perfectly flat
                mad = np.std(valid_diff) + 1e-6
            
            # We are looking for large NEGATIVE diffs
            thresh = -5 * mad
        else:
            thresh = -abs(diff_threshold)
            
        # 3. Find Candidates
        # Points where the drop is sharper than threshold
        # Using np.where checks condition. NaNs will be false.
        candidates = np.where(diff < thresh)[0]
        
        if len(candidates) == 0:
            continue
            
        # 4. Sustain Check
        first_drop_idx = -1
        
        for t in candidates:
            # Need enough context
            if t < window or t > n_time - window:
                continue
            
            # Check pre and post means
            pre_window = trace[t-window:t]
            post_window = trace[t:t+window]
            
            # Skip if windows contain NaNs (or handle them)
            # Assuming raw data is mostly clean, or we valid-mean
            mean_pre = np.nanmean(pre_window)
            mean_post = np.nanmean(post_window)
            
            if np.isnan(mean_pre) or np.isnan(mean_post) or mean_pre == 0:
                continue
                
            # Check if drop is sustained and large enough
            # e.g. Post mean is significantly lower than Pre mean
            if mean_post < mean_pre * (1 - drop_ratio):
                # Confirmed Step Drop
                first_drop_idx = t
                break # We assume irreversible, so the first one bricks the rest
        
        # 5. Mask
        if first_drop_idx != -1:
            data[i, first_drop_idx:] = np.nan
            
    # Restore Shape/Type
    if is_df:
        return pd.DataFrame(data, index=index, columns=columns)
    elif len(original_shape) == 1:
        return data.flatten()
    else:
        return data
