import numpy as np
import pandas as pd

def _despike_trace(trace, mad_factor=5, tolerance_factor=2):
    """Internal helper to remove single-point artifacts."""
    n = len(trace)
    if n < 3: return trace
    
    # Compute MAD of diff for local thresholding
    diff = np.diff(trace)
    valid_diff = diff[np.isfinite(diff)]
    if len(valid_diff) == 0: return trace
    
    median_diff = np.median(valid_diff)
    mad = np.median(np.abs(valid_diff - median_diff))
    if mad == 0: 
        # If signal is very flat, use a small fraction of the median intensity as floor
        mad = np.nanmedian(np.abs(trace)) * 0.01 + 1e-6
    
    # We look for both "V" (drop-recovery) and "Inverse-V" (spike-decay)
    # But for step detection errors, "V" is the primary target
    for t in range(1, n - 1):
        # Case: Drop followed by recovery
        if (trace[t-1] - trace[t] > mad_factor * mad and 
            trace[t+1] - trace[t] > mad_factor * mad):
            if abs(trace[t-1] - trace[t+1]) < tolerance_factor * mad:
                trace[t] = (trace[t-1] + trace[t+1]) / 2.0
        # Case: Spike followed by decay (optional but good for stability)
        elif (trace[t] - trace[t-1] > mad_factor * mad and 
              trace[t] - trace[t+1] > mad_factor * mad):
            if abs(trace[t-1] - trace[t+1]) < tolerance_factor * mad:
                trace[t] = (trace[t-1] + trace[t+1]) / 2.0
    return trace

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
        
        # 0. Pre-Correction: Despiking (prevents single-point artifacts from triggering or skewing)
        trace = _despike_trace(trace)
        
        # 1. Calculate Diff
        diff = np.diff(trace, prepend=trace[0])
        
        # 2. Determine Threshold
        valid_diff = diff[np.isfinite(diff)]
        if len(valid_diff) == 0:
            mad = 1e-6
        else:
            median_diff = np.median(valid_diff)
            mad = np.median(np.abs(valid_diff - median_diff))
            if mad == 0:
                mad = np.std(valid_diff) + 1e-6

        if diff_threshold is None:
            thresh = -5 * mad
        else:
            thresh = -abs(diff_threshold)
            
        # 3. Find Candidates
        candidates = np.where(diff < thresh)[0]
        
        if len(candidates) == 0:
            continue
            
        # 3b. Adaptive Confirmation Floor
        # Calculate a signal-relative "noise floor" (Median + 5 * MAD)
        p50 = np.nanmedian(trace)
        noise_floor = p50 + 2 * mad
                
        # 4. Sustain Check
        first_drop_idx = -1
        
        for t in candidates:
            # Need enough context
            if t < window or t > n_time - window:
                continue
            
            # Check pre and post medians
            pre_window = trace[t-window:t]
            post_window = trace[t:t+window]
            val_pre = np.nanmedian(pre_window)
            val_post = np.nanmedian(post_window)
            
            # Fallback to mean if median is 0 (very sparse signal) but we have signal
            if val_pre == 0:
                val_pre = np.nanmean(pre_window)
                val_post = np.nanmean(post_window)
            
            if np.isnan(val_pre) or np.isnan(val_post) or val_pre == 0:
                continue
                
            # Check if drop is sustained, large enough, AND falls below noise floor
            is_large_drop = val_post < val_pre * (1 - drop_ratio)
            is_below_floor = val_post < noise_floor
            is_extreme_drop = val_post < val_pre * 0.51 
            
            if is_large_drop and (is_below_floor or is_extreme_drop):
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
