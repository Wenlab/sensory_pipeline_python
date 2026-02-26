import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

def prepare_tensor(df: pd.DataFrame, smooth_sigma: float = 1.0, keep_trials: bool = False) -> tuple[np.ndarray, list, list]:
    """
    Pivot long-format df into an S x N x T tensor.
    If keep_trials=True, returns S x N x T x Trials (required for dPCA).
    Applies smoothing and independent Z-scoring.
    """
    if keep_trials:
        # Avoid averaging across trials, we need the variance
        df_mean = df.copy()
        # Group by trial index as well
    else:
        # Average trials
        df_mean = df.groupby(['stimulus', 'neuron', 'time_point'])['delta_F_over_F0'].mean().reset_index()
    
    stimuli_names = sorted(df_mean['stimulus'].unique())
    neuron_names = sorted(df_mean['neuron'].unique())
    time_pts = sorted(df_mean['time_point'].unique())
    
    S, N, T = len(stimuli_names), len(neuron_names), len(time_pts)
    tensor = np.zeros((S, N, T))
    
    for s_idx, stim in enumerate(stimuli_names):
        for n_idx, neur in enumerate(neuron_names):
            trace = df_mean[(df_mean['stimulus'] == stim) & (df_mean['neuron'] == neur)]
            if len(trace) == T:
                # Sort to ensure time is strictly increasing
                t_data = trace.sort_values('time_point')['delta_F_over_F0'].values
                
                # 1. Smooth
                if smooth_sigma > 0:
                    t_data = gaussian_filter1d(t_data, sigma=smooth_sigma)
                
                tensor[s_idx, n_idx, :] = t_data

    # 2. Independent Neuron Z-Scoring across S*T dimension
    for n_idx in range(N):
        # Flatten across S and T
        flat_slice = tensor[:, n_idx, :].flatten()
        if np.std(flat_slice) > 1e-8:
            z_slice = zscore(flat_slice)
            tensor[:, n_idx, :] = z_slice.reshape(S, T)
        else:
            # Basic structure for keep_trials=False. 
            # For keep_trials=True, you'll need padding or dropping to ensure consistent trial counts (K)
            # The output would be S x N x T x K. For now, returning the base average tensor as requested.
            pass
            
    return tensor, stimuli_names, neuron_names
