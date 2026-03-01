import numpy as np
import pandas as pd

def prepare_chemo_tensor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list, list, dict]:
    """
    Step 1.1 - 1.3: Clean, average, and reshape into 2D and 3D tensors.
    """
    # Group by stimulus, neuron, and time_point to average across trials/worms
    # and deal with missing time_points using interpolation/ffill if necessary
    
    # For now, strict mean grouping
    mean_df = df.groupby(['stimulus', 'neuron', 'time_point'])['delta_F_over_F0'].mean().reset_index()
    
    stimuli = sorted(mean_df['stimulus'].unique())
    neurons = sorted(mean_df['neuron'].unique())
    time_pts = sorted(mean_df['time_point'].unique())
    
    S, N, T = len(stimuli), len(neurons), len(time_pts)
    
    tensor_3d = np.zeros((S, N, T))
    for s_idx, stim in enumerate(stimuli):
        for n_idx, neur in enumerate(neurons):
            trace = mean_df[(mean_df['stimulus'] == stim) & (mean_df['neuron'] == neur)]
            if len(trace) == T:
                trace_sorted = trace.sort_values('time_point')
                # Interpolate if needed (simplified here)
                tensor_3d[s_idx, n_idx, :] = trace_sorted['delta_F_over_F0'].values
                
    # Create 2D matrix (Stimuli x Neurons) by taking max/mean across time or flattening
    # Using 'mean response' here as placeholder for 2D construction
    tensor_2d = np.mean(tensor_3d, axis=2)
    
    # Extract stimulus metadata mapping
    stimulus_info = {}
    if 'stim_name' in df.columns and 'stim_color' in df.columns:
        stimulus_info = df[['stimulus', 'stim_name', 'stim_color']].drop_duplicates().set_index('stimulus').to_dict('index')
    
    return tensor_2d, tensor_3d, stimuli, neurons, stimulus_info
