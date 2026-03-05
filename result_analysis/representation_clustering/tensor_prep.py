import numpy as np
import pandas as pd


def prepare_chemo_tensor(df: pd.DataFrame, combine_lr: bool = True) -> tuple[np.ndarray, np.ndarray, list, list, dict, list, tuple, tuple]:
    """
    Step 1.1 - 1.3: Clean, average, and reshape into 2D and 3D tensors.
    """
    df_processed = df.copy()

    if combine_lr:
        neuron_names = df_processed['neuron'].unique()
        lr_candidates = {}
        mapping = {}

        for neuron in neuron_names:
            if neuron.endswith('L') or neuron.endswith('R'):
                base_name = neuron[:-1]
                if base_name not in lr_candidates:
                    lr_candidates[base_name] = []
                lr_candidates[base_name].append(neuron)

        for base_name, neurons_lr in lr_candidates.items():
            if base_name == 'ASE':  # ASE L/R have distinct functions, keep separate
                continue
            if len(neurons_lr) == 2:
                has_left = any(n.endswith('L') for n in neurons_lr)
                has_right = any(n.endswith('R') for n in neurons_lr)
                if has_left and has_right:
                    for neuron in neurons_lr:
                        mapping[neuron] = base_name

        if mapping:
            # Replace neuron names with their base name
            df_processed['neuron'] = df_processed['neuron'].replace(mapping)

    # Group by stimulus, neuron, and time_point to average across trials/worms/L-R pairs
    # and deal with missing time_points using interpolation/ffill if necessary

    # For now, strict mean grouping
    mean_df = df_processed.groupby(['stimulus', 'neuron', 'time_point'])['delta_F_over_F0'].mean().reset_index()

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

    # Extract ON and OFF windows dynamically
    on_window = (5, 15)  # defaults
    off_window = (15, max(time_pts) if time_pts else 45)

    if 'start_time' in df.columns and 'end_time' in df.columns:
        start_t = df['start_time'].iloc[0]
        end_t = df['end_time'].iloc[0]
        on_window = (start_t, end_t)
        off_window = (end_t, max(time_pts))

    return tensor_2d, tensor_3d, stimuli, neurons, stimulus_info, time_pts, on_window, off_window


def prepare_chemo_trial_tensor(
    df: pd.DataFrame,
    combine_lr: bool = True
) -> tuple[np.ndarray, list, list]:
    """
    Builds a 4D trial-level tensor of shape (S, N, T, max_trials) with NaN padding.

    Each entry represents a single worm trial's response. Conditions with fewer
    trials than max_trials are padded with np.nan. This is the preferred input
    for dPCA, which ignores NaN values when computing marginal covariances.

    Args:
        df: Long-format DataFrame with columns:
            'neuron', 'stimulus', 'time_point', 'delta_F_over_F0', 'worm_key'
        combine_lr: If True, apply the same L/R neuron merging as prepare_chemo_tensor.

    Returns:
        trial_tensor: np.ndarray of shape (S, N, T, max_trials), dtype float64
        stimuli: sorted list of stimulus labels
        neurons: sorted list of neuron labels
    """
    df_processed = df.copy()

    if combine_lr:
        neuron_names = df_processed['neuron'].unique()
        mapping = {}
        lr_candidates = {}
        for neuron in neuron_names:
            if neuron.endswith('L') or neuron.endswith('R'):
                base_name = neuron[:-1]
                lr_candidates.setdefault(base_name, []).append(neuron)
        for base_name, neurons_lr in lr_candidates.items():
            if base_name == 'ASE':
                continue
            if len(neurons_lr) == 2:
                has_left = any(n.endswith('L') for n in neurons_lr)
                has_right = any(n.endswith('R') for n in neurons_lr)
                if has_left and has_right:
                    for neuron in neurons_lr:
                        mapping[neuron] = base_name
        if mapping:
            df_processed['neuron'] = df_processed['neuron'].replace(mapping)

    # Group by (stimulus, neuron, time_point, worm_key) -> mean within that trial
    # (handles L/R merging where two rows map to same worm_key)
    trial_df = (
        df_processed
        .groupby(['stimulus', 'neuron', 'time_point', 'worm_key'])['delta_F_over_F0']
        .mean()
        .reset_index()
    )

    stimuli = sorted(trial_df['stimulus'].unique())
    neurons = sorted(trial_df['neuron'].unique())
    time_pts = sorted(trial_df['time_point'].unique())
    S, N, T = len(stimuli), len(neurons), len(time_pts)

    # Find max number of trials across all (stimulus, neuron) combinations
    trials_per_cond = (
        trial_df.groupby(['stimulus', 'neuron'])['worm_key']
        .nunique()
    )
    max_trials = int(trials_per_cond.max()) if len(trials_per_cond) > 0 else 1

    trial_tensor = np.full((S, N, T, max_trials), np.nan, dtype=np.float64)

    stim_map = {s: i for i, s in enumerate(stimuli)}
    neur_map = {n: i for i, n in enumerate(neurons)}
    time_map = {t: i for i, t in enumerate(time_pts)}

    for (stim, neur, worm_key), grp in trial_df.groupby(['stimulus', 'neuron', 'worm_key']):
        s_idx = stim_map[stim]
        n_idx = neur_map[neur]
        grp_sorted = grp.sort_values('time_point')
        # Assign to the next available trial slot (first NaN slot)
        for t_slot in range(max_trials):
            if np.all(np.isnan(trial_tensor[s_idx, n_idx, :, t_slot])):
                for _, row in grp_sorted.iterrows():
                    t_idx = time_map[row['time_point']]
                    trial_tensor[s_idx, n_idx, t_idx, t_slot] = row['delta_F_over_F0']
                break

    return trial_tensor, stimuli, neurons
