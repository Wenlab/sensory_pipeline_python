import numpy as np
import pandas as pd
import copy
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def dtw_distance(seq_a, seq_b):
    seq_a = np.asarray(seq_a, dtype=float)
    seq_b = np.asarray(seq_b, dtype=float)
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = abs(seq_a[i - 1] - seq_b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m]

def compute_dtw_distance_matrix(series_list):
    size = len(series_list)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dist = dtw_distance(series_list[i], series_list[j])
            matrix[i, j] = matrix[j, i] = dist
    return matrix

def dtw_cluster(neuron_segments_dict, eps=12, min_samples=2, **kwargs):
    dtw_cluster_records = []
    dtw_distance_matrices = {}
    stim_name = kwargs.get('stim_name', {})
    for neuron, segments in neuron_segments_dict.items():
        for stimulus, trials in segments.items():
            waveforms = []
            trial_indices = []
            worm_key_list = []
            for idx, trial in enumerate(trials):
                values = trial.get("deltaFoverF_0")
                worm_key = trial.get('worm_key')
                trial_id = trial.get('segment_index')

                if values is None:
                    continue
                arr = np.asarray(values, dtype=float)
                if arr.size == 0:
                    continue
                waveforms.append(arr)
                worm_key_list.append(worm_key)
                trial_indices.append(trial_id)
            
            if not waveforms:
                continue

            dist_matrix = compute_dtw_distance_matrix(waveforms)
            dtw_distance_matrices[(neuron, stimulus)] = dist_matrix

            clustering = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples).fit(dist_matrix)
            labels = clustering.labels_

            # save result
            for i, (trial_id, worm_key) in enumerate(zip(trial_indices, worm_key_list)):
                dtw_cluster_records.append({
                        "neuron": neuron,
                        "stimulus": stimulus,
                        "stimulus_name": stim_name.get(stimulus, stimulus),
                        "worm_key": worm_key,
                        "trial_id": trial_id,
                        "cluster": int(labels[i]),
                        "is_noise": labels[i] == -1
                    })
            
    dtw_clusters_df = pd.DataFrame(dtw_cluster_records)
    noise_trials_df = (
        dtw_clusters_df[dtw_clusters_df["is_noise"]]
        .copy()
        .reset_index(drop=True)
    )

    return dtw_clusters_df, noise_trials_df, dtw_distance_matrices

def noise_clean(neuron_segments_dict, noise_trials_df, **kwargs):
    if noise_trials_df is None or noise_trials_df.empty:
        return copy.deepcopy(neuron_segments_dict), pd.DataFrame()
    stim_name = kwargs.get('stim_name', {})
    noise_keys = {
        (row.neuron, row.stimulus, row.worm_key, row.trial_id)
        for row in noise_trials_df.itertuples(index=False)
    }

    clean_dict = {}
    removed_records = []
    removed_dict = {}

    for neuron, segments in neuron_segments_dict.items():
        for stimulus, trials in segments.items():
            clean_trials = []
            removed_trials =[]
            for trial in trials:
                trial_key = (
                    neuron,
                    stimulus,
                    trial.get("worm_key"),
                    trial.get("segment_index")
                )

                if trial_key in noise_keys:
                    removed_records.append({
                        "neuron": neuron,
                        "stimulus": stimulus,
                        "stimulus_name": stim_name.get(stimulus, stimulus),
                        "worm_key": trial.get("worm_key"),
                        "segment_index": trial.get("segment_index"),
                    })
                    removed_trials.append(trial)
                    continue
                clean_trials.append(trial)
            if clean_trials:
                clean_dict.setdefault(neuron, {})[stimulus] = clean_trials
            if removed_trials:
                removed_dict.setdefault(neuron, {})[stimulus] = removed_trials

    removed_df = pd.DataFrame(removed_records)

    return clean_dict, removed_dict, removed_df

def clean_noise_trials(neuron_segments_dict, eps=12, min_samples=2, **kwargs):
    dtw_clusters_df, noise_trials_df, dtw_distance_matrices = dtw_cluster(neuron_segments_dict, eps=eps, min_samples=min_samples, **kwargs)

    clean_dict, removed_dict, removed_df = noise_clean(neuron_segments_dict, noise_trials_df, **kwargs)

    dtw_cluster_dict = {
        "cluster_df": dtw_clusters_df,
        "noise_trials_df": noise_trials_df,
        "removed_df": removed_df,
        "dtw_distance_matrices": dtw_distance_matrices
    }
    return clean_dict, removed_dict, dtw_cluster_dict

# plot result
def compute_k_distance_curve(distance_matrix, k_neighbors):
    if distance_matrix.size == 0:
        return np.array([])
    distances = []
    for row in distance_matrix:
        finite_row = row[np.isfinite(row)]
        finite_row = finite_row[finite_row > 0]
        if finite_row.size == 0:
            continue
        sorted_row = np.sort(finite_row)
        if sorted_row.size >= k_neighbors:
            distances.append(sorted_row[k_neighbors - 1])
        else:
            distances.append(sorted_row[-1])
    return np.sort(distances)

def plot_k_distance_curves(distance_matrices, k_neighbors=4, pdf_path=None, **kwargs):
    stim_name = kwargs.get('stim_name', {})
    if not distance_matrices:
        print("No distance matrices available for plotting.")
        return {}

    if pdf_path is None:
        pdf_path = "dtw_k_distance_plots.pdf"
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    curves = {}
    pdf = PdfPages(pdf_path)
    for (neuron, stimulus), matrix in distance_matrices.items():
        curve = compute_k_distance_curve(matrix, k_neighbors)
        curves[(neuron, stimulus)] = curve
        if curve.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curve, marker="o", linestyle="-")
        ax.set_title(f"k-distance curve (k={k_neighbors})\nNeuron: {neuron} | Stimulus: {stim_name.get(stimulus,stimulus)}")
        ax.set_xlabel("Points sorted by distance")
        ax.set_ylabel("Distance to k-th neighbor")
        ax.grid(alpha=0.1)
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print(f"Saved k-distance plots to {pdf_path}")
    return curves

