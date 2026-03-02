import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

def extract_peak_features(tensor_3d: np.ndarray, time_pts: list, on_window: tuple, off_window: tuple) -> np.ndarray:
    """
    Extracts an N-dimensional feature vector representing each neuron's max absolute deviation.
    Evaluates across the entire response window (ON + OFF).
    Peak is defined as the maximum absolute deviation from baseline, preserving its original sign.
    """
    time_arr = np.array(time_pts)
    # Combine the windows to capture both ON and OFF responses
    mask = (time_arr >= on_window[0]) & (time_arr <= off_window[1])
    
    def _get_max_abs_dev(sub_tensor):
        if sub_tensor.shape[-1] == 0:
            return np.zeros(sub_tensor.shape[:-1])
        min_vals = np.min(sub_tensor, axis=-1)
        max_vals = np.max(sub_tensor, axis=-1)
        return np.where(np.abs(min_vals) > np.abs(max_vals), min_vals, max_vals)

    # Returning features (S x N)
    return _get_max_abs_dev(tensor_3d[:, :, mask])

def cluster_static_scalars(tensor_3d: np.ndarray, metric: str = 'peak', time_pts: list = None, on_window: tuple = None, off_window: tuple = None) -> tuple:
    """
    Phase 2: Extract scalar metrics and compute Euclidean distances with Ward linkage.
    """
    S, N, T = tensor_3d.shape
    
    # Step 2.1 Extract Scalar
    if metric == 'peak':
        if time_pts is not None and on_window is not None and off_window is not None:
            matrix_2d = extract_peak_features(tensor_3d, time_pts, on_window, off_window)
        else:
            matrix_2d = np.max(tensor_3d, axis=2)
    else:
        matrix_2d = np.mean(tensor_3d, axis=2)
        
    # Step 2.2 Agglomerative Ward Clustering
    Z = linkage(matrix_2d, method='ward')
    
    # Step 2.3 Objective Scoring
    best_k = 2
    best_score = -1
    best_labels = fcluster(Z, 2, criterion='maxclust')
    
    max_k = min(S, 10)
    for k in range(2, max_k):
        labels = fcluster(Z, k, criterion='maxclust')
        if len(np.unique(labels)) > 1 and len(np.unique(labels)) < S:
            try:
                 score = silhouette_score(matrix_2d, labels)
                 if score > best_score:
                     best_score, best_k, best_labels = score, k, labels
            except ValueError:
                 pass
                 
    return best_labels, best_k, best_score, Z
