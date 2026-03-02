import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

def compute_gap_statistic(data: np.ndarray, labels: np.ndarray, k: int, n_refs: int = 10) -> float:
    """
    Computes the Gap Statistic for a given clustering.
    Uses uniform reference distributions bounding the actual data.
    """
    if k <= 1 or len(np.unique(labels)) < k:
        return 0.0
        
    def _compute_Wk(cluster_data, cluster_labels):
        wk = 0.0
        for i in np.unique(cluster_labels):
            pts = cluster_data[cluster_labels == i]
            if len(pts) > 0:
                # Sum of squared distances to the centroid
                centroid = np.mean(pts, axis=0)
                wk += np.sum(np.linalg.norm(pts - centroid, axis=1)**2)
        return wk

    # Wk for actual data
    wk_actual = _compute_Wk(data, labels)
    if wk_actual == 0:
        return 0.0

    # Wk for reference data
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    ref_wks = []
    
    for _ in range(n_refs):
        ref_data = np.random.uniform(mins, maxs, data.shape)
        Z_ref = linkage(ref_data, method='ward')
        ref_labels = fcluster(Z_ref, k, criterion='maxclust')
        ref_wk = _compute_Wk(ref_data, ref_labels)
        if ref_wk > 0:
            ref_wks.append(ref_wk)
            
    if not ref_wks:
        return 0.0
        
    expected_log_wk = np.mean(np.log(ref_wks))
    gap = expected_log_wk - np.log(wk_actual)
    return gap

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

def cluster_static_scalars(tensor_3d: np.ndarray, metric: str = 'peak', time_pts: list = None, on_window: tuple = None, off_window: tuple = None, scoring: str = 'gap') -> tuple:
    """
    Phase 2: Extract scalar metrics and compute Euclidean distances with Ward linkage.
    """
    S, N, T = tensor_3d.shape
    
    # Step 2.1 Extract Scalar
    if metric == 'peak':
        if time_pts is None or on_window is None or off_window is None:
            raise ValueError("time_pts, on_window, and off_window must be provided to accurately calculate peak features.")
        matrix_2d = extract_peak_features(tensor_3d, time_pts, on_window, off_window)
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
                 if scoring == 'gap':
                     score = compute_gap_statistic(matrix_2d, labels, k)
                 else:
                     score = silhouette_score(matrix_2d, labels)
                     
                 if score > best_score:
                     best_score, best_k, best_labels = score, k, labels
            except ValueError:
                 pass
                 
    return best_labels, best_k, best_score, Z
