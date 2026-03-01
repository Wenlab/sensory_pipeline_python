import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

def cluster_static_scalars(tensor_3d: np.ndarray, metric: str = 'peak') -> tuple:
    """
    Phase 2: Extract scalar (peak/mean), compute Euclidean distances with Ward linkage.
    """
    S, N, T = tensor_3d.shape
    
    # Step 2.1 Extract Scalar
    if metric == 'peak':
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
