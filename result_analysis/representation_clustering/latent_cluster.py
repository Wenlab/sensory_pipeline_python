import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from result_analysis.representation_clustering.static_cluster import compute_gap_statistic

def cluster_latent_space(tensor_3d: np.ndarray, method: str = 'pca', n_comp: int = 3, scoring: str = 'gap') -> tuple:
    """
    Phase 3a: Preliminary temporal clustering using a flattened matrix.
    Future updates will integrate multi-dimensional TCA/dPCA.
    """
    S, N, T = tensor_3d.shape
    
    # Flatten to Stimuli x (Neurons * Time)
    flattened = tensor_3d.reshape(S, N * T)
    
    # CRITICAL: Standardize the flattened time-series before dimensionality reduction
    scaler = StandardScaler()
    scaled_flattened = scaler.fit_transform(flattened)
    
    if method == 'pca':
        # PCA on the standardized, flattened time-series
        pca = PCA(n_components=min(n_comp, S))
        components = pca.fit_transform(scaled_flattened)
    elif method == 'tca':
        # Placeholder for future TCA integration
        # components = apply_tca(tensor_3d, rank=n_comp)
        pass
    else:
        components = scaled_flattened
        
    # Agglomerative clustering on the latent components
    Z = linkage(components, method='ward')
    
    # Objective Scoring
    best_labels = fcluster(Z, 2, criterion='maxclust')
    best_k = 2
    best_score = -1
    
    max_k = min(S, 10)
    for k in range(2, max_k):
        labels = fcluster(Z, k, criterion='maxclust')
        if 1 < len(np.unique(labels)) < S:
            try:
                if scoring == 'gap':
                    score = compute_gap_statistic(components, labels, k)
                else:
                    score = silhouette_score(components, labels)
                    
                if score > best_score:
                    best_score, best_k, best_labels = score, k, labels
            except ValueError:
                pass
                
    return best_labels, best_k, best_score, Z, components
