import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

def cluster_latent_space(tensor_3d: np.ndarray, method: str = 'pca', n_comp: int = 3) -> tuple:
    """
    Phase 3: Tensor decomposition (PCA/TCA) and latent space clustering.
    
    Args:
        tensor_3d: np.ndarray of shape (S, N, T)
        method: Latent decomposition method ('pca')
        n_comp: Number of components for decomposition
        
    Returns:
        tuple: (best_labels, best_k, best_score, Z, components)
    """
    S, N, T = tensor_3d.shape
    
    # Flatten to Stimuli x (Neurons*Time)
    # This captures the temporal dynamics of all neurons as features for each stimulus
    flattened = tensor_3d.reshape(S, N * T)
    
    if method == 'pca':
        # Limit n_components to min(n_comp, S, N*T)
        actual_n_comp = min(n_comp, S, N * T)
        pca = PCA(n_components=actual_n_comp)
        components = pca.fit_transform(flattened)
    else:
        # Placeholder for dPCA/TCA integration
        # For now, return flattened if method is unknown
        components = flattened
        
    # Hierarchical Clustering on latent components
    Z = linkage(components, method='ward')
    
    # Initialize values
    best_labels = fcluster(Z, 2, criterion='maxclust')
    best_k = 2
    best_score = -1.0
    
    # Search for optimal k (up to 10 or S-1)
    max_k = min(S, 10)
    if max_k > 2:
        for k in range(2, max_k):
            labels = fcluster(Z, k, criterion='maxclust')
            unique_labels = len(np.unique(labels))
            if 1 < unique_labels < S:
                try:
                    score = silhouette_score(components, labels)
                    if score > best_score:
                        best_score, best_k, best_labels = score, k, labels
                except Exception:
                    # In case silhouette score fails for some reason
                    pass
    else:
        # Default to 2 clusters if S is small
        if S > 1:
            best_labels = fcluster(Z, 2, criterion='maxclust')
            best_k = 2
            try:
                best_score = silhouette_score(components, best_labels)
            except Exception:
                best_score = -1.0
                
    return best_labels, best_k, best_score, Z, components
