import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import warnings
import tensorly as tl
from tensorly.decomposition import parafac
from tslearn.metrics import dtw
from result_analysis.representation_clustering.static_cluster import compute_gap_statistic

def cluster_latent_space(
    tensor_3d: np.ndarray, 
    method: str = 'pca', 
    n_comp: int = 3, 
    n_iterations: int = 50,
    metric: str = 'dtw',
    scoring: str = 'gap'
) -> tuple:
    """
    Phase 3b: Unified latent clustering with PCA and robust TCA consensus logic.
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
        # 1. TCA (CP Decomposition)
        # Tensorly uses (S, N, T) format
        weights, factors = parafac(tensor_3d, rank=n_comp, init='random', normalize_factors=True)
        stimulus_factors = factors[0] # Shape: (S, rank)
        components = stimulus_factors
        
        # 2. Consensus Clustering Iterations
        consensus_counts = np.zeros((S, S))
        
        for _ in range(n_iterations):
            # Add Gaussian noise for stability testing
            noisy_factors = stimulus_factors + np.random.normal(0, 0.05, stimulus_factors.shape)
            
            distances = np.zeros((S, S))
            for i in range(S):
                for j in range(i+1, S):
                    if metric == 'dtw':
                        # Flattening temporal/neuron factors for DTW isn't what's usually done for factors,
                        # but in this context, we apply it to the latent representation vectors.
                        d = dtw(noisy_factors[i], noisy_factors[j])
                    else:
                        d = np.linalg.norm(noisy_factors[i] - noisy_factors[j])
                    distances[i, j] = distances[j, i] = d
                    
            condensed_dist = distances[np.triu_indices(S, k=1)]
            Z_iter = linkage(condensed_dist, method='ward')
            
            # Using a fixed k (or small range) for co-occurrence evidence
            labels_iter = fcluster(Z_iter, min(3, S), criterion='maxclust')
            for i in range(S):
                for j in range(S):
                    if labels_iter[i] == labels_iter[j]:
                        consensus_counts[i, j] += 1
                        
        # 3. Final Partitioning on Consensus Matrix
        consensus_dist = 1.0 - (consensus_counts / n_iterations)
        condensed_consensus = consensus_dist[np.triu_indices(S, k=1)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Final linkage on the consensus-based distance matrix
            Z = linkage(condensed_consensus, method='ward')
            
        # Skip the standard linkage below since we already computed Z
        return _perform_objective_scoring(Z, components, S, scoring)

    else:
        components = scaled_flattened
        
    # Agglomerative clustering on the latent components
    Z = linkage(components, method='ward')
    return _perform_objective_scoring(Z, components, S, scoring)

def _perform_objective_scoring(Z, components, S, scoring) -> tuple:
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
