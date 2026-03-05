"""
Latent space clustering entry point.

cluster_latent_space is a lean dispatcher: it delegates decomposition to
the appropriate function in decompositions.py, then runs objective scoring
on the returned linkage matrix.

Supported methods:
    'pca'  - Flattened + scaled PCA with Ward linkage
    'tca'  - CP decomposition with consensus clustering linkage
    'dpca' - Demixed PCA with stimulus marginalization components

Design reference: docs/plans/2026-03-05-dpca-integration-design.md
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
from result_analysis.representation_clustering.static_cluster import compute_gap_statistic
from result_analysis.representation_clustering.decompositions import (
    apply_pca,
    apply_tca,
    apply_dpca,
)


def cluster_latent_space(
    tensor_3d: np.ndarray,
    tensor_trial: np.ndarray = None,
    method: str = 'pca',
    n_comp: int = 3,
    n_iterations: int = 50,
    metric: str = 'dtw',
    scoring: str = 'gap',
    scaling: str = 'none',
    soft_scaling_eps: float = None,
    return_factors: bool = False,
) -> tuple:
    """
    Unified latent clustering dispatcher for neural data stimuli.

    Decomposes the stimulus tensor into a low-dimensional representation,
    builds a hierarchical linkage matrix, and determines the optimal number
    of clusters via objective scoring.

    Args:
        tensor_3d:        Mean tensor of shape (S, N, T).
        tensor_trial:     Optional 4D trial tensor of shape (S, N, T, max_trials)
                          with NaN padding. Used only when method='dpca'.
        method:           Decomposition method: 'pca' | 'tca' | 'dpca'.
        n_comp:           Number of components / rank.
        n_iterations:     Consensus clustering iterations (TCA only).
        metric:           Distance metric for consensus clustering: 'dtw' | 'euclidean'
                          (TCA only).
        scoring:          Cluster count scoring method: 'gap' | 'silhouette'.
        scaling:          Pre-decomposition scaling (PCA / TCA only):
                          'none' | 'standard' | 'soft'.
        soft_scaling_eps: Epsilon for soft scaling (computed adaptively if None).
        return_factors:   If True, append method-specific factors to the return tuple.

    Returns:
        Tuple of (labels, best_k, best_score, Z, components[, factors])
        - labels:      Cluster assignment per stimulus (length S).
        - best_k:      Optimal cluster count determined by scoring.
        - best_score:  Best objective score.
        - Z:           Linkage matrix (S-1, 4).
        - components:  Low-dimensional stimulus embedding (S, n_comp).
        - factors:     (Only when return_factors=True) Method-specific extra data.
                       None for PCA; list [S,N,T factors] for TCA;
                       dict of marginalizations for dPCA.
    """
    S = tensor_3d.shape[0]

    # --- Dispatch ---
    if method == 'pca':
        Z, components, factors = apply_pca(
            tensor_3d, n_comp=n_comp, scaling=scaling
        )
    elif method == 'tca':
        Z, components, factors = apply_tca(
            tensor_3d,
            n_comp=n_comp,
            n_iterations=n_iterations,
            metric=metric,
            scaling=scaling,
            soft_scaling_eps=soft_scaling_eps,
        )
        # Preserve the full CP factors list for return_factors
        tca_factor_list = factors
    elif method == 'dpca':
        Z, components, factors = apply_dpca(
            tensor_3d, tensor_trial=tensor_trial, n_comp=n_comp
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'pca', 'tca', 'dpca'."
        )

    # --- Objective Scoring ---
    results = _perform_objective_scoring(Z, components, S, scoring)

    if return_factors:
        if method == 'tca':
            # Backward-compatible: return stimulus factor array directly (S, n_comp)
            return (*results, tca_factor_list[0])
        return (*results, factors)

    return results


def _perform_objective_scoring(
    Z: np.ndarray,
    components: np.ndarray,
    S: int,
    scoring: str,
) -> tuple:
    """Find the best cluster cut on the linkage matrix via gap statistic or silhouette."""
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
