"""
Dispatcher functions for latent decomposition methods.

Each function takes a tensor and decomposition parameters, performs its
own decomposition and clustering, and returns a uniform:
    (Z, components, factors)

Where:
    Z          : scipy linkage matrix (S-1, 4) for hierarchical clustering
    components : (S, n_comp) array used for objective scoring / scatter plots
    factors    : method-specific extra data (None for PCA, list of arrays for TCA/dPCA)
"""
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import tensorly as tl
from tensorly.decomposition import parafac
from tslearn.metrics import dtw
from result_analysis.representation_clustering.scaling_utils import (
    calculate_adaptive_epsilon,
    apply_soft_scaling,
)


def standardize_tensor(
    tensor_3d: np.ndarray,
    orientation: str = 'stimuluswise',
) -> np.ndarray:
    """
    Standardize a tensor while preserving its (S, N, T) shape.

    Args:
        tensor_3d: Tensor with shape (S, N, T).
        orientation:
            - 'stimuluswise': flatten to (S, N*T), scale columns, reshape back.
            - 'neuronwise': reshape to (S*T, N), scale columns, reshape back.
    """
    S, N, T = tensor_3d.shape

    if orientation == 'stimuluswise':
        flattened = tensor_3d.reshape(S, N * T)
        return StandardScaler().fit_transform(flattened).reshape(S, N, T)

    if orientation == 'neuronwise':
        flattened = tensor_3d.transpose(0, 2, 1).reshape(S * T, N)
        scaled = StandardScaler().fit_transform(flattened)
        return scaled.reshape(S, T, N).transpose(0, 2, 1)

    raise ValueError(
        f"Unknown scaling orientation '{orientation}'. Supported: 'stimuluswise', 'neuronwise'."
    )


def apply_pca(
    tensor_3d: np.ndarray,
    n_comp: int = 3,
    scaling: str = 'standard',
    scaling_orientation: str = 'stimuluswise',
) -> tuple[np.ndarray, np.ndarray, None]:
    """
    Flatten the 3D tensor, scale, apply PCA, and compute Ward linkage.

    Args:
        tensor_3d: Array of shape (S, N, T).
        n_comp:    Number of principal components to retain.
        scaling:   'standard' (default) | 'none'. PCA is naturally scale-sensitive;
                   'standard' is recommended.

    Returns:
        Z:          Linkage matrix of shape (S-1, 4).
        components: PCA scores of shape (S, n_comp).
        factors:    Always None for PCA.
    """
    S, N, T = tensor_3d.shape
    effective_tensor = tensor_3d

    if scaling == 'standard':
        effective_tensor = standardize_tensor(tensor_3d, orientation=scaling_orientation)

    flattened = effective_tensor.reshape(S, N * T)

    pca = PCA(n_components=min(n_comp, S))
    components = pca.fit_transform(flattened)

    Z = linkage(components, method='ward')
    return Z, components, None


def apply_tca(
    tensor_3d: np.ndarray,
    n_comp: int = 3,
    n_iterations: int = 50,
    metric: str = 'dtw',
    scaling: str = 'none',
    soft_scaling_eps: float = None,
    scaling_orientation: str = 'stimuluswise',
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Apply CP (PARAFAC) decomposition and build consensus clustering linkage.

    Args:
        tensor_3d:        Array of shape (S, N, T).
        n_comp:           CP rank (number of components).
        n_iterations:     Number of consensus clustering iterations.
        metric:           Distance metric for noisy factor comparison: 'dtw' | 'euclidean'.
        scaling:          'none' (raw, recommended) | 'standard' | 'soft'.
        soft_scaling_eps: Epsilon for soft scaling; auto-computed if None and scaling='soft'.

    Returns:
        Z:          Consensus-based linkage matrix of shape (S-1, 4).
        components: Stimulus factor matrix of shape (S, n_comp).
        factors:    List [S_factors, N_factors, T_factors] from the CP decomposition.
    """
    S, N, T = tensor_3d.shape

    # --- Scaling ---
    if scaling == 'standard':
        effective_tensor = standardize_tensor(tensor_3d, orientation=scaling_orientation)
    elif scaling == 'soft':
        if soft_scaling_eps is None:
            soft_scaling_eps = calculate_adaptive_epsilon(tensor_3d)
        effective_tensor = apply_soft_scaling(tensor_3d, soft_scaling_eps)
    else:
        effective_tensor = tensor_3d

    # --- CP Decomposition ---
    tl.set_backend('numpy')
    weights, cp_factors = parafac(
        effective_tensor.astype(np.float64),
        rank=n_comp,
        init='random',
        normalize_factors=True,
    )
    stimulus_factors = cp_factors[0]  # (S, rank)

    # --- Consensus Clustering ---
    consensus_counts = np.zeros((S, S))
    for _ in range(n_iterations):
        noisy = stimulus_factors + np.random.normal(0, 0.05, stimulus_factors.shape)

        distances = np.zeros((S, S))
        for i in range(S):
            for j in range(i + 1, S):
                if metric == 'dtw':
                    d = dtw(noisy[i], noisy[j])
                else:
                    d = np.linalg.norm(noisy[i] - noisy[j])
                distances[i, j] = distances[j, i] = d

        condensed = distances[np.triu_indices(S, k=1)]
        Z_iter = linkage(condensed, method='ward')
        labels_iter = fcluster(Z_iter, min(3, S), criterion='maxclust')

        for i in range(S):
            for j in range(S):
                if labels_iter[i] == labels_iter[j]:
                    consensus_counts[i, j] += 1

    consensus_dist = 1.0 - (consensus_counts / n_iterations)
    condensed_consensus = consensus_dist[np.triu_indices(S, k=1)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = linkage(condensed_consensus, method='ward')

    # Return factors as a plain list for easy unpacking by the caller
    factors = [cp_factors[0], cp_factors[1], cp_factors[2]]
    return Z, stimulus_factors, factors


def apply_dpca(
    tensor_3d: np.ndarray,
    tensor_trial: np.ndarray = None,
    n_comp: int = 13,
    regularizer: float = 1e-4,
    use_reconstruction: bool = True,
    var_cum_threshold: float = 0.9,
    scaling_orientation: str = 'neuronwise',
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply Demixed PCA (dPCA) and extract stimulus marginalization components.

    Standard dPCA expects data in (Neurons, Stimuli, Time[, Trials]) format.
    We handle the transposition from our (S, N, T[, R]) convention internally.

    When `tensor_trial` is provided, dPCA uses per-trial data, which gives
    better noise estimates. Conditions with fewer trials must have NaN in the
    padding slots — dPCA ignores NaN values when computing marginal covariances.

    When `tensor_trial=None`, falls back to mean data (the 3D tensor), fitting
    dPCA without a trial dimension.

    Note: `regularizer='auto'` is NOT used because it requires cross-validation
    that fails with NaN-padded trial data. A fixed regularizer (default 1e-4)
    is used instead for robustness across both mean and trial fits.

    Args:
        tensor_3d:    Mean tensor of shape (S, N, T).
        tensor_trial: Optional trial tensor of shape (S, N, T, max_trials) with NaN
                      padding. Preferred over tensor_3d when available.
        n_comp:       Number of dPCA components per marginalization.
        regularizer:  Ridge regularization strength (default 1e-4). Increase if
                      dPCA is numerically unstable.

    Returns:
        Z:          Linkage matrix of shape (S-1, 4) based on Ward clustering of
                    stimulus components.
        components: Stimulus dPCA scores of shape (S, n_comp).
        factors:    Dict of dPCA encoder arrays, keyed by marginalization label
                    (e.g. 's', 't', 'st'). Callers can inspect full decomposition.
    """
    from dPCA.dPCA import dPCA as dPCA_lib

    S, N, T = tensor_3d.shape

    # --- Build the dPCA-format mean array: (N, S, T) ---
    mean_data = tensor_3d.transpose(1, 0, 2)  # (N, S, T)

    # Use fixed regularizer — 'auto' requires cross-val that breaks with NaN data
    dpca = dPCA_lib(labels='st', n_components=n_comp, regularizer=regularizer)
    dpca.protect = ['t']  # Protect time axis during data shuffling

    if tensor_trial is not None:
        # trial_data shape: (S, N, T, max_trials) -> (N, S, T, max_trials)
        trial_data = tensor_trial.transpose(1, 0, 2, 3)  # (N, S, T, R)
        dpca.fit(mean_data, trial_data)
    else:
        dpca.fit(mean_data)

    # --- Extract stimulus components ---
    transformed = dpca.transform(mean_data)
    
    if use_reconstruction:
        # Reconstruct (N, S, T) using encoders (P) and transformed data (Z)
        recon = np.zeros_like(mean_data)
        
        var_dict = dpca.explained_variance_ratio_
        kept_components = {'s': [], 'st': []}
        
        for label in ['s', 'st']:
            if label not in var_dict: continue
            
            # Find how many components are needed to hit the cumulative variance threshold
            # Normalized by the total variance explained by this marginalization alone
            marginal_vars = var_dict[label][:n_comp]
            total_margin_var = np.sum(marginal_vars)
            
            if total_margin_var > 0:
                cum_var = np.cumsum(marginal_vars) / total_margin_var
                keep_idx = np.where(cum_var >= var_cum_threshold)[0]
                n_keep = keep_idx[0] + 1 if len(keep_idx) > 0 else n_comp
            else:
                n_keep = 0
                
            kept_components[label] = list(range(n_keep))
            
            for i in range(n_keep):
                p_vec = dpca.P[label][:, i]          # encoder shape: (N,)
                z_mat = transformed[label][i]        # latent shape: (S, T)
                # Outer product addition: recon += P_i * Z_i
                recon += np.einsum('i,jk->ijk', p_vec, z_mat)
        
        # Transform reconstructed shape to (S, N, T)
        X_recon = recon.transpose(1, 0, 2)
        
        # Apply configurable scaling while preserving the reconstructed shape.
        X_recon = standardize_tensor(X_recon, orientation=scaling_orientation)
        stim_embedding = X_recon.reshape(S, N * T)
        
        factors = {
            'dpca_model':          dpca,
            'reconstructed':       X_recon,
            'kept_components':     kept_components,
            'variance_explained':  var_dict,
            'decoders':            dpca.D,
            'encoders':            dpca.P,
            'transformed':         transformed,
        }
    else:
        stim_components_raw = transformed['s']
        stim_embedding = stim_components_raw.mean(axis=-1).T
        factors = {
            'dpca_model':          dpca,
            'reconstructed':       None,
            'kept_components':     {'s': list(range(n_comp)), 'st': []},
            'variance_explained':  dpca.explained_variance_ratio_,
            'decoders':            dpca.D,
            'encoders':            dpca.P,
            'transformed':         transformed,
        }

    # --- Ward Linkage ---
    Z = linkage(stim_embedding, method='ward')
    
    # Generate 3D components for downstream visualization compatibility
    components_for_plot = PCA(n_components=min(3, S)).fit_transform(stim_embedding) if use_reconstruction else stim_embedding
    
    return Z, components_for_plot, factors
