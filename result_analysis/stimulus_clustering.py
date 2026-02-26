import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from sklearn.decomposition import PCA
import tensorly as tl
from tensorly.decomposition import parafac
from dPCA import dPCA

def prepare_tensor(df: pd.DataFrame, smooth_sigma: float = 1.0, keep_trials: bool = False) -> tuple[np.ndarray, list, list]:
    """
    Pivot long-format df into an S x N x T tensor.
    If keep_trials=True, returns S x N x T x Trials (required for dPCA).
    Applies smoothing and independent Z-scoring.
    """
    if keep_trials:
        # Avoid averaging across trials, we need the variance
        df_mean = df.copy()
        # Group by trial index as well
    else:
        # Average trials
        df_mean = df.groupby(['stimulus', 'neuron', 'time_point'])['delta_F_over_F0'].mean().reset_index()
    
    stimuli_names = sorted(df_mean['stimulus'].unique())
    neuron_names = sorted(df_mean['neuron'].unique())
    time_pts = sorted(df_mean['time_point'].unique())
    
    S, N, T = len(stimuli_names), len(neuron_names), len(time_pts)
    tensor = np.zeros((S, N, T))
    
    for s_idx, stim in enumerate(stimuli_names):
        for n_idx, neur in enumerate(neuron_names):
            trace = df_mean[(df_mean['stimulus'] == stim) & (df_mean['neuron'] == neur)]
            if len(trace) == T:
                # Sort to ensure time is strictly increasing
                t_data = trace.sort_values('time_point')['delta_F_over_F0'].values
                
                # 1. Smooth
                if smooth_sigma > 0:
                    t_data = gaussian_filter1d(t_data, sigma=smooth_sigma)
                
                tensor[s_idx, n_idx, :] = t_data

    # 2. Independent Neuron Z-Scoring across S*T dimension
    for n_idx in range(N):
        # Flatten across S and T
        flat_slice = tensor[:, n_idx, :].flatten()
        if np.std(flat_slice) > 1e-8:
            z_slice = zscore(flat_slice)
            tensor[:, n_idx, :] = z_slice.reshape(S, T)
        else:
            # Basic structure for keep_trials=False. 
            # For keep_trials=True, you'll need padding or dropping to ensure consistent trial counts (K)
            # The output would be S x N x T x K. For now, returning the base average tensor as requested.
            pass
            
    return tensor, stimuli_names, neuron_names

def decompose_latent_features(tensor: np.ndarray, method: str = 'pca', n_components: int = 3) -> np.ndarray:
    """
    Decompose S x N x T tensor into an S x k feature matrix.
    methods: 'pca', 'tca', 'dpca'
    """
    S, N, T = tensor.shape
    
    if method == 'pca':
        # Flatten specific for Stimulus PCA: Average across time first, or unroll
        # Easiest baseline: just take mean across Time, then PCA across Neurons
        mean_time = np.mean(tensor, axis=2) # S x N
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(mean_time) # S x k
        return features
        
    elif method == 'tca':
        # Apply CP decomposition initialized with SVD for stability
        weights, factors = parafac(tensor, rank=n_components, init='svd', tol=1e-5)
        # factors is a list: [Stimulus_Factor, Neuron_Factor, Time_Factor]
        # We exclusively return the Stimulus factor matrix S x k
        return factors[0] 
        
    elif method == 'dpca':
        # dPCA requires a trial dimension (S x N x T x Trials) to compute noise covariance matrix.
        # Ensure keep_trials=True in prepare_tensor if routing to dpca.
        # Since we might have different trial counts, data should be formatted as N x S x T x Trials
        
        # NOTE: If tensor is 3D (trial averaged), dPCA will fail to demix properly.
        if len(tensor.shape) < 4:
           raise ValueError("dPCA requires tensor with trial dimension (S x N x T x Trials). Use keep_trials=True.")
           
        S, N, T, K = tensor.shape
        # Reshape to fit dpca expectation N x S x T x K
        data_n_s_t_k = np.transpose(tensor, (1, 0, 2, 3))
        
        # dPCA requires mean subtracted data
        mean_data = np.mean(data_n_s_t_k, axis=3, keepdims=True)
        data_centered = data_n_s_t_k - mean_data
        
        dpca = dPCA.dPCA(labels='st', n_components=n_components)
        dpca.protect = ['t']
        
        try:
            # fit_transform(X, X_trial)
            # X is the trial-averaged data N x S x T
            # X_trial is the full data N x S x T x K
            X_avg = np.mean(data_centered, axis=-1)
            Z = dpca.fit_transform(X_avg, data_centered)
            # return the stimulus specific components averaged over time -> S x k
            # Z is a dict: Z['s'] is N_components x S x T
            s_components = Z['s']
            return np.mean(s_components, axis=2).T
        except Exception as e:
             raise RuntimeError(f"dPCA execution failed: {e}")
             
    else:
        raise ValueError(f"Unknown method {method}")
