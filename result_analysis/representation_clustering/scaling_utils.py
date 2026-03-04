import numpy as np

def calculate_adaptive_epsilon(tensor_3d: np.ndarray, fraction: float = 0.1) -> float:
    """
    Calculates epsilon as a fraction of the median standard deviation across neurons.
    Standard deviation is calculated for each neuron across all stimuli and time points.
    
    Args:
        tensor_3d: Tensor of shape (S, N, T)
        fraction: Fraction of the median sigma to use for epsilon.
        
    Returns:
        float: The calculated adaptive epsilon.
    """
    S, N, T = tensor_3d.shape
    # Reshape to (N, S * T) to get variance per neuron across stimuli and time
    reshaped = tensor_3d.transpose(1, 0, 2).reshape(N, S * T)
    sigmas = np.std(reshaped, axis=1)
    
    # Filter out exactly zero sigmas if any to avoid bias
    valid_sigmas = sigmas[sigmas > 0]
    if len(valid_sigmas) == 0:
        return 1e-6
        
    epsilon = fraction * np.median(valid_sigmas)
    return float(epsilon)

def apply_soft_scaling(tensor_3d: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Applies Soft-Scaling: X_nist / (sigma_n + epsilon)
    
    Args:
        tensor_3d: Tensor of shape (S, N, T)
        epsilon: The smoothing constant.
        
    Returns:
        np.ndarray: Soft-scaled tensor of shape (S, N, T)
    """
    S, N, T = tensor_3d.shape
    # sigmas: (N, 1)
    reshaped = tensor_3d.transpose(1, 0, 2).reshape(N, S * T)
    sigmas = np.std(reshaped, axis=1, keepdims=True)
    
    scaled_reshaped = reshaped / (sigmas + epsilon)
    
    # Back to (S, N, T): (N, S*T) -> (N, S, T) -> (S, N, T)
    return scaled_reshaped.reshape(N, S, T).transpose(1, 0, 2)
