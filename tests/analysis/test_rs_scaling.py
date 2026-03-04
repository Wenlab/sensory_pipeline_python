import numpy as np
import pytest
from result_analysis.representation_clustering.scaling_utils import calculate_adaptive_epsilon, apply_soft_scaling

def test_adaptive_scaling():
    # 2 Stimuli, 2 Neurons, 10 Time points
    np.random.seed(42)
    tensor = np.zeros((2, 2, 10))
    # Neuron 0: high variance
    tensor[0, 0, :] = 1.0 + np.random.normal(0, 0.1, 10)
    tensor[1, 0, :] = 2.0 + np.random.normal(0, 0.1, 10)
    # Neuron 1: low variance
    tensor[:, 1, :] = 0.001 + np.random.normal(0, 0.0001, 10)
    
    eps = calculate_adaptive_epsilon(tensor)
    # Expected: eps should be 0.1 * median(std_neuron0, std_neuron1)
    # std_neuron0 is ~0.5, std_neuron1 is ~0.0001. Median is mid way.
    assert eps > 0
    assert eps < 1.0
    
    scaled = apply_soft_scaling(tensor, eps)
    assert scaled.shape == tensor.shape
    
    # Verify scaling: High variance neuron should still have significant amplitude
    # Low variance neuron should be relatively squashed by eps
    std_orig = np.std(tensor.transpose(1, 0, 2).reshape(2, 20), axis=1)
    std_scaled = np.std(scaled.transpose(1, 0, 2).reshape(2, 20), axis=1)
    
    # The 'softness' means the ratio of stds should decrease
    assert std_scaled[0] > std_scaled[1]
