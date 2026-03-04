import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorly as tl
from tensorly.decomposition import parafac
from result_analysis.representation_clustering.tensor_prep import prepare_chemo_tensor
from result_analysis.representation_clustering.latent_cluster import cluster_latent_space
from result_analysis.representation_clustering.scaling_utils import calculate_adaptive_epsilon, apply_soft_scaling

# --- CONFIGURATION ---
TARGET_RANK = 3  # The rank you want to use for the final cluster comparison
MAX_SWEEP_RANK = 10
N_ITERATIONS = 20  # Consensus iterations for TCA
N_STABILITY_REPS = 5 # Repetitions to assess stability
DATA_PATH = "tests/cluster/20260226_useful_data.parquet"
OUTPUT_BASE = "tests/cluster"

def run_research_driver():
    tl.set_backend('numpy')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE, f"diagnostics_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Neural Representation Clustering Research Driver ===")
    print(f"Output Directory: {output_dir}")
    
    # 1. Load Data
    print(f"\n[1/4] Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    _, tensor_3d, stimuli, neurons, stimulus_info, time_pts, on_window, off_window = prepare_chemo_tensor(df)
    S, N, T = tensor_3d.shape
    tensor_3d = tensor_3d.astype(np.float64)
    print(f"Tensor Shape: {tensor_3d.shape} (S: {S}, N: {N}, T: {T})")
    
    # 2. Rank Selection Sweep
    print(f"\n[2/4] Running Rank Selection Sweep (R=1 to {MAX_SWEEP_RANK})...")
    ranks = range(1, MAX_SWEEP_RANK + 1)
    
    # PCA Diagnostics
    flattened = tensor_3d.reshape(S, N * T)
    scaler = StandardScaler()
    scaled_flattened = scaler.fit_transform(flattened)
    pca_full = PCA().fit(scaled_flattened)
    exp_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    # TCA Diagnostics Data
    tca_methods = ['none', 'standard', 'soft']
    tca_errors = {m: [] for m in tca_methods}
    
    eps = calculate_adaptive_epsilon(tensor_3d)
    print(f"Calculated Adaptive Epsilon: {eps:.6f}")
    
    for r in ranks:
        for m in tca_methods:
            # Simple error calculation for sweep
            if m == 'standard':
                eff = StandardScaler().fit_transform(flattened).reshape(S, N, T)
            elif m == 'soft':
                eff = apply_soft_scaling(tensor_3d, eps)
            else:
                eff = tensor_3d
                
            weights, factors = parafac(eff, rank=r, init='random', n_iter_max=500)
            rec = tl.cp_to_tensor((weights, factors))
            err = np.linalg.norm(eff - rec) / np.linalg.norm(eff)
            tca_errors[m].append(err)
        print(f"  Rank {r} processed.")

    # Plot Rank Selection
    fig_rank, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Panel A: PCA Scree
    axes[0].bar(range(1, 16), exp_var[:15], alpha=0.5, align='center', label='Individual', color='teal')
    axes[0].step(range(1, 16), cum_var[:15], where='mid', label='Cumulative', color='crimson', lw=2)
    axes[0].axhline(y=0.9, color='gray', linestyle='--', label='90% Variance')
    axes[0].set_title("PCA Scree Plot & Cumulative Variance")
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Variance Explained")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel B: TCA Error
    for m in tca_methods:
        axes[1].plot(ranks, tca_errors[m], 'o-', label=f'TCA ({m})')
    axes[1].set_title("TCA Reconstruction Error Comparison")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Relative Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_rank.savefig(os.path.join(output_dir, "01_rank_selection.png"), dpi=300)
    print("Saved rank selection plot: 01_rank_selection.png")

    # 3. Method Comparison at TARGET_RANK
    print(f"\n[3/4] Running Method Comparison at Rank={TARGET_RANK}...")
    methods = [
        ('PCA (Std)', 'pca', 'standard'),
        ('TCA (Raw)', 'tca', 'none'),
        ('TCA (Std)', 'tca', 'standard'),
        ('TCA (Soft)', 'tca', 'soft')
    ]
    
    results = {}
    for name, m_type, m_scale in methods:
        print(f"  Running {name}...")
        results[name] = cluster_latent_space(
            tensor_3d, method=m_type, n_comp=TARGET_RANK, 
            scaling=m_scale, soft_scaling_eps=eps, n_iterations=N_ITERATIONS
        )
        
    # Plot Comparison
    fig_comp, axes = plt.subplots(1, 4, figsize=(26, 6))
    for i, (name, _, _) in enumerate(methods):
        labels, k, score, Z, comp = results[name]
        ax = axes[i]
        scatter = ax.scatter(comp[:, 0], comp[:, 1], c=labels, cmap='tab20', s=100, edgecolors='k', alpha=0.8)
        ax.set_title(f"{name}\nK={k}, Score={score:.3f}")
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    fig_comp.savefig(os.path.join(output_dir, "02_method_comparison.png"), dpi=300)
    print("Saved method comparison: 02_method_comparison.png")

    # 4. Deep Factor Analysis (Using TCA Soft as default)
    print(f"\n[4/4] Generating Deep Factor Analysis for TCA (Soft-Scaled)...")
    labels, k, score, Z, stimulus_loadings, factors = cluster_latent_space(
        tensor_3d, method='tca', n_comp=TARGET_RANK, 
        scaling='soft', soft_scaling_eps=eps, return_factors=True
    )
    
    neuron_factors = factors[1]
    temporal_factors = factors[2]
    stim_factors = factors[0]
    stim_names = [stimulus_info.get(s, {}).get('stim_name', s) for s in stimuli]
    
    fig_factors, axes = plt.subplots(TARGET_RANK, 3, figsize=(22, 5 * TARGET_RANK))
    plt.subplots_adjust(hspace=0.4)
    
    for i in range(TARGET_RANK):
        # Neurons
        axes[i, 0].bar(range(N), neuron_factors[:, i], color='teal', alpha=0.7)
        axes[i, 0].set_xticks(range(N))
        axes[i, 0].set_xticklabels(neurons, rotation=90, fontsize=8)
        axes[i, 0].set_title(f"Comp {i+1}: Neuron Weights")
        
        # Temporal
        axes[i, 1].plot(time_pts, temporal_factors[:, i], color='crimson', lw=2)
        if on_window and off_window:
            axes[i, 1].axvspan(on_window[0], on_window[1], color='gray', alpha=0.1)
            axes[i, 1].axvspan(off_window[0], off_window[1], color='gray', alpha=0.2)
        axes[i, 1].set_title(f"Comp {i+1}: Temporal Shape")
        
        # Stimuli
        axes[i, 2].bar(range(S), stim_factors[:, i], color='orange', alpha=0.7)
        axes[i, 2].set_xticks(range(S))
        axes[i, 2].set_xticklabels(stim_names, rotation=90, fontsize=8)
        axes[i, 2].set_title(f"Comp {i+1}: Stimulus Loading")

    fig_factors.savefig(os.path.join(output_dir, "03_factor_analysis.png"), bbox_inches='tight', dpi=300)
    print("Saved factor analysis: 03_factor_analysis.png")
    
    # 5. Text Report
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write(f"Clustering Diagnostic Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Target Rank: {TARGET_RANK}\n\n")
        f.write(f"{'Method':<20} | {'Optimal K':<10} | {'Score':<10}\n")
        f.write("-" * 45 + "\n")
        for name, _, _ in methods:
            _, k, score, _, _ = results[name]
            f.write(f"{name:<20} | {k:<10} | {score:.4f}\n")
            
    print(f"\nReport generated successfully in: {output_dir}")

if __name__ == "__main__":
    run_research_driver()
