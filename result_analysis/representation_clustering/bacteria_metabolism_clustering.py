import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import os

def compute_gap_statistic(data: np.ndarray, labels: np.ndarray, k: int, n_refs: int = 20) -> float:
    """
    Computes the Gap Statistic for a given clustering.
    Uses uniform reference distributions bounding the actual data.
    """
    if k <= 1 or len(np.unique(labels)) < k:
        return 0.0
        
    def _compute_Wk(cluster_data, cluster_labels):
        wk = 0.0
        for i in np.unique(cluster_labels):
            pts = cluster_data[cluster_labels == i]
            if len(pts) > 0:
                # Sum of squared distances to the centroid
                centroid = np.mean(pts, axis=0)
                wk += np.sum(np.linalg.norm(pts - centroid, axis=1)**2)
        return wk

    # Wk for actual data
    wk_actual = _compute_Wk(data, labels)
    if wk_actual == 0:
        return 0.0

    # Wk for reference data
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    ref_wks = []
    
    for _ in range(n_refs):
        ref_data = np.random.uniform(mins, maxs, data.shape)
        Z_ref = linkage(ref_data, method='ward')
        ref_labels = fcluster(Z_ref, k, criterion='maxclust')
        ref_wk = _compute_Wk(ref_data, ref_labels)
        if ref_wk > 0:
            ref_wks.append(ref_wk)
            
    if not ref_wks:
        return 0.0
        
    expected_log_wk = np.mean(np.log(ref_wks))
    gap = expected_log_wk - np.log(wk_actual)
    return gap

def load_metabolism_data(file_path: str, sample_list: list = None):
    """
    Loads and prepares the metabolism Excel file.
    Optionally filters by a list of bacteria names.
    """
    df = pd.read_excel(Path(file_path))
    
    # Set the first column as index (assuming it contains bacteria names)
    if 'Unnamed: 0' in df.columns:
        df = df.set_index('Unnamed: 0')
    else:
        df = df.set_index(df.columns[0])
        
    # Apply filtering if a sample list is provided
    if sample_list is not None:
        original_count = len(df)
        # Intersection to ensure we only look for samples that exist
        valid_samples = [s for s in sample_list if s in df.index]
        df = df.loc[valid_samples]
        print(f"Filtered samples: {len(df)} remaining out of {original_count} original.")
    else:
        print(f"Loaded ALL data: {df.shape[0]} bacteria.")

    print(f"Data shape: {df.shape[0]} bacteria x {df.shape[1]} compounds.")
    return df


def build_reference_clustering(
    df: pd.DataFrame,
    n_comp: int = 3,
    scoring: str = 'gap',
) -> dict:
    """
    Build a structured clustering payload for downstream neural-method benchmarking.

    Args:
        df: Metabolism matrix indexed by sample name.
        n_comp: Number of PCA components to retain for the embedding.
        scoring: Cluster count selection rule. Supports 'gap' and 'silhouette'.

    Returns:
        Dict containing ordered sample names, labels, selected cluster count,
        PCA embedding, linkage matrix, and model metadata.
    """
    if len(df) < 2:
        raise ValueError("At least 2 samples are required for clustering analysis.")

    pca_df, Z, pca_model = perform_pca_clustering(df, n_comp=n_comp, scoring=scoring)
    component_cols = [col for col in pca_df.columns if col.startswith('PC')]
    embedding = pca_df.set_index('Bacteria')[component_cols]
    labels = pca_df['Cluster'].tolist()

    return {
        'samples': pca_df['Bacteria'].tolist(),
        'labels': labels,
        'n_clusters': len(set(labels)),
        'embedding': embedding,
        'linkage': Z,
        'pca_model': pca_model,
        'scoring': scoring,
    }

def perform_pca_clustering(df: pd.DataFrame, n_comp: int = 3, scoring: str = 'gap'):
    """
    Standardizes data, applies PCA, and performs Ward hierarchical clustering.
    Returns (pca_df, linkage_matrix, pca_model).
    """
    if len(df) < 2:
        raise ValueError("At least 2 samples are required for clustering analysis.")

    # 1. Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # 2. PCA
    # Adjust n_components if samples < n_comp
    actual_n_comp = min(n_comp, len(df), df.shape[1])
    pca = PCA(n_components=actual_n_comp)
    pca_results = pca.fit_transform(scaled_data)

    # 3. Clustering (Ward's Method on PCA space)
    Z = linkage(pca_results, method='ward')

    # 4. Automatic K Selection
    best_k = 1 if scoring == 'gap' else 2
    best_score = -np.inf
    max_k = min(len(df), 11)

    for k in range(2 if scoring == 'silhouette' else 1, max_k):
        labels = fcluster(Z, k, criterion='maxclust')
        
        if scoring == 'gap':
            score = compute_gap_statistic(pca_results, labels, k)
        else: # silhouette
            if len(np.unique(labels)) > 1:
                score = silhouette_score(pca_results, labels)
            else:
                score = -1

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Optimal clusters found: {best_k} ({scoring.capitalize()} Score: {best_score:.3f})")
    
    # 5. Assemble result DataFrame
    col_names = [f'PC{i+1}' for i in range(actual_n_comp)]
    pca_df = pd.DataFrame(
        data=pca_results,
        columns=col_names,
        index=df.index
    )
    pca_df.index.name = 'Bacteria'
    pca_df = pca_df.reset_index()
    pca_df['Cluster'] = fcluster(Z, best_k, criterion='maxclust').astype(str)
    
    # Ensure PC3 exists for 3D plotting even if variance is low
    if 'PC3' not in pca_df.columns and actual_n_comp < 3:
         pca_df['PC3'] = 0 

    return pca_df, Z, pca

def plot_clustering_results(pca_df: pd.DataFrame, Z: np.ndarray, pca_model: PCA, save_path: str = None):
    """Generates Dendrogram and 2D Static Plot."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Dendrogram
    dendrogram(Z, labels=pca_df['Bacteria'].values, ax=axes[0], leaf_rotation=90)
    axes[0].set_title('Hierarchical Clustering (Dendrogram)')
    axes[0].set_ylabel('Ward Distance')

    # 2D PCA Plot
    sns.scatterplot(
        data=pca_df, x='PC1', y='PC2', hue='Cluster', 
        style='Cluster', s=100, palette='viridis', ax=axes[1]
    )
    # Labels
    for i, txt in enumerate(pca_df['Bacteria']):
        axes[1].annotate(txt, (pca_df.PC1[i], pca_df.PC2[i]), fontsize=8, alpha=0.7)
    
    var_exp = pca_model.explained_variance_ratio_[:2].sum()
    axes[1].set_title(f'2D PCA Projection (Exp. Var: {var_exp:.2%})')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Static figure saved to: {save_path}")
    plt.show()

def plot_interactive_3d(pca_df: pd.DataFrame, save_path: str = None):
    """Generates an interactive Plotly 3D scatter plot."""
    fig = px.scatter_3d(
        pca_df, 
        x='PC1', y='PC2', z='PC3',
        color='Cluster',
        text='Bacteria',
        title='3D Metabolic Space (Interactive)',
        labels={'Cluster': 'Metabolic Group'},
        template='plotly_dark'
    )
    fig.update_traces(marker=dict(size=5))
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive figure saved to: {save_path}")
    fig.show()

def run_bacteria_analysis(file_path: str, sample_list: list = None, scoring: str = 'gap'):
    """Main execution flow."""
    # Load with optional filter
    df = load_metabolism_data(file_path, sample_list=sample_list)
    
    # Analyze
    pca_df, Z, pca_model = perform_pca_clustering(df, scoring=scoring)
    
    # Save data
    csv_path = file_path.replace('matrix.xlsx', 'clustered_bacteria.csv')
    pca_df.to_csv(csv_path, index=False)
    print(f"Success! Results saved to: {csv_path}")

    # Define figure paths
    static_fig_path = file_path.replace('matrix.xlsx', 'bacteria_pca_static.png')
    interactive_fig_path = file_path.replace('matrix.xlsx', 'bacteria_pca_3d.html')

    # Visualize and Save Figures
    plot_clustering_results(pca_df, Z, pca_model, save_path=static_fig_path)
    plot_interactive_3d(pca_df, save_path=interactive_fig_path)
    
    return pca_df

def launch_interactive_dashboard(file_path: str, sample_list: list = None):
    """Launches the Dash-based interactive PCA dashboard without clustering overhead."""
    df = load_metabolism_data(file_path, sample_list=sample_list)
    
    # 1. Pure PCA Projection up to 20 dimensions
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    actual_n_comp = min(20, len(df), df.shape[1])
    pca_model = PCA(n_components=actual_n_comp)
    pca_results = pca_model.fit_transform(scaled_data)
    
    col_names = [f'PC{i+1}' for i in range(actual_n_comp)]
    pca_df = pd.DataFrame(data=pca_results, columns=col_names, index=df.index)
    pca_df.index.name = 'Bacteria'
    pca_df = pca_df.reset_index()
    
    # 2. Launch Dashboard
    try:
        from result_plot.vis_metabolism_pca import create_metabolism_dashboard
    except ImportError:
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from result_plot.vis_metabolism_pca import create_metabolism_dashboard
        
    app = create_metabolism_dashboard(pca_df, pca_model)
    print("Launching Dash server... (Press Ctrl+C to stop)")
    app.run(debug=True)

# --- Execution ---
if __name__ == "__main__":
    DATA_PATH = r"H:\Process_temporary\WJH\sensory_pipeline_python\data\bacteria\metabolism\matrix.xlsx"
    
    # Example: Filter for specific bacteria (uncomment to use)
    # samples_to_keep = ['Bact_A', 'Bact_B', 'Bact_C'] 
    
    results = run_bacteria_analysis(DATA_PATH, sample_list=None)
    
    # To launch the interactive Dash dashboard, uncomment to use:
    # launch_interactive_dashboard(DATA_PATH, sample_list=None)
