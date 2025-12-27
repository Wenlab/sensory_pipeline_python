#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import zscore

import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.get_symmetric_neuron import get_symmetric_neuron
from utils.parse_stimulus_info import group_and_sort_stimuli
from data_load.get_stimulus_info import *


#%%
def relplot_mean_signal(neuron_segments_df, 
                        height=2, 
                        aspect=2, 
                        col_wrap=2,
                        kind='line',
                        errorbar='se',
                        stimulus_color_map=None,
                        stimulus_info_dict=None,
                        save_path=None,
                        **kwargs):
        """
        Plot mean signal trends for each neuron and stimulus type using seaborn's relplot.
        
        Args:
            height (float, optional): Height of each facet. Default is 2.
            aspect (float, optional): Aspect ratio of each facet. Default is 2.
            col_wrap (int, optional): Number of columns to wrap the facets. Default is 2.
            kind (str, optional): Type of plot to draw. Default is 'line'.
            errorbar (str or callable, optional): Error bar type. Default is 'se' (standard error).
            save_path (str, optional): Path to save the plot. If None, the plot is not saved.
            **kwargs: Additional keyword arguments passed to seaborn's relplot.
        """
        g = sns.relplot(
            data=neuron_segments_df,
            x='time_point',
            y='delta_F_over_F0',
            col='neuron',
            hue='stimulus',
            palette= stimulus_color_map,
            kind=kind,
            errorbar=errorbar,
            height=height,
            aspect=aspect,
            col_wrap=col_wrap,
            **kwargs
        )
        for ax in g.axes.flat:
            ax.axvline(x=5, color='orange', linestyle='--', label='Stimulus Onset')
            ax.axvline(x=15, color='gray', linestyle='--', label='Stimulus Offset')

        g.figure.suptitle('')
        g.set_axis_labels('Time(s)', 'ΔF/F0')
        g.set_xlabels('Time(s)')
        g.set_ylabels('ΔF/F0')
        g.set_titles('{col_name}')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        
        return g

    
def get_cluster_order(neuron_segments_df, all_neurons, cluster_stimulus=None):
    stimulus_neuron_counts = neuron_segments_df.groupby('stimulus')['neuron'].nunique()
    cluster_stimulus = cluster_stimulus if cluster_stimulus else stimulus_neuron_counts.idxmax()
    print(f"cluster based on {cluster_stimulus}")
    
    df_cluster_stimulus = neuron_segments_df[neuron_segments_df['stimulus'] == cluster_stimulus]
    
    if df_cluster_stimulus.empty:
        print(f"No data available for stimulus type '{cluster_stimulus}' to perform clustering.")
        return all_neurons, None, cluster_stimulus

    cluster_matrix_partial = df_cluster_stimulus.pivot_table(
        index='neuron',
        columns='time_point',
        values='delta_F_over_F0',
        aggfunc='mean'
    )

    time_columns = cluster_matrix_partial.columns
    cluster_matrix_full = pd.DataFrame(index=all_neurons, columns=time_columns)

    present_neurons = set(cluster_matrix_partial.index)
    
    neurons_imputed_with_zeros = []

    # Fill in the response_matrix with available data
    for neuron in all_neurons:
        if neuron in present_neurons:
            cluster_matrix_full.loc[neuron] = cluster_matrix_partial.loc[neuron]
        else:
            symmetric_neuron = get_symmetric_neuron(neuron)
            if symmetric_neuron in present_neurons:
                cluster_matrix_full.loc[neuron] = cluster_matrix_partial.loc[symmetric_neuron]
            else:
                cluster_matrix_full.loc[neuron] = 0
                neurons_imputed_with_zeros.append(neuron)
    
    cluster_matrix_full = pd.to_numeric(cluster_matrix_full.stack(), errors='coerce').unstack().fillna(0).astype(float)
    
    # Z-score normalization across time points for each neuron
    cluster_matrix_zscored = zscore(cluster_matrix_full, axis=1)
    cluster_matrix_zscored = np.nan_to_num(cluster_matrix_zscored)
    cluster_matrix_zscored = pd.DataFrame(cluster_matrix_zscored, index=cluster_matrix_full.index, columns=cluster_matrix_full.columns)

    neurons_to_cluster = [n for n in all_neurons if n not in neurons_imputed_with_zeros]
    
    if len(neurons_to_cluster) > 1:
        matrix_for_clustering = cluster_matrix_zscored.loc[neurons_to_cluster]
        distance_matrix = pdist(matrix_for_clustering, metric='correlation')
        linkage_matrix = linkage(distance_matrix, method='ward')
        cluster_order_indices = leaves_list(linkage_matrix)

        clustered_part = matrix_for_clustering.index[cluster_order_indices].tolist()
        neurons_in_cluster_order = clustered_part + neurons_imputed_with_zeros
    else:
        neurons_in_cluster_order = neurons_to_cluster + neurons_imputed_with_zeros
        
    return neurons_in_cluster_order, cluster_matrix_full, cluster_stimulus


def plot_covariance_matrix(cluster_matrix_full, neurons_in_cluster_order, cluster_stimulus, save_folder):
    if cluster_matrix_full is None:
        return None
        
    ordered_matrix = cluster_matrix_full.loc[neurons_in_cluster_order]
    covariance_matrix = np.cov(ordered_matrix.values)
    n_dim = len(neurons_in_cluster_order)

    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(covariance_matrix, dtype=bool), k=1)
    sns.heatmap(covariance_matrix, 
                mask=mask,
                annot=False, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                xticklabels=neurons_in_cluster_order,
                yticklabels=neurons_in_cluster_order,
                cbar_kws={'label': 'Covariance'})
    
    plt.title(f'Covariance Matrix (clustered by {cluster_stimulus})\nDimensions: {n_dim} x {n_dim}')
    plt.xlabel('Neurons')
    plt.ylabel('Neurons')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save covariance heatmap
    plt.savefig(f"{save_folder}/covariance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_folder}/covariance_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    return covariance_matrix


def draw_mean_signal_cluster(neuron_segments_df,
                             y_offset=1.5,
                             fig_width=12,
                             fig_height_per_neuron = 0.5,
                             stimulus_info_dict=None,
                             cluster_stimulus=None,
                             save_folder=None,
                             plot_covariance=True,
                             custom_order=None
                             ):
    if neuron_segments_df.empty:
        print("neuron_segments_df is empty. Nothing to plot.")
        return None

    all_neurons = sorted(neuron_segments_df['neuron'].unique())
    # Group and sort stimuli by concentration
    if stimulus_info_dict:
        grouped_stimuli = group_and_sort_stimuli(stimulus_info_dict)
        # Flatten to get ordered stimulus list
        stimulus_types = []
        for compound_name, stimulus_codes in grouped_stimuli:
            stimulus_types.extend(stimulus_codes)
        
        # Filter to only include stimuli that exist in the data
        available_stimuli = set(neuron_segments_df['stimulus'].unique())
        stimulus_types = [s for s in stimulus_types if s in available_stimuli]
    else:
        stimulus_types = sorted(neuron_segments_df['stimulus'].unique())
        grouped_stimuli = [(s, [s]) for s in stimulus_types]  # Each stimulus as its own group

    if not custom_order:
        neurons_in_cluster_order, cluster_matrix_full, cluster_stimulus = get_cluster_order(neuron_segments_df, all_neurons, cluster_stimulus)
        
        covariance_matrix = None
        if plot_covariance and save_folder:
            covariance_matrix = plot_covariance_matrix(cluster_matrix_full, neurons_in_cluster_order, cluster_stimulus, save_folder)
    else:
        neurons_in_cluster_order = custom_order
        cluster_matrix_full = None
        covariance_matrix = None
        
    n_dim = len(neurons_in_cluster_order)

    # PLOTTING
    n_neurons = len(neurons_in_cluster_order)
    n_stimuli = len(stimulus_types)

    total_height = n_neurons * fig_height_per_neuron
    if total_height < 6: total_height = 6
    
    # Reserve fixed space for labels (e.g. 1.5 inches)
    label_height_inches = 1.5
    bottom_fraction = label_height_inches / total_height

    fig, axes = plt.subplots(1, n_stimuli, figsize=(fig_width, total_height), sharey=True, squeeze=False)
    axes = axes.flatten()
    
    # Adjust layout to leave space for labels at bottom
    plt.subplots_adjust(wspace=0.05, bottom=bottom_fraction)

    x_min = neuron_segments_df['time_point'].min()
    x_max = neuron_segments_df['time_point'].max()

    # Collect x_centers for labels
    stimulus_x_centers = []

    for j, stimulus in enumerate(stimulus_types):
        ax = axes[j]
        ax.set_facecolor('none')
        
        # Get axis position for label alignment
        bbox = ax.get_position()
        x_center = bbox.x0 + bbox.width / 2
        stimulus_x_centers.append(x_center)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add stimulus lines
        ax.axvline(x=5, color='#E6A61C', linestyle='--', linewidth=1.5, zorder=0)
        ax.axvline(x=15, color='#76642E', linestyle='--', linewidth=1.5, zorder=0)
        
        # Iterate neurons
        for i, neuron in enumerate(neurons_in_cluster_order):
            # Calculate offset
            current_offset = i * y_offset
            
            # Z-order: lower neurons (smaller i) should be in front?
            # If i=0 is bottom, and we want it to cover i=1 (above).
            # So i=0 should be drawn LAST (highest zorder).
            z_order = n_neurons - i

            # Add baseline at y=0 relative to offset
            ax.plot([x_min, x_max], [current_offset, current_offset], 
                    color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=z_order+0.05)

            df_subset = neuron_segments_df[(neuron_segments_df['neuron']==neuron) & (neuron_segments_df['stimulus']==stimulus)]
            
            mean_trace = None
            sem_trace = None
            is_symmetric = False
            
            if not df_subset.empty:
                mean_trace = df_subset.groupby('time_point')['delta_F_over_F0'].mean()
                sem_trace = df_subset.groupby('time_point')['delta_F_over_F0'].sem()
            else:
                symmetric_neuron = get_symmetric_neuron(neuron)
                df_symmetric = neuron_segments_df[(neuron_segments_df['neuron']==symmetric_neuron) & (neuron_segments_df['stimulus']==stimulus)]
                if not df_symmetric.empty:
                    mean_trace = df_symmetric.groupby('time_point')['delta_F_over_F0'].mean()
                    sem_trace = df_symmetric.groupby('time_point')['delta_F_over_F0'].sem()
                    is_symmetric = True
            
            if mean_trace is not None:
                y_values = mean_trace.values + current_offset
                y_upper = y_values + sem_trace.values
                y_lower = y_values - sem_trace.values
                
                color = 'blue'
                linestyle = '-'
                if is_symmetric:
                    linestyle = '-.'
                
                # Fill
                # White fill to block background
                ax.fill_between(mean_trace.index, y_lower, y_upper, color='white', alpha=1.0, zorder=z_order)
                # Color fill
                ax.fill_between(mean_trace.index, y_lower, y_upper, color='gray', alpha=0.3, zorder=z_order+0.1)
                # Line
                ax.plot(mean_trace.index, y_values, color=color, linestyle=linestyle, zorder=z_order+0.2)
                
            # Add neuron label on the first subplot (even if trace is missing)
            if j == 0:
                # Label at the baseline of the trace (current_offset)
                ax.text(x_min - (x_max-x_min)*0.05, current_offset, neuron, 
                        ha='right', va='center', fontsize=10)

        # Add scale bar on the last subplot
        if j == n_stimuli - 1:
            scale_value = 1.0
            bar_x = x_max - (x_max - x_min) * 0.02
            top_neuron_idx = len(neurons_in_cluster_order) - 1
            bar_bottom = top_neuron_idx * y_offset + y_offset * 0.5
            bar_top = bar_bottom + scale_value
            
            ax.plot([bar_x, bar_x], [bar_bottom, bar_top], color='black', linewidth=2)
            ax.text(bar_x - (x_max - x_min) * 0.01, (bar_bottom + bar_top)/2, f'{scale_value} $\Delta F/F_0$', 
                    ha='right', va='center', fontsize=8)

        ax.set_xlim(x_min, x_max)
        # Y-lim will be autoscaled or we can set it
        # ax.set_ylim(-1, n_neurons * y_offset + 2)

    label_ax = fig.add_axes([0, 0, 1, bottom_fraction])
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(0, 1)
    label_ax.axis('off')
    # Add concentration labels for each stimulus column
    for j, stimulus in enumerate(stimulus_types):
        x_center = stimulus_x_centers[j]
        
        # Extract concentration from stimulus name
        if stimulus_info_dict and stimulus in stimulus_info_dict:
            full_name = stimulus_info_dict[stimulus]
            # Extract concentration part
            if ' E' in full_name:
                conc_part = 'E' + full_name.split(' E')[1]
            else:
                parts = full_name.split()
                conc_part = parts[-1] if len(parts) > 1 else stimulus
                if len(conc_part) > 3:
                    conc_part = conc_part[:3] + '.'
        else:
            conc_part = stimulus
            
        label_ax.text(x_center, 0.7, conc_part, ha='center', va='center', 
                     fontsize=8, rotation=0)    

    # Add tree brackets for compound groups
    bracket_y = 0.4
    bracket_height = 0.1
    
    for compound_name, stimulus_codes in grouped_stimuli:
        # Only draw bracket if there are stimuli from this group in the plot
        group_stimuli_in_plot = [s for s in stimulus_codes if s in stimulus_types]
        if len(group_stimuli_in_plot) > 1:  # Only draw bracket if more than one stimulus
            start_idx = stimulus_types.index(group_stimuli_in_plot[0])
            end_idx = stimulus_types.index(group_stimuli_in_plot[-1])
            
            x_start = stimulus_x_centers[start_idx]
            x_end = stimulus_x_centers[end_idx]
            
            # Draw bracket
            label_ax.plot([x_start, x_end], [bracket_y, bracket_y], 'k-', linewidth=1)
            label_ax.plot([x_start, x_start], [bracket_y, bracket_y + bracket_height], 'k-', linewidth=1)
            label_ax.plot([x_end, x_end], [bracket_y, bracket_y + bracket_height], 'k-', linewidth=1)
            
            # Add compound name
            label_ax.text((x_start + x_end) / 2, bracket_y - 0.1, compound_name, 
                         ha='center', va='top', fontsize=8, weight='bold')
        elif len(group_stimuli_in_plot) == 1: # If only one stimulus, just add the name without bracket
            idx = stimulus_types.index(group_stimuli_in_plot[0])
            x_center = stimulus_x_centers[idx]
            label_ax.text(x_center, bracket_y - 0.1, compound_name, 
                         ha='center', va='top', fontsize=8, weight='bold')
            
    if save_folder:
        fig.savefig(f"{save_folder}/mean_signal.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{save_folder}/mean_signal.svg", dpi=300, bbox_inches='tight')
    
    # plt.show()
    plt.close(fig)

    return {
        'covariance_matrix': covariance_matrix,
        'neurons_in_cluster_order': neurons_in_cluster_order,
        'n_dim': n_dim,
        'cluster_stimulus': cluster_stimulus if not custom_order else None,
        'stimulus_order': stimulus_types,
        'grouped_stimuli': grouped_stimuli
    }
