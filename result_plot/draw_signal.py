#%%
import os
import h5py
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import sys
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import zscore
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.HDF5Toolkit import load_h5file
from utils.get_symmetric_neuron import get_symmetric_neuron
from utils.parse_stimulus_info import group_and_sort_stimuli
from data_load.get_stimulus_info import *
from data_load.curve_fit import calculate_delta_F_over_F0
from data_load.load_worm_data import load_worm_ID
from result_plot.visweb import generate_compound_color_scheme
import seaborn as sns
#%%
def load_intensity_and_dID(h5_file_path, root_name=None):
    """
    Load intensity and dital_ID from the HDF5 file.
    """
    if root_name is None:
        with h5py.File(h5_file_path, 'r') as f:
            root_name = list(f.keys())[0]
    data = load_h5file(h5_file_path, root_name)
    intensity = data['intensity']
    dID = data['ID']
    return intensity, dID

def plot_neuron_signals(
    intensity,
    neuron_ids,
    n_cols=2,
    row_height=2.5,
    col_width=10,
    xtick_num=20,
    file_name="neuron_signals.png",
    params_description="default",
    curve_fit=None,
    odor_config=None,
    light_on_data=None,
    stimulus_list=None,
    stimulus_color_dict=None,
    neurons_list=None,
    alpha=0.7,
    ylabel = "Fluorescence Intensity"
):
    """
    Plot neural signal traces for multiple neurons, with optional light stimulation periods colored by stimulus type.
    Args:
        intensity (np.ndarray): 2D array of intensity values, shape (num_neurons, num_volumes).
        neuron_ids (list): array of neuron IDs corresponding to the rows in intensity.
        n_cols (int, optional): Number of columns in the subplot grid. Default is 2.
        row_height (float, optional): Height of each subplot row in inches. Default is 2.5.
        col_width (float, optional): Width of each subplot column in inches. Default is 10.
        xtick_num (int, optional): Approximate number of x-axis ticks. Default is 20.
        file_name (str, optional): Output file name for the saved plot image. Default is "neuron_signals.png".
        params_description (str, optional): Description text to display in the last subplot. Default is "default".
        curve_fit (np.ndarray, optional): Fitted F0 values for each neuron, shape (num_neurons, num_volumes). Default is None.
        odor_config (dict, optional): Configuration for odor stimulation, map stimulus symbol to stimulus name.
        light_on_data (list of tuple, optional): List of (start, end) tuples indicating periods of light stimulation.
        stimulus_list (list, optional): List of stimulus types corresponding to light_on_data periods.
        stimulus_color_dict (dict, optional): Dictionary mapping stimulus types to colors. If None, uses generate_compound_color_scheme.
        neurons_list (list, optional): List of neuron IDs to include in the plot. If None, all neurons are included.
        alpha (float, optional): Transparency level for plot lines and shaded regions. Default is 0.7.
        ylabel (str, optional): Label for the y-axis. Default is "Fluorescence Intensity".
    Returns:
        None. The function saves the generated plot to the specified file.
    """

    # Filter neurons if neurons_list is provided
    if neurons_list is not None:
        # Find indices of neurons to include
        filtered_indices = []
        filtered_neuron_ids = []
        
        for target_id in neurons_list:
            # Find matching neuron indices
            for idx, neuron_id in enumerate(neuron_ids):
                # Convert both to string for comparison to handle mixed types
                if str(neuron_id) == str(target_id):
                    filtered_indices.append(idx)
                    filtered_neuron_ids.append(neuron_id)
                    break
        
        if len(filtered_indices) == 0:
            print(f"Warning: No neurons found matching the provided neurons_list: {neurons_list}")
            return
            
        # Filter intensity data and neuron IDs
        intensity = intensity[filtered_indices, :]
        neuron_ids = filtered_neuron_ids
        
        # Filter curve_fit data if provided
        if curve_fit is not None:
            curve_fit = curve_fit[filtered_indices, :]

    num_neurons = len(neuron_ids)
    num_subplots = num_neurons + 1  # +1 for params_description
    n_rows = (num_subplots + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * col_width, n_rows * row_height))
    axs = np.ravel(axs)  # Ensure axs is a flat array for easy iteration.
    
    # Create volume indices
    full_volumes = np.arange(intensity.shape[1])
    
    min_volume = 0 
    max_volume = intensity.shape[1]

    # Generate "nice" ticks for x-axis, close to xtick_num and human-friendly.
    locator = MaxNLocator(nbins=xtick_num, integer=True, prune=None)
    x_ticks = locator.tick_values(min_volume, max_volume)
    x_ticks = x_ticks[(x_ticks >= min_volume) & (x_ticks <= max_volume)]

    # Find the global min and max signal values for y-axis limits
    intensity_flat = intensity.flatten()
    intensity_wo_nan = intensity_flat[~np.isnan(intensity_flat)]
    y_min = np.percentile(intensity_wo_nan, 1)   # 1 percentile for lower bound
    y_max = np.percentile(intensity_wo_nan, 100)  # 99 percentile for upper bound

    # Automatically assign colors to different stimulus types
    if stimulus_color_dict is None:
        stimulus_color_dict = {}
    legend_dict = {}
    
    if stimulus_list is not None:
        if not stimulus_color_dict:
            stimulus_color_dict = generate_compound_color_scheme(stimulus_list)


    for i, neuron_id in enumerate(tqdm(neuron_ids, desc="Plotting neuron signals", leave=False)):
        if i < len(axs) - 1:  # Skip the last subplot for params_description
            cur_ax = axs[i]
            signals = intensity[i, :]
            if curve_fit is not None:
                # Use fitted_F0 for curve fitting if provided
                signals_F0 = curve_fit[i, :]
                # Plot curve_fit (fitted_F0) as a yellow line
                cur_ax.plot(
                    full_volumes,
                    signals_F0,
                    linestyle="-",
                    linewidth=2,
                    color='orange',
                    alpha=0.7,
                    label='curve_fit'
                )
            
            # Plot main signal line
            cur_ax.plot(
                full_volumes,
                signals,
                linestyle="-",
                linewidth=2,
                color='black',
                alpha=alpha,
                label='raw_trace'
            )
            
            # Highlight isolated points (surrounded by NaN)
            isolate_mask = np.full_like(signals, False, dtype=bool)
            for j in range(len(signals)):
                if not np.isnan(signals[j]):
                    left_nan = (j == 0) or np.isnan(signals[j-1])
                    right_nan = (j == len(signals)-1) or np.isnan(signals[j+1])
                    if left_nan and right_nan:
                        isolate_mask[j] = True
            
            cur_ax.plot(
                full_volumes[isolate_mask],
                signals[isolate_mask],
                marker=".",
                markersize=4,
                linestyle="",
                color='red',
                alpha=alpha,
            )
            cur_ax.legend(loc='upper right', fontsize=8)
            # Add light stimulation periods with stimulus-specific colors
            if light_on_data is not None and stimulus_list is not None:
                for idx, light_period in enumerate(light_on_data):
                    if idx < len(stimulus_list):
                        stimulus_type = stimulus_list[idx]
                        color = stimulus_color_dict.get(stimulus_type, 'gray')
                        cur_ax.axvspan(
                            light_period[0],
                            light_period[1],
                            color=color,
                            alpha=0.3,
                        )
                        
                        # Add to legend dict (avoid duplicates)
                        if stimulus_type not in legend_dict:
                            stimulus_name = odor_config.get(stimulus_type, stimulus_type) if odor_config else stimulus_type
                            legend_dict[stimulus_name] = plt.Rectangle((0,0),1,1, 
                                                                     color=color, alpha=0.3)
            
            cur_ax.set_xticks(x_ticks)
            cur_ax.set_xticklabels([f"{int(x)}" for x in x_ticks], rotation=45)
            # Check if neuron_id is a numeric string that equals i
            if (isinstance(neuron_id, str) and neuron_id.isdigit() and int(neuron_id) == i) or (isinstance(neuron_id, (int, float, np.integer, np.floating)) and int(neuron_id) == i):
                cur_ax.set_title(f"Neuron {i}")
            else:
                cur_ax.set_title(f"{i} --Neuron {neuron_id}")
            cur_ax.set_xlabel("Volumes")
            cur_ax.set_ylabel(ylabel)
            cur_ax.set_ylim(y_min, y_max)

    # Add parameters description and legend to the last subplot
    if len(axs) > num_neurons:
        axs[-1].text(0.1, 0.7, params_description, transform=axs[-1].transAxes, 
                    fontsize=12, verticalalignment='center')
        axs[-1].set_title("Parameters")
        axs[-1].axis('off')
        
        # Add legend for stimulus types
        if legend_dict:
            axs[-1].legend(legend_dict.values(), legend_dict.keys(), 
                          loc='center', bbox_to_anchor=(0.5, 0.3),
                          title="Stimulus Types")

    # Hide unused subplots
    for i in range(num_subplots, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.savefig(file_name.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    return

def draw_raw_signal(h5_file_path, save_folder, exp_name, root_name = None, 
                    odor_config_file = None, bi_ID_path=None, labjack_excel_path = None, stimulus_color_path=None,
                    date=None, if_curve_fit = True, n_cols=2, row_height=2.5, col_width=10, xtick_num=20,
                    alpha=0.7, neurons_list=None, baseline_pre=5):
    """
    Draw raw signal from HDF5 file and save the plot.
    
    Args:
        h5_file_path (str): Path to the HDF5 file containing intensity data.
        save_folder (str): Directory to save the output plot.
        exp_name (str): Name of the experiment.
        root_name (str, optional): Root name in HDF5 file structure.
        odor_config_file (str, optional): Path to odor configuration JSON file.
        bi_ID_path (str, optional): Path to biological ID Excel file.
        labjack_excel_path (str, optional): Path to labjack Excel file with stimulus information.
        stimulus_color_path (str, optional): Path to JSON file containing stimulus color mapping.
        date (str, optional): Experiment date.
        if_curve_fit (bool, optional): Whether to include curve fitting. Default is True.
        n_cols (int, optional): Number of columns in subplot grid. Default is 2.
        row_height (float, optional): Height of each subplot row. Default is 2.5.
        col_width (float, optional): Width of each subplot column. Default is 10.
        xtick_num (int, optional): Number of x-axis ticks. Default is 20.
        alpha (float, optional): Transparency level. Default is 0.7.
        neurons_list (list, optional): List of neuron IDs to include in the plot. If None, all neurons are included.
        baseline_pre (int, optional): volume number before stimulus to use for baseline calculation. Default is 5.
    """
    # Load intensity and dID from the HDF5 file
    intensity, dID = load_intensity_and_dID(h5_file_path, root_name=root_name)
    neuron_ids = dID
    intensity_df = pd.DataFrame(intensity)
    # Load light stimulation data and stimulus list from Excel file
    light_on_data = None
    stimulus_list = None
    if labjack_excel_path:
        experiment_df, stimulus_lists = get_stimulus_info(labjack_excel_path, if_generate_stimulus_lists=True)
        exp_stimulus_intervals, _ = extract_intervals(experiment_df, exp_name)
        ex_stimulus_list = stimulus_lists.get(exp_name, [])
        light_on_data = exp_stimulus_intervals
        stimulus_list = ex_stimulus_list
        delta_F_over_F0, fitted_F0, quality_info = calculate_delta_F_over_F0(intensity_df, exp_stimulus_intervals, baseline_pre=baseline_pre)

        fitted_F0 = fitted_F0.values

    if odor_config_file:
        with open(odor_config_file, 'r') as f:
            odor_config = json.load(f)
    else:
        odor_config = None

    if bi_ID_path:
        biological_ID_info = load_worm_ID(bi_ID_path)
        bi_ID = biological_ID_info[root_name]['biological']
        # Filter neuron_ids based on biological ID
        neuron_ids = [
            id_value if isinstance(id_value, str) else f'{index}'
            for index, id_value in enumerate(bi_ID)
        ]
        neuron_ids = np.array(neuron_ids)
    if stimulus_color_path:
        with open(stimulus_color_path, 'r') as f:
            stimulus_color_dict = json.load(f)
    else:
        stimulus_color_dict = None

    # Prepare parameters description
    params_description = exp_name + f"\nDate: {date}" if date else exp_name

    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True) 
    # Plot neuron signals
    file_name = os.path.join(save_folder, f"{exp_name}_raw_neuron_signals.png")
    plot_neuron_signals(
        intensity,
        neuron_ids,
        n_cols=n_cols,
        row_height=row_height,
        col_width=col_width,
        xtick_num=xtick_num,
        file_name=file_name,
        params_description=params_description,
        curve_fit=fitted_F0 if if_curve_fit else None,
        odor_config=odor_config,
        light_on_data=light_on_data,
        stimulus_list=stimulus_list,
        stimulus_color_dict=stimulus_color_dict,
        neurons_list=neurons_list,
        alpha=alpha
    )
    
    print(f"Raw signal plot saved to {file_name}")


def draw_trend_signal(h5_file_path, save_folder, exp_name, root_name=None,
                      odor_config_file=None, bi_ID_path=None, labjack_excel_path=None, stimulus_color_path=None,
                   date=None, n_cols=2, row_height=2.5, col_width=10, xtick_num=20,
                   alpha=0.7, ylabel = "delta_F/F_0", neurons_list=None, baseline_pre=5):
    """
    Draw trend signal (delta F/F0) from HDF5 file and save the plot.
    
    Args:
        h5_file_path (str): Path to the HDF5 file containing intensity data.
        save_folder (str): Directory to save the output plot.
        exp_name (str): Name of the experiment.
        root_name (str, optional): Root name in HDF5 file structure.
        odor_config_file (str, optional): Path to odor configuration JSON file.
        bi_ID_path (str, optional): Path to biological ID Excel file.
        labjack_excel_path (str, optional): Path to labjack Excel file with stimulus information.
        stimulus_color_path (str, optional): Path to JSON file containing stimulus color mapping.
        date (str, optional): Experiment date.
        n_cols (int, optional): Number of columns in subplot grid. Default is 2.
        row_height (float, optional): Height of each subplot row. Default is 2.5.
        col_width (float, optional): Width of each subplot column. Default is 10.
        xtick_num (int, optional): Number of x-axis ticks. Default is 20.
        alpha (float, optional): Transparency level. Default is 0.7.
        ylabel (str, optional): Label for y-axis. Default is "delta_F/F_0".
        neurons_list (list, optional): List of neuron IDs to include in the plot. If None, all neurons are included.
        baseline_pre (int, optional): Volume number before stimulus to use for baseline calculation. Default is 5.
    """
    intensity, dID = load_intensity_and_dID(h5_file_path, root_name=root_name)
    neuron_ids = dID
    intensity_df = pd.DataFrame(intensity)
    # Load light stimulation data and stimulus list from Excel file
    light_on_data = None
    stimulus_list = None
    if labjack_excel_path:
        experiment_df, stimulus_lists = get_stimulus_info(labjack_excel_path, if_generate_stimulus_lists=True)
        exp_stimulus_intervals, _ = extract_intervals(experiment_df, exp_name)
        ex_stimulus_list = stimulus_lists.get(exp_name, [])
        light_on_data = exp_stimulus_intervals
        stimulus_list = ex_stimulus_list
        delta_F_over_F0, fitted_F0, quality_info = calculate_delta_F_over_F0(intensity_df, exp_stimulus_intervals, baseline_pre=baseline_pre) # delta_F_over_F0 is a dataframe

    if odor_config_file:
        with open(odor_config_file, 'r') as f:
            odor_config = json.load(f)
    else:
        odor_config = None

    if bi_ID_path:
        biological_ID_info = load_worm_ID(bi_ID_path)
        bi_ID = biological_ID_info[root_name]['biological']
        # Filter neuron_ids based on biological ID
        neuron_ids = [
            id_value if isinstance(id_value, str) else f'{index}'
            for index, id_value in enumerate(bi_ID)
        ]
        neuron_ids = np.array(neuron_ids)
    if stimulus_color_path:
        with open(stimulus_color_path, 'r') as f:
            stimulus_color_dict = json.load(f)
    else:
        stimulus_color_dict = None

    delta_F_over_F0 = delta_F_over_F0.values
    # Get rid of abnormal values  absolute value larger than 6
    delta_F_over_F0[np.abs(delta_F_over_F0) > 6] = np.nan

    # Prepare parameters description
    params_description = exp_name + f"\nDate: {date}" if date else exp_name

    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True) 
    # Plot neuron signals
    file_name = os.path.join(save_folder, f"{exp_name}_dfof_neuron_signals.png")
    plot_neuron_signals(
        delta_F_over_F0,
        neuron_ids,
        n_cols=n_cols,
        row_height=row_height,
        col_width=col_width,
        xtick_num=xtick_num,
        file_name=file_name,
        params_description=params_description,
        odor_config=odor_config,
        light_on_data=light_on_data,
        stimulus_list=stimulus_list,
        stimulus_color_dict=stimulus_color_dict,
        neurons_list=neurons_list,
        alpha=alpha,
        ylabel= ylabel
    )
    
    print(f"dfof signal plot saved to {file_name}")

    return

#%%
def transfer_dict2dataframe(neuron_segments_dict):
    """
    convert a nested dictionary into a pandas DataFrame for easier group and statistics calculate.
    """ 
    data_list = []
    for neuron_name, stimuli_data in neuron_segments_dict.items():
        for stimulus_type, segments in stimuli_data.items():
                for segment in segments:
                    delta_F_over_F0 = np.array(segment['deltaFoverF_0'])
                    time_points = len(delta_F_over_F0)

                    for t, dff in zip(range(time_points), delta_F_over_F0):
                        data_list.append({
                            'neuron': neuron_name,
                            'stimulus': stimulus_type,
                            'time_point': t,
                            'delta_F_over_F0': dff,
                            'worm_key': segment.get('worm_key', 'unknown'),
                            'segment_index': segment.get('segment_index', 'unknown'),
                            'date': segment.get('date', 'unknown')
                        })

    df = pd.DataFrame(data_list)
    return df


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

    
def draw_mean_signal_cluster(neuron_segments_df,
                             vertical_overlap=0.5,
                             fig_width=12,
                             fig_height_per_neuron = 1.5,
                             stimulus_info_dict=None,
                             cluster_stimulus=None,
                             save_folder=None,
                             plot_covariance=True
                             ):
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


    # cluster with the stimulus type with the stimulus type with most neurons responding
    stimulus_neuron_counts = neuron_segments_df.groupby('stimulus')['neuron'].nunique()

    # cluster_stimulus = stimulus_neuron_counts.idxmax()
    cluster_stimulus = cluster_stimulus if cluster_stimulus else stimulus_neuron_counts.idxmax()
    print(f"cluster based on {cluster_stimulus}")
    df_cluster_stimulus = neuron_segments_df[neuron_segments_df['stimulus'] == cluster_stimulus]
    if df_cluster_stimulus.empty:
        print(f"No data available for stimulus type '{cluster_stimulus}' to perform clustering.")
        neurons_in_cluster_order = all_neurons
        missing_neurons_from_clustering = set(all_neurons)
        return
    else:
        cluster_matrix_partial = df_cluster_stimulus.pivot_table(
            index='neuron',
            columns='time_point',
            values='delta_F_over_F0',
            aggfunc='mean'
        )

        time_columns = cluster_matrix_partial.columns
        cluster_matrix_full = pd.DataFrame(index=all_neurons, columns=time_columns)

        present_neurons = set(cluster_matrix_partial.index)
        missing_neurons_from_clustering = set(all_neurons) - present_neurons

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
        cluster_matrix_zscored= zscore(cluster_matrix_full, axis=1)
        cluster_matrix_zscored = np.nan_to_num(cluster_matrix_zscored)  # Replace NaNs with 0 after z-scoring
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
    # calculate and plot co1variance matrix
    covariance_matrix = None
    n_dim = len(neurons_in_cluster_order)

    if plot_covariance and save_folder:
        ordered_matrix = cluster_matrix_full.loc[neurons_in_cluster_order]
        covariance_matrix = np.cov(ordered_matrix.values)

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

    # Calculate y_axis limits
    grouped = neuron_segments_df.groupby(['neuron', 'stimulus'])
    agg = grouped['delta_F_over_F0'].agg(['mean', 'sem']).reset_index()
    global_y_min = (agg['mean'] - agg['sem']).min()
    global_y_max = (agg['mean'] + agg['sem']).max()
    padding = (global_y_max - global_y_min) * 0.1
    global_y_min -= padding
    global_y_max += padding
    x_min = neuron_segments_df['time_point'].min()
    x_max = neuron_segments_df['time_point'].max()


    # PLOTTING
    n_neurons = len(all_neurons)
    n_stimuli = len(stimulus_types)

    extra_bottom_space = 0.5
    total_height = fig_height_per_neuron * (1 + (n_neurons - 1) * (1 - vertical_overlap))
    fig = plt.figure(figsize=(fig_width, total_height))
    fig.suptitle('')

    subplot_width = 1 / n_stimuli
    subplot_height = fig_height_per_neuron / total_height
    bottom_offset = extra_bottom_space / total_height

    for i, neuron in enumerate(neurons_in_cluster_order):
        for j, stimulus in enumerate(stimulus_types):
            left = j * subplot_width
            bottom = bottom_offset + (n_neurons - i - 1) * (subplot_height * (1 - vertical_overlap))
            ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
            ax.set_facecolor('none')

            df_subset = neuron_segments_df[(neuron_segments_df['neuron']==neuron) & (neuron_segments_df['stimulus']==stimulus)]
            if not df_subset.empty:
                mean_trace = df_subset.groupby('time_point')['delta_F_over_F0'].mean()
                if not mean_trace.empty:
                    sem_trace = df_subset.groupby('time_point')['delta_F_over_F0'].sem()
                    ax.plot(mean_trace.index, mean_trace.values, color='blue', zorder=10)
                    ax.fill_between(mean_trace.index,
                                    mean_trace.values - sem_trace.values,
                                    mean_trace.values + sem_trace.values,
                                    color='gray', alpha=0.3, zorder=9)
            ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5,zorder=1)
            # remove spines and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(global_y_min, global_y_max)
            ax.set_xlim(x_min, x_max)

            ax.axvline(x=5, color='#E6A61C', linestyle='--', label='Stimulus Onset', linewidth=1.5, zorder=1)
            ax.axvline(x=15, color='#76642E', linestyle='--', label='Stimulus Offset', linewidth=1.5, zorder=1)


            if j == 0:
                ax.text(-0.1, 0, neuron, transform=ax.transData, 
                        rotation = 45, ha = 'right', va = 'center', fontsize=10)
            else:
                ax.spines['left'].set_visible(False)

            # if i == n_neurons - 1:
            #     # ax.spines['bottom'].set_visible(True)
            #     # ax.set_xticks([5, 15])
            #     # ax.set_xticklabels(['0', '10'])
            #     ax.set_xlabel(stimulus, fontsize=10)
    label_ax = fig.add_axes([0, 0, 1, bottom_offset])
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(0, 1)
    label_ax.axis('off')
    # Add concentration labels for each stimulus column
    for j, stimulus in enumerate(stimulus_types):
        x_center = (j + 0.5) / n_stimuli
        
        # Extract concentration from stimulus name
        if stimulus_info_dict and stimulus in stimulus_info_dict:
            full_name = stimulus_info_dict[stimulus]
            # Extract concentration part
            if ' E' in full_name:
                conc_part = 'E' + full_name.split(' E')[1]
            else:
                parts = full_name.split()
                conc_part = parts[-1] if len(parts) > 1 else stimulus
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
            
            x_start = start_idx / n_stimuli
            x_end = (end_idx + 1) / n_stimuli
            
            # Draw bracket
            label_ax.plot([x_start, x_end], [bracket_y, bracket_y], 'k-', linewidth=1)
            label_ax.plot([x_start, x_start], [bracket_y, bracket_y + bracket_height], 'k-', linewidth=1)
            label_ax.plot([x_end, x_end], [bracket_y, bracket_y + bracket_height], 'k-', linewidth=1)
            
            # Add compound name
            label_ax.text((x_start + x_end) / 2, bracket_y - 0.1, compound_name, 
                         ha='center', va='top', fontsize=8, weight='bold')
        elif len(group_stimuli_in_plot) == 1: # If only one stimulus, just add the name without bracket
            idx = stimulus_types.index(group_stimuli_in_plot[0])
            x_center = (idx + 0.5) / n_stimuli
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
        'cluster_stimulus': cluster_stimulus,
        'stimulus_order': stimulus_types,
        'grouped_stimuli': grouped_stimuli
    }
if __name__ == "__main__":
    # args = {
    #     "h5_file_path": h5_path,
    #     "save_folder": save_folder,
    #     "exp_name": "w1",
    #     "date": "2024-06-28_LYP",
    #     "labjack_excel_path": labjack_excel_path,
    #     "n_cols": 2,
    #     "row_height": 2.5,
    #     "col_width": 10,
    #     "xtick_num": 20,
    #     "alpha": 0.7,
    #     "ylabel": "deltaF/F_0"
    # }
    # # draw_raw_signal(**args)
    # draw_trend_signal(**args)

    # for i in range(1, 19):
    #     h5_path = fr"I:\WJH\0628_LYP\w{i}\pixel_intensity.h5"
    #     save_folder = fr"I:\WJH\0628_LYP\w{i}\plot_sort"
    #     labjack_excel_path = r"I:\WJH\0628_LYP\Labjack\output_volumes.xlsx"
    #     trend_args = {
    #         "h5_file_path": h5_path,
    #         "save_folder": save_folder,
    #         "exp_name": f"w{i}",
    #         "date": "2024-06-28_LYP",
    #         "labjack_excel_path": labjack_excel_path,
    #         "stimulus_color_path": r"I:\WJH\0628_LYP\stimulus_color.json",
    #         "n_cols": 2,
    #         "row_height": 2.5,
    #         "col_width": 10,
    #         "xtick_num": 20,
    #         "alpha": 0.7,
    #         "ylabel": "deltaF/F_0"
    #     }
    #     draw_trend_signal(**trend_args)
    #     raw_args = {
    #         "h5_file_path": h5_path,
    #         "save_folder": save_folder,
    #         "exp_name": f"w{i}",
    #         "date": "2024-06-28_LYP",
    #         "labjack_excel_path": labjack_excel_path,
    #         "stimulus_color_path": r"I:\WJH\0628_LYP\stimulus_color.json",
    #         "n_cols": 2,
    #         "row_height": 2.5,
    #         "col_width": 10,
    #         "xtick_num": 20,
    #         "alpha": 0.7
    #     }
    #     draw_raw_signal(**raw_args)

    from utils.HDF5Toolkit import load_h5file
    import json
    with open(r"H:\Process_temporary\WJH\sensory_pipeline_python\data_load\config\compound_info.json")as f:
        stimulus_info_dict = json.load(f)
    neuron_segments_dict = load_h5file(r"I:\WJH\flavor\neuron_segments_dict_filter_corrected.h5", 'neuron_segments_dict')
    neuron_segments_df = transfer_dict2dataframe(neuron_segments_dict)
    save_folder = r"I:\WJH\flavor\plot"
    os.makedirs(save_folder, exist_ok=True)
    # relplot_mean_signal(neuron_segments_df, 
    #                     height=2, 
    #                     aspect=2, 
    #                     col_wrap=2,
    #                     kind='line',
    #                     errorbar='se',
    #                     stimulus_color_map=None,
    #                     save_path=save_folder+f"/mean_signal.png")
    plot_dict = draw_mean_signal_cluster(neuron_segments_df,
                             vertical_overlap=0.4,
                             fig_width=20,
                             fig_height_per_neuron = 1.5,
                                stimulus_info_dict=stimulus_info_dict,
                             save_folder=save_folder
                             )
