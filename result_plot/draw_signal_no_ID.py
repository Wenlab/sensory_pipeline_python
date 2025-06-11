if __name__ == "__main__":
    h5_path = r"I:\WJH\0628_LYP\w1\pixel_intensity.h5"
    save_folder = r"I:\WJH\0628_LYP\w1\plot"
    labjack_excel_path = r"I:\WJH\0628_LYP\Labjack\output_volumes.xlsx"
#%%
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import sys
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.HDF5_load import load_h5file
from data_load.get_stimulus_info import *
from data_load.curve_fit import calculate_delta_F_over_F0
from data_load.load_worm_data import load_worm_ID
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
    light_on_data=None,
    stimulus_list=None,
    alpha=0.7,
    ylabel = "Fluorescence Intensity"
):
    """
    Plot neural signal traces for multiple neurons, with optional light stimulation periods colored by stimulus type.
    Args:
        intensity (np.ndarray): 2D array of intensity values, shape (num_neurons, num_volumes).
        dID (list): array of neuron IDs corresponding to the rows in intensity.
        n_cols (int, optional): Number of columns in the subplot grid. Default is 2.
        row_height (float, optional): Height of each subplot row in inches. Default is 2.5.
        col_width (float, optional): Width of each subplot column in inches. Default is 10.
        xtick_num (int, optional): Approximate number of x-axis ticks. Default is 20.
        file_name (str, optional): Output file name for the saved plot image. Default is "neuron_signals.png".
        params_description (str, optional): Description text to display in the last subplot. Default is "default".
        light_on_data (list of tuple, optional): List of (start, end) tuples indicating periods of light stimulation.
        stimulus_list (list, optional): List of stimulus types corresponding to light_on_data periods.
        alpha (float, optional): Transparency level for plot lines and shaded regions. Default is 0.7.
    Returns:
        None. The function saves the generated plot to the specified file.
    """

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
    stimulus_colors = {}
    legend_dict = {}
    
    if stimulus_list is not None:
        # Get unique stimulus types
        unique_stimuli = list(set(stimulus_list))
        # Use matplotlib colormap to generate distinct colors
        cmap = plt.cm.get_cmap('tab10')  # You can change to 'Set3', 'tab20', etc.
        colors = [cmap(i / len(unique_stimuli)) for i in range(len(unique_stimuli))]
        
        # Assign colors to stimulus types
        for i, stimulus in enumerate(unique_stimuli):
            stimulus_colors[stimulus] = colors[i]

    for i, neuron_id in enumerate(tqdm(neuron_ids, desc="Plotting neuron signals", leave=False)):
        if i < len(axs) - 1:  # Skip the last subplot for params_description
            cur_ax = axs[i]
            signals = intensity[i, :]
            
            # Plot main signal line
            cur_ax.plot(
                full_volumes,
                signals,
                linestyle="-",
                linewidth=2,
                color='black',
                alpha=alpha,
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
            
            # Add light stimulation periods with stimulus-specific colors
            if light_on_data is not None and stimulus_list is not None:
                for idx, light_period in enumerate(light_on_data):
                    if idx < len(stimulus_list):
                        stimulus_type = stimulus_list[idx]
                        color = stimulus_colors.get(stimulus_type, 'gray')
                        cur_ax.axvspan(
                            light_period[0],
                            light_period[1],
                            color=color,
                            alpha=0.3,
                        )
                        
                        # Add to legend dict (avoid duplicates)
                        if stimulus_type not in legend_dict:
                            legend_dict[stimulus_type] = plt.Rectangle((0,0),1,1, 
                                                                     color=color, alpha=0.3)
            
            cur_ax.set_xticks(x_ticks)
            cur_ax.set_xticklabels([f"{int(x)}" for x in x_ticks], rotation=45)
            cur_ax.set_title(f"Neuron {neuron_id}")
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
    plt.close()
    return

def draw_raw_signal(h5_file_path, save_folder, exp_name, root_name = None, bi_ID_path=None, date=None, labjack_excel_path = None,
                   n_cols=2, row_height=2.5, col_width=10, xtick_num=20,
                   alpha=0.7):
    """
    Draw raw signal from HDF5 file and save the plot.
    """
    # Load intensity and dID from the HDF5 file
    intensity, dID = load_intensity_and_dID(h5_file_path, root_name=root_name)
    neuron_ids = dID
    # Load light stimulation data and stimulus list from Excel file
    light_on_data = None
    stimulus_list = None
    if labjack_excel_path:
        experiment_df, stimulus_lists = get_stimulus_info(labjack_excel_path, if_generate_stimulus_lists=True)
        exp_stimulus_intervals, _ = extract_intervals(experiment_df, exp_name)
        ex_stimulus_list = stimulus_lists.get(exp_name, [])
        light_on_data = exp_stimulus_intervals
        stimulus_list = ex_stimulus_list

    if bi_ID_path:
        biological_ID_info = load_worm_ID(bi_ID_path)
        bi_ID = biological_ID_info[root_name]['biological']
        # Filter neuron_ids based on biological ID
        neuron_ids = [
            id_value if isinstance(id_value, str) else f'{index}'
            for index, id_value in enumerate(bi_ID)
        ]
        neuron_ids = np.array(neuron_ids)

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
        light_on_data=light_on_data,
        stimulus_list=stimulus_list,
        alpha=alpha
    )
    
    print(f"Raw signal plot saved to {file_name}")

def draw_trend_signal(h5_file_path, save_folder, exp_name, root_name=None, bi_ID_path=None, date=None, labjack_excel_path = None,
                   n_cols=2, row_height=2.5, col_width=10, xtick_num=20,
                   alpha=0.7, ylabel = "delta_F/F_0"):
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
        delta_F_over_F0, fitted_F0, quality_info = calculate_delta_F_over_F0(intensity_df, exp_stimulus_intervals) # delta_F_over_F0 is a dataframe
    
    if bi_ID_path:
        biological_ID_info = load_worm_ID(bi_ID_path)
        bi_ID = biological_ID_info[root_name]['biological']
        # Filter neuron_ids based on biological ID
        neuron_ids = [
            id_value if isinstance(id_value, str) else f'{index}'
            for index, id_value in enumerate(bi_ID)
        ]
        neuron_ids = np.array(neuron_ids)

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
        light_on_data=light_on_data,
        stimulus_list=stimulus_list,
        alpha=alpha,
        ylabel= ylabel
    )
    
    print(f"dfof signal plot saved to {file_name}")

    return
    

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

    intensity_path = r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\20250604.h5"
    import h5py
    from result_plot.draw_signal_no_ID import *
    with h5py.File(intensity_path, 'r') as f:
        key_list = list(f.keys())

    labjack_excel_path = r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\labjack\output_volumes.xlsx"
    for key in key_list:
        save_folder = fr"H:\Process_temporary\WJH\olfactory\ID\result\20250604\tiffplot\{key}"
        trend_args = {
            "h5_file_path": intensity_path,
            "save_folder": save_folder,
            "exp_name": f"{key}",
            "root_name": key,
            "bi_ID_path": r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\ID0604_odor.xlsx",
            "date": "20250529_odor",
            "labjack_excel_path": labjack_excel_path,
            "n_cols": 2,
            "row_height": 2.5,
            "col_width": 10,
            "xtick_num": 20,
            "alpha": 0.7,
            "ylabel": "deltaF/F_0"
        }
        draw_trend_signal(**trend_args)
        raw_args = {
            "h5_file_path": intensity_path,
            "save_folder": save_folder,
            "exp_name": f"{key}",
            "root_name": key,
            "bi_ID_path": r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\ID0604_odor.xlsx",
            "date": "20250529_odor",
            "labjack_excel_path": labjack_excel_path,
            "n_cols": 2,
            "row_height": 2.5,
            "col_width": 10,
            "xtick_num": 20,
            "alpha": 0.7
        }
        draw_raw_signal(**raw_args)