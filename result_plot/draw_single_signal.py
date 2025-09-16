import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import sys
import inspect
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
#%%
def draw_params_description(ax,params_description):
    ax.axis("off")
    ax.text(0, 1, params_description, ha="left", va="top", wrap=True)
    ax.set_title("Parameters Description")

def generate_params_description(params_dict,exclude_list=[]):
    """
    生成包含所有参数信息的文本。
    :param params_dict: 字典形式的参数，用于包含所有参数信息。
    :return: 参数信息的字符串。
    """
    return "\n".join([f"{key}: {value}" for key, value in params_dict.items()if key not in exclude_list])

def draw_raw_neuron_signal(
        intensity_dicts, 
        full_volumes, 
        colors=None,
        n_cols=3,
        row_height=2.5,
        col_width=10,
        xtick_num=10,
        params_description="default",
        stimulus_dict=None,
        stimulus_color_dict=None,
        alpha=0.5,
        vps=5,
        filename=None):
    """
    Draws the neuron signal from the provided intensity dictionaries and full volumes as x axis.
    Args:
        intensity_dicts (dict): {channel:(N,T)}.
        full_volumes (list): X-axis.
        colors (dict, optional): Dictionary mapping channel names to colors for plotting. If None, default colormap is used.
        n_cols (int, optional): Number of columns in the plot. Defaults to 3.
        row_height (float, optional): Height of each row in the plot. Defaults to 2.5.
        col_width (float, optional): Width of each column in the plot. Defaults to 10.
        xtick_num (int, optional): Number of x-ticks. Defaults to 10.
        params_description (str, optional): Description text to display in the last subplot. Default is "default".
        stimulus_dict (dict, optional): {stimuli: times x (start,period)} applied during the experiment. Defaults to None.
        stimulus_color_dict (dict, optional): Dictionary mapping stimuli to colors. Defaults to None.
        alpha (float, optional): Transparency level for the stimuli highlight. Defaults to 0.5.
        vps (int, optional): Volumes per second for x-axis scaling. Defaults to 5.
        filename (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
    """
    num_neurons = next(iter(intensity_dicts.values())).shape[0]
    num_subplots = num_neurons + 1  # +1 for params_description
    n_rows = (num_subplots + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, row_height * n_rows))
    # axs = np.ravel(axs)
    axs = axs.flatten()
    default_colors = plt.colormaps["tab20"]
    lengend_dict = {}


    if stimulus_dict is not None:
        unique_stimuli = list(set(stimulus_dict.keys()))
        if stimulus_color_dict is None:
            cmap = plt.cm.get_cmap("pastel1", len(unique_stimuli))
            stimulus_color_dict = {stimulus: cmap(i) for i, stimulus in enumerate(unique_stimuli)}
                
    for channel_idx, (channel,data) in enumerate(intensity_dicts.items()):
        # ensure data is a 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)  # 如果是一维，转换为 (1, T)
        elif data.ndim == 0:
            raise ValueError(f"Channel {channel} data is a scalar, expected array")
        
        channel_color = colors.get(channel, default_colors(channel_idx *2)) if colors else default_colors(channel_idx *2)
        min_volume, max_volume = min(full_volumes), max(full_volumes)

        # Generate x-ticks
        locator = MaxNLocator(nbins=xtick_num, integer=True, prune=None)  # 不剪切任何刻度
        x_ticks = locator.tick_values(min_volume//5, max_volume//5)
        x_ticks = x_ticks[(x_ticks >= min_volume//5) & (x_ticks <= max_volume//5)]
        
        # 确保包含0
        if min_volume//5 <= 0 <= max_volume//5 and 0 not in x_ticks:
            x_ticks = np.append(x_ticks, 0)
            x_ticks = np.sort(x_ticks)

        # find the global min and max for y-axis scaling
        y_min = np.percentile(data, 0)
        y_max = np.percentile(data, 99.5)

        for neuron_idx, neuron_signal in enumerate(data):
            # Check if neuron_signal is a scalar or array
            if np.isscalar(neuron_signal):
                raise ValueError(f"Channel {channel}, Neuron {neuron_idx}: Expected array but got scalar")
            
            if channel_idx ==0:
                cur_ax = axs[neuron_idx]
                cur_ax.spines['left'].set_color(channel_color)
            else:
                cur_ax = axs[neuron_idx].twinx()
                cur_ax.spines['right'].set_color(channel_color)
                cur_ax.spines['left'].set_alpha(0)
                cur_ax.spines['right'].set_position(('axes', 1.0+0.08*(channel_idx-1))) # move right spine 8% of axis width to the right

            isolate_mask = np.full_like(neuron_signal, False, dtype=bool)
            for i in range(len(neuron_signal)):
                if not np.isnan(neuron_signal[i]):
                    left_nan = (i == 0) or np.isnan(neuron_signal[i-1])
                    right_nan = (i == len(neuron_signal)-1) or np.isnan(neuron_signal[i+1])
                    if left_nan and right_nan:
                        isolate_mask[i] = True
            cur_ax.plot(
                full_volumes / vps, 
                neuron_signal,
                linestyle='-',
                marker='', 
                color=channel_color, 
                label=str(channel), 
                alpha=alpha
            )
            cur_ax.plot(
                full_volumes[isolate_mask] / vps, 
                neuron_signal[isolate_mask], 
                marker='.',
                linestyle='',
                color=channel_color,
                label= str(channel),
                alpha=alpha
            )
            if stimulus_dict is not None:
                for stimulus, epoch in stimulus_dict.items():
                    for i, (start, period) in enumerate(epoch):
                        if stimulus in stimulus_color_dict:
                            color = stimulus_color_dict[stimulus]
                            cur_ax.axvspan(
                                start / vps, 
                                (start + period) / vps, 
                                color=color, 
                                alpha=0.2,
                                label=str(stimulus)
                            )
            cur_ax.set_xticks(x_ticks)
            cur_ax.set_xticklabels([str(int(x)) for x in x_ticks], rotation=45)
            cur_ax.set_title(f"Neuron {neuron_idx}")
            cur_ax.set_ylabel(f"Signal_{channel}", color=channel_color)
            cur_ax.set_ylim(y_min, y_max)
            cur_ax.tick_params(axis='y', labelcolor=channel_color, color=channel_color)
            lines, labels = cur_ax.get_legend_handles_labels()
            for line, label in zip(lines, labels):
                if label not in lengend_dict:
                    lengend_dict[label] = line
            
    # Draw the last subplot for parameters description
    draw_params_description(axs[-1], params_description)
    axs[-1].legend(lengend_dict.values(), lengend_dict.keys(), loc='lower right')

    plt.tight_layout()
    if filename is not None:
        # plt.savefig(file_name + ".png", dpi=300)
        plt.savefig(filename, dpi=300)
    plt.close()


def draw_neuron_signal(
                    intensity_dicts,
                    full_volumes,
                    root_save_path,
                    kwargs):
    """
    params:
        intensity_dicts (dict): {channel:(N,T)}.
        full_volumes (list): X-axis.
        root_save_path (str): Path to save the plot.
        kwargs (dict): Additional parameters for drawing, including:
            - colors (dict): Dictionary mapping channel names to colors for plotting.
            - n_cols (int): Number of columns in the plot.
            - row_height (float): Height of each row in the plot.
            - col_width (float): Width of each column in the plot.
            - xtick_num (int): Number of x-ticks.
            - params_description (str): Description text to display in the last subplot.
            - stimulus_dict (dict): {stimuli: times x (start,period)} applied during the experiment.
            - stimulus_color_dict (dict): Dictionary mapping stimuli to colors.
            - alpha (float): Transparency level for the stimuli highlight.
            - vps (int): Volumes per second for x-axis scaling.
    """
    params_dict = dict(kwargs)
    params_dict['root_save_path'] = root_save_path
    params_description = generate_params_description(params_dict)
    
    # file name suffix
    exp_name = kwargs.get("exp_name", None)
    date = kwargs.get("date", None)
    channel_names = "_".join(intensity_dicts.keys())
    file_name_surfix = f"{exp_name}_{date}_{channel_names}"
    filename = os.path.join(root_save_path, f"neuron_signals_{file_name_surfix}.pdf")
    
    draw_raw_neuron_signal(
        intensity_dicts=intensity_dicts,
        full_volumes=full_volumes,
        colors=kwargs.get("colors", None),
        n_cols=kwargs.get("n_cols", 3),
        row_height=kwargs.get("row_height", 2.5),
        col_width=kwargs.get("col_width", 10),
        xtick_num=kwargs.get("xtick_num", 10),
        params_description=params_description,
        stimulus_dict=kwargs.get("stimulus_dict", None),
        stimulus_color_dict=kwargs.get("stimulus_color_dict", None),
        alpha=kwargs.get("alpha", 0.5),
        vps=kwargs.get("vps", 5),
        filename=filename
    )



if __name__ == "__main__":
    root_save_path = rf"H:\Process_temporary\WJH\immobile_data\20250712_test_panneuron_4chemicals\w1\try"  # 结果保存路径
    kwargs = dict(
        exp_name = "simulate_data_2025-07-12",
        colors = {'Channel1': 'blue', 'Channel2': 'green'},
        stimulus_dict = {
            'Stimulus1': np.array([(0, 10), (20, 10)]),
            'Stimulus2': np.array([(60, 8)])
        },
        stimulus_color_dict = {'Stimulus1': 'red', 'Stimulus2': 'orange'},
        n_cols=2,
        row_height=2.5,
        col_width=10,
        xtick_num=10,
        date="2025-07-12",
        vps=5,
    )
    # Example usage
    intensity_dicts = {
        'Channel1': np.random.rand(5, 100),
        'Channel2': np.random.rand(5, 100)
    }
    full_volumes = np.arange(100)
    
    draw_neuron_signal(
        intensity_dicts=intensity_dicts,
        full_volumes=full_volumes,
        root_save_path=root_save_path,
        kwargs=kwargs
    )

