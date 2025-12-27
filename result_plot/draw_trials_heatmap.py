import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_load.process_worm_data import transfer_dict2dataframe
def draw_pair_heatmap(df, save_folder, stimulus_onset=5, **kwargs):
    """
    Draws heatmaps of trials for each neuron-stimulus pair using the DataFrame structure.
    
    Parameters:
    - df: pandas DataFrame, output from transfer_dict2dataframe. 
          Expected columns: 'neuron', 'stimulus', 'time_point', 'delta_F_over_F0', 'worm_key', 'segment_index', 'date'
    - save_folder: str, path to save the figures
    - stimulus_onset: int, time point where stimulus starts (to adjust x-axis). Default is 5.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    stim_name = kwargs.get('stim_name', {})
    # Get unique stimuli
    stimuli = sorted(df['stimulus'].unique())

    for stimulus in stimuli:
        stim_df = df[df['stimulus'] == stimulus]
        neurons = sorted(stim_df['neuron'].unique())
        n_neurons = len(neurons)
        
        if n_neurons == 0:
            continue

        # Calculate grid size
        n_cols = 1  # Fixed number of columns
        n_rows = math.ceil(n_neurons / n_cols)
        
        # Adjust figure size based on rows and columns
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows), constrained_layout=True)
        
        # Flatten axes for easy iteration if it's an array
        if n_neurons > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for i, neuron in enumerate(neurons):
            ax = axes_flat[i]
            neuron_df = stim_df[stim_df['neuron'] == neuron]
            
            # Pivot to get trials x time
            # We use ['worm_key', 'segment_index', 'date'] as the unique identifier for a trial
            pivot_df = neuron_df.pivot_table(
                index=['date','worm_key', 'segment_index'], 
                columns='time_point', 
                values='delta_F_over_F0'
            )
            
            # Sort trials if needed (e.g. by date, then worm) - pivot_table sorts by index automatically
            
            heatmap_data = pivot_df.values
            
            if heatmap_data.size == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                continue
            
            

            # Plot Heatmap
            # center=0 for diverging colormap
            sns.heatmap(heatmap_data, ax=ax, cmap="RdBu_r", center=0, cbar=True, 
                        xticklabels=False, yticklabels=False) 
            
            # Customize X-axis labels
            n_timepoints = heatmap_data.shape[1]
            
            # Determine tick positions and labels
            # We want to show labels relative to stimulus_onset
            # E.g. if range is 0 to 20, and onset is 5. Labels: -5, 0, 5, 10, 15...
            
            # Generate ticks every 5 units (or appropriate interval)
            tick_interval = 5
            
            xticks = []
            xticklabels = []
            
            for t in range(n_timepoints):
                rel_time = t - stimulus_onset
                if rel_time % tick_interval == 0:
                    xticks.append(t + 0.5) # Center of the pixel
                    xticklabels.append(str(rel_time))
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=0)

            # set y label
            y_labels = []
            for date_val, worm_key, segment_index in pivot_df.index:
                d = date_val.decode('utf-8') if isinstance(date_val, bytes) else str(date_val)
                k = worm_key.decode('utf-8') if isinstance(worm_key, bytes) else str(worm_key)
                s = segment_index.decode('utf-8') if isinstance(segment_index, bytes) else str(segment_index)
                y_labels.append(f"{d}_{k}_{s}")
            ax.set_yticks(np.arange(len(y_labels)) + 0.5)
            ax.set_yticklabels(y_labels, rotation=45, fontsize=4)
            
            # Add vertical line for stimulus onset
            # In heatmap coordinates (0..N), x=stimulus_onset corresponds to the start of that column.
            ax.axvline(x=stimulus_onset, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.set_title(f"{neuron} (n={len(heatmap_data)})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Trials")

        # Hide unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
            
        fig.suptitle(f"Stimulus: {stim_name.get(stimulus, stimulus)}", fontsize=16)
        
        # Save
        safe_stim_name = str(stim_name.get(stimulus, stimulus)).replace('/', '_').replace('\\', '_').replace(" ", "_")
        jpg_path = os.path.join(save_folder, f"{safe_stim_name}_heatmap.jpg")
        pdf_path = os.path.join(save_folder, f"{safe_stim_name}_heatmap.pdf")
        
        plt.savefig(jpg_path, dpi=300)
        plt.savefig(pdf_path, dpi=300)
        plt.close(fig)
        
        print(f"Saved heatmap for {stimulus} to {save_folder}")


def draw_trials_heatmap(neuron_segments_dict, save_folder, stimulus_onset=5, **kwargs):
    df = transfer_dict2dataframe(neuron_segments_dict)

    draw_pair_heatmap(df, save_folder, stimulus_onset=5, **kwargs)
