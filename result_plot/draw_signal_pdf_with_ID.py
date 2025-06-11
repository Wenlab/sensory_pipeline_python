import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors  # Add this import
#%%
def plot_neurons_to_pdf(neuron_segments_dict, odor_information, output_file, 
                       neurons_per_page=4, stimuli_to_include=None, if_combine=False):
    """
    Create a multi-page PDF with neuron responses to different stimuli.
    
    Parameters:
    -----------
    neuron_segments_dict : dict
        Dictionary with structure {neuron_id: {stimulus_type: [trial_data, ...]}}
    odor_information : dict
        Dictionary mapping stimulus codes to descriptions
    output_file : str
        Path to save the PDF file
    neurons_per_page : int
        Number of neurons to display on each page
    stimuli_to_include : list or None
        List of stimulus types to include (None = all stimuli)
    if_combine : bool
        If True, combine left and right neuron pairs by averaging responses
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    from scipy import stats
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # If combining L/R neurons, transform the dictionary
    if if_combine:
        neuron_segments_dict = combine_neuron_pairs(neuron_segments_dict)
    
    # Get all neurons and stimuli
    all_neurons = sorted(neuron_segments_dict.keys())
    
    # Get all unique stimuli across all neurons
    all_stimuli = set()
    for neuron in neuron_segments_dict:
        all_stimuli.update(neuron_segments_dict[neuron].keys())
    all_stimuli = sorted(all_stimuli)
    
    # Filter stimuli if specified
    if stimuli_to_include:
        all_stimuli = [s for s in all_stimuli if s in stimuli_to_include]
    
    # Create stimulus color map
    stimulus_colors = create_stimulus_color_map(all_stimuli)
    
    # Calculate total pages needed
    total_pages = (len(all_neurons) + neurons_per_page - 1) // neurons_per_page
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Add a title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, "Neuronal Response Visualization", fontsize=24, ha='center')
        plt.text(0.5, 0.45, f"Total neurons: {len(all_neurons)}", fontsize=14, ha='center')
        plt.text(0.5, 0.4, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}", fontsize=14, ha='center')
        pdf.savefig()
        plt.close()
        
        # Create a page for each group of neurons
        for page in range(total_pages):
            # Get subset of neurons for this page
            start_idx = page * neurons_per_page
            end_idx = min(start_idx + neurons_per_page, len(all_neurons))
            page_neurons = all_neurons[start_idx:end_idx]
            
            # Create figure with subplots - one row per neuron
            fig = plt.figure(figsize=(11, 8.5))  # Landscape orientation
            fig.suptitle(f"Neuronal Responses (Page {page+1} of {total_pages})", fontsize=16)
            
            # Create a grid for the layout
            nrows = len(page_neurons)
            gs = gridspec.GridSpec(nrows, 1, height_ratios=[1]*nrows)
            
            # Plot each neuron
            for i, neuron in enumerate(page_neurons):
                ax = plt.subplot(gs[i])
                
                # Track if any stimulus has data (for y-limit setting)
                has_data = False
                max_response = 0
                min_response = 0
                
                # Plot each stimulus type
                for stim in all_stimuli:
                    if stim in neuron_segments_dict[neuron]:
                        trials = neuron_segments_dict[neuron][stim]
                        if trials:  # If we have trials for this stimulus
                            has_data = True
                            
                            # Extract time series data from all trials
                            all_series = []
                            for trial in trials:
                                # Choose 'deltaFoverF_0' or 'z_scored' as needed
                                if 'deltaFoverF_0' in trial:
                                    time_series = trial['deltaFoverF_0']
                                elif 'z_scored' in trial:
                                    time_series = trial['z_scored']
                                else:
                                    continue
                                    
                                all_series.append(time_series)
                                
                            # Get time points (adjust x-axis if needed)
                            if all_series:
                                t = np.arange(len(all_series[0])) - 30  # Assuming stimulus at t=30
                                
                                # Calculate mean and SEM
                                all_series_array = np.array([s[:len(t)] for s in all_series if len(s) >= len(t)])
                                if len(all_series_array) > 0:
                                    mean_response = np.mean(all_series_array, axis=0)
                                    sem_response = stats.sem(all_series_array, axis=0)
                                    
                                    # Update min/max response
                                    max_response = max(max_response, np.max(mean_response + sem_response))
                                    min_response = min(min_response, np.min(mean_response - sem_response))
                                    
                                    # Plot mean response
                                    color = stimulus_colors.get(stim, 'gray')
                                    stim_info = odor_information.get(stim, stim)
                                    ax.plot(t, mean_response, label=stim_info, color=color, linewidth=1.5)
                                    
                                    # Plot error bars (SEM)
                                    ax.fill_between(t, mean_response - sem_response, 
                                                  mean_response + sem_response, 
                                                  color=color, alpha=0.2)
                
                # Add stimulus period highlight
                ax.axvspan(0, 50, alpha=0.1, color='yellow')
                
                # Adjust y-limits if we have data
                if has_data:
                    buffer = (max_response - min_response) * 0.1  # 10% buffer
                    ax.set_ylim(min_response - buffer, max_response + buffer)
                
                # Set title and labels
                ax.set_title(f"Neuron: {neuron}", fontsize=10, loc='left')
                
                # Only add x-label on bottom plot
                if i == len(page_neurons) - 1:
                    ax.set_xlabel("Time (frames relative to stimulus onset)")
                    
                ax.set_ylabel("∆F/F₀")
                
                # Add stimulus legend on first plot only
                if i == 0:
                    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
                
                # Add grid lines
                # ax.grid(True, linestyle='--', alpha=0.3)
                ax.grid(False)  # Disable grid for cleaner look
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
            
            # Add page to PDF
            pdf.savefig()
            plt.close()
    
    print(f"PDF saved to {output_file}")
    return output_file

def combine_neuron_pairs(neuron_segments_dict):
    """Combine left/right neuron pairs by averaging their responses"""
    combined_dict = {}
    
    # Group neurons by their base name (without L/R suffix)
    neuron_groups = {}
    for neuron in neuron_segments_dict:
        if neuron.endswith('L') or neuron.endswith('R'):
            base_name = neuron[:-1]  # Remove the last character (L or R)
            if base_name not in neuron_groups:
                neuron_groups[base_name] = []
            neuron_groups[base_name].append(neuron)
    
    # Process each group
    for base_name, neurons in neuron_groups.items():
        if len(neurons) == 2:
            # Make sure they're actually an L/R pair
            if (neurons[0].endswith('L') and neurons[1].endswith('R')) or \
               (neurons[0].endswith('R') and neurons[1].endswith('L')):
                # Create combined entry
                combined_dict[base_name] = {}
                
                # Find common stimuli
                stimuli_sets = [set(neuron_segments_dict[n].keys()) for n in neurons]
                common_stimuli = stimuli_sets[0].intersection(*stimuli_sets[1:])
                
                # Combine responses for common stimuli
                for stimulus in common_stimuli:
                    combined_trials = []
                    
                    # Find max number of trials among neurons
                    max_trials = max(len(neuron_segments_dict[n][stimulus]) for n in neurons)
                    
                    # Combine each trial
                    for trial_idx in range(max_trials):
                        combined_trial = {}
                        
                        # Get trials from both neurons (if available)
                        valid_trials = []
                        for n in neurons:
                            if trial_idx < len(neuron_segments_dict[n][stimulus]):
                                valid_trials.append(neuron_segments_dict[n][stimulus][trial_idx])
                        
                        if valid_trials:
                            # Average the values for each key
                            for key in valid_trials[0]:
                                if isinstance(valid_trials[0][key], np.ndarray):
                                    # For time series data, average across neurons
                                    series_data = [t[key] for t in valid_trials if key in t]
                                    min_len = min(len(s) for s in series_data)
                                    truncated_data = [s[:min_len] for s in series_data]
                                    combined_trial[key] = np.mean(truncated_data, axis=0)
                                else:
                                    # For metadata, just take from first trial
                                    combined_trial[key] = valid_trials[0][key]
                            
                            combined_trials.append(combined_trial)
                    
                    combined_dict[base_name][stimulus] = combined_trials
        else:
            # Keep neurons with only one side
            if len(neurons) == 1:
                combined_dict[neurons[0]] = neuron_segments_dict[neurons[0]]
    
    # Add neurons that don't follow the L/R pattern
    for neuron in neuron_segments_dict:
        if not (neuron.endswith('L') or neuron.endswith('R')):
            combined_dict[neuron] = neuron_segments_dict[neuron]
    
    return combined_dict

def create_stimulus_color_map(stimuli):
    """Create a color map for the given stimuli"""
    # Base colors for stimulus categories
    base_colors = {
        'c1': '#FF9AA2',  # Pinkish for EGCG
        'c2': '#C7CEEA',  # Blue for Quininic acid
        'c3': '#B5EAD7',  # Green for TF
        'c4': '#FFDAC1',  # Orange for L-Theanine
        'c5': '#E2F0CB',  # Light green for caffeine
    }
    
    color_map = {}
    for stim in stimuli:
        # Extract the compound code (e.g., 'c1', 'c2')
        parts = stim.split('_')
        if len(parts) > 0:
            compound = parts[0]
            if compound in base_colors:
                # Get base color and adjust intensity based on concentration
                base_color = base_colors[compound]
                if len(parts) > 1:
                    try:
                        # Adjust darkness based on concentration number
                        conc_num = int(parts[1])
                        # Convert hex to RGB, adjust, convert back to hex
                        r = int(base_color[1:3], 16)
                        g = int(base_color[3:5], 16)
                        b = int(base_color[5:7], 16)
                        
                        # Darken for higher concentrations
                        factor = 0.7 + (conc_num * 0.05)  # Darker for higher numbers
                        r = min(255, int(r * factor))
                        g = min(255, int(g * factor))
                        b = min(255, int(b * factor))
                        
                        color_map[stim] = f'#{r:02x}{g:02x}{b:02x}'
                    except ValueError:
                        color_map[stim] = base_color
                else:
                    color_map[stim] = base_color
            else:
                # Fallback to a default color scheme
                color_map[stim] = plt.cm.tab20(hash(stim) % 20)
        else:
            color_map[stim] = plt.cm.tab20(hash(stim) % 20)
    
    return color_map


def plot_neurons_to_html(neuron_segments_dict, odor_information, output_file, 
                        neurons_to_include=None, stimuli_to_include=None, if_combine=False):
    """
    Create an interactive HTML visualization of neuron responses using Plotly.
    
    Parameters:
    -----------
    neuron_segments_dict : dict
        Dictionary with structure {neuron_id: {stimulus_type: [trial_data, ...]}}
    odor_information : dict
        Dictionary mapping stimulus codes to descriptions
    output_file : str
        Path to save the HTML file
    neurons_to_include : list or None
        List of neuron IDs to include (None = all neurons)
    stimuli_to_include : list or None
        List of stimulus types to include (None = all stimuli)
    if_combine : bool
        If True, combine left and right neuron pairs by averaging responses
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # If combining L/R neurons, transform the dictionary
    if if_combine:
        neuron_segments_dict = combine_neuron_pairs(neuron_segments_dict)
    
    # Get all neurons
    all_neurons = sorted(neuron_segments_dict.keys())
    if neurons_to_include:
        all_neurons = [n for n in all_neurons if n in neurons_to_include]
    
    # Get all unique stimuli across all neurons
    all_stimuli = set()
    for neuron in all_neurons:
        if neuron in neuron_segments_dict:
            all_stimuli.update(neuron_segments_dict[neuron].keys())
    all_stimuli = sorted(all_stimuli)
    
    # Filter stimuli if specified
    if stimuli_to_include:
        all_stimuli = [s for s in all_stimuli if s in stimuli_to_include]
    
    # Create stimulus color map
    stimulus_colors = create_stimulus_color_map(all_stimuli)
    
    # Convert odor codes to descriptions
    stimulus_descriptions = {}
    for stim in all_stimuli:
        if stim in odor_information:
            stimulus_descriptions[stim] = odor_information[stim]
        else:
            stimulus_descriptions[stim] = stim
    
    # Create figure with tabs for each neuron
    fig = go.Figure()
    
    # For each neuron
    for neuron_idx, neuron in enumerate(all_neurons):
        # Create neuron-specific traces
        neuron_traces = []
        
        # Set visibility for first neuron
        is_visible = neuron_idx == 0
        
        # Plot each stimulus type
        for stim in all_stimuli:
            if stim in neuron_segments_dict[neuron]:
                trials = neuron_segments_dict[neuron][stim]
                if trials:  # If we have trials for this stimulus
                    # Extract time series data from all trials
                    all_series = []
                    for trial in trials:
                        # Choose 'deltaFoverF_0' or 'z_scored' as needed
                        if 'deltaFoverF_0' in trial:
                            time_series = trial['deltaFoverF_0']
                        elif 'z_scored' in trial:
                            time_series = trial['z_scored']
                        else:
                            continue
                            
                        all_series.append(time_series)
                    
                    # Get time points (adjust x-axis if needed)
                    if all_series:
                        t = np.arange(len(all_series[0])) - 30  # Assuming stimulus at t=30
                        
                        # Calculate mean and SEM
                        all_series_array = np.array([s[:len(t)] for s in all_series if len(s) >= len(t)])
                        if len(all_series_array) > 0:
                            mean_response = np.mean(all_series_array, axis=0)
                            sem_response = stats.sem(all_series_array, axis=0)
                            
                            # Plot mean response
                            color = stimulus_colors.get(stim, 'gray')
                            description = stimulus_descriptions.get(stim, stim)
                            
                            # Create mean line
                            mean_trace = go.Scatter(
                                x=t, 
                                y=mean_response,
                                mode='lines',
                                line=dict(color=color, width=2),
                                name=description,
                                legendgroup=stim,
                                visible=is_visible,
                                hovertemplate=f"Time: %{{x}}<br>Response: %{{y:.3f}}<br>{description}<extra></extra>"
                            )

                            # Use mcolors.to_rgba instead of plt.to_rgba
                            rgba = mcolors.to_rgba(color)
                            fill_color = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},0.2)'
                            
                            # Create error band
                            error_trace = go.Scatter(
                                x=np.concatenate([t, t[::-1]]),
                                y=np.concatenate([mean_response + sem_response, 
                                                (mean_response - sem_response)[::-1]]),
                                fill='toself',
                                fillcolor=fill_color,
                                line=dict(color='rgba(255,255,255,0)'),
                                name=description + " (SEM)",
                                legendgroup=stim,
                                showlegend=False,
                                visible=is_visible,
                                hoverinfo='none'
                            )
                            
                            # Add traces for this stimulus
                            neuron_traces.append(mean_trace)
                            neuron_traces.append(error_trace)
                            
                            # Add individual trials as well (initially hidden)
                            for i, series in enumerate(all_series_array):
                                trial_trace = go.Scatter(
                                    x=t,
                                    y=series,
                                    mode='lines',
                                    line=dict(color=color, width=1),
                                    opacity=0.3,
                                    name=f"{description} - Trial {i+1}",
                                    legendgroup=stim,
                                    showlegend=False,
                                    visible='legendonly',  # Hidden by default, show on legend click
                                    hoverinfo='none'
                                )
                                neuron_traces.append(trial_trace)
        
        # Add all traces to the figure
        for trace in neuron_traces:
            fig.add_trace(trace)
    
    # Add buttons for switching between neurons
    buttons = []
    for i, neuron in enumerate(all_neurons):
        visible = [False] * len(fig.data)
        
        # Find all traces for this neuron
        start_idx = 0
        for j in range(i):
            # Count traces for previous neurons
            if j < len(all_neurons):
                prev_neuron = all_neurons[j]
                for stim in all_stimuli:
                    if prev_neuron in neuron_segments_dict and stim in neuron_segments_dict[prev_neuron]:
                        trials = neuron_segments_dict[prev_neuron][stim]
                        if trials:
                            # 2 traces per stimulus (mean + error) + individual trials
                            start_idx += 2 + len(trials)
        
        # Set traces for current neuron to visible
        end_idx = start_idx
        for stim in all_stimuli:
            if neuron in neuron_segments_dict and stim in neuron_segments_dict[neuron]:
                trials = neuron_segments_dict[neuron][stim]
                if trials:
                    # Set mean and error bands to visible
                    visible[end_idx] = True
                    visible[end_idx + 1] = True
                    
                    # Individual trials remain 'legendonly'
                    for j in range(2, 2 + len(trials)):
                        if end_idx + j < len(visible):
                            visible[end_idx + j] = 'legendonly'
                    
                    end_idx += 2 + len(trials)
        
        button = dict(
            label=neuron,
            method="update",
            args=[{"visible": visible}, 
                 {"title": f"Neuron: {neuron} - Responses to Different Stimuli"}]
        )
        buttons.append(button)
    
    # Create dropdown menu
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            active=0,
            x=1.0,
            y=1.15,
            buttons=buttons
        )]
    )
    
    # Add a yellow rectangle for the stimulus period
    fig.add_shape(
        type="rect",
        x0=0,
        x1=50,
        y0=-100,  # Will be adjusted in the update_layout
        y1=100,   # Will be adjusted in the update_layout
        fillcolor="yellow",
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    # Update layout
    fig.update_layout(
        title=f"Neuron: {all_neurons[0]} - Responses to Different Stimuli",
        xaxis_title="Time (frames relative to stimulus onset)",
        yaxis_title="∆F/F₀",
        hovermode="closest",
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        legend=dict(
            x=1.05,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.2)',
            zerolinewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.2)',
            zerolinewidth=1
        ),
        height=700,
        margin=dict(l=50, r=50, t=100, b=50),
        annotations=[
            dict(
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                text="Select Neuron:",
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    # Auto-adjust y-axis limits based on data
    y_min = float('inf')
    y_max = float('-inf')
    for trace in fig.data:
        if hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
            y_min = min(y_min, min(y for y in trace.y if y is not None))
            y_max = max(y_max, max(y for y in trace.y if y is not None))
    
    # Add some padding
    padding = (y_max - y_min) * 0.1
    fig.update_layout(
        yaxis=dict(range=[y_min - padding, y_max + padding])
    )
    
    # Adjust stimulus rectangle
    fig.update_shapes(dict(y0=y_min - padding, y1=y_max + padding))
    
    # Save to HTML
    fig.write_html(output_file, include_plotlyjs='cdn', full_html=True)
    
    print(f"Interactive HTML visualization saved to {output_file}")
    return output_file

#%%
if __name__ == "__main__":
    # Example usage
    neuron_segments_dict = {
        'Neuron1': {
            'c1_1': [{'deltaFoverF_0': np.random.rand(100), 'z_scored': np.random.rand(100)}],
            'c2_1': [{'deltaFoverF_0': np.random.rand(100), 'z_scored': np.random.rand(100)}]
        },
        'Neuron2': {
            'c1_1': [{'deltaFoverF_0': np.random.rand(100), 'z_scored': np.random.rand(100)}],
            'c3_1': [{'deltaFoverF_0': np.random.rand(100), 'z_scored': np.random.rand(100)}]
        }
    }
    
    odor_information = {
        'c1_1': 'EGCG',
        'c2_1': 'Quininic acid',
        'c3_1': 'TF'
    }
    
    output_pdf = "neuron_responses.pdf"
    output_html = "neuron_responses.html"
    
    plot_neurons_to_pdf(neuron_segments_dict, odor_information, output_pdf)
    plot_neurons_to_html(neuron_segments_dict, odor_information, output_html)