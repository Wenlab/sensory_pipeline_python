import dash
from dash import dcc, html, Input, Output, State, callback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import sys
import os

#%%
def preprocess_data_for_plotting(neuron_segments_dict):
    """
    Preprocess neuron segments data for faster plotting.
    
    Returns:
    --------
    dict: Preprocessed data with structure {neuron: {stimulus: [segment_data]}}
          where segment_data includes 'x_values', 'deltaFoverF_0', 'z_scored', etc.
    """
    preprocessed = {}
    
    for neuron, stimuli in neuron_segments_dict.items():
        preprocessed[neuron] = {}
        
        for stim, segments in stimuli.items():
            preprocessed[neuron][stim] = []
            
            for seg in segments:
                processed_seg = {}
                
                # Copy metadata
                for key in ['worm_key', 'segment_index', 'date']:
                    if key in seg:
                        processed_seg[key] = seg[key]
                    else:
                        processed_seg[key] = 'unknown'
                
                # Process time series data
                for data_key in ['deltaFoverF_0', 'z_scored', 'scaled_data']:
                    if data_key in seg and seg[data_key] is not None:
                        data = seg[data_key]
                        processed_seg[data_key] = data
                        
                        # Only compute x_values once
                        if 'x_values' not in processed_seg:
                            processed_seg['x_values'] = np.arange(len(data)) - 30
                
                if 'x_values' in processed_seg:  # Only add if we have some data
                    preprocessed[neuron][stim].append(processed_seg)
    
    return preprocessed

def get_stimulus_label(stimulus_code, odor_information=None):
    """Convert stimulus code to readable label using odor_information if available."""
    if odor_information and stimulus_code in odor_information:
        return f"{stimulus_code}: {odor_information[stimulus_code]}"
    return stimulus_code

def create_neuronal_dashboard(neuron_segments_dict, odor_information=None):
    """
    Create a streamlined Dash app for visualizing neuronal responses.
    
    Parameters:
    -----------
    neuron_segments_dict : dict
        Dictionary with structure {neuron_group: {stimulus_type: [trial data]}}
    odor_information : dict, optional
        Dictionary mapping stimulus codes to descriptions
    """
    
    # Extract all available data options
    all_neurons = sorted(neuron_segments_dict.keys())
    
    # Extract all unique stimuli
    all_stimuli = set()
    for neuron in neuron_segments_dict:
        all_stimuli.update(neuron_segments_dict[neuron].keys())
    all_stimuli = sorted(all_stimuli)
    
    # Generate colors automatically for all stimuli
    stimulus_color_map = generate_compound_color_scheme(all_stimuli)
    
    # Preprocess data for faster plotting
    preprocessed_data = preprocess_data_for_plotting(neuron_segments_dict)
    
    # Initialize Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Create app layout
    app.layout = html.Div([
        html.H1("Neuron Activity Visualization", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.Label("Select Neurons:"),
                dcc.Checklist(
                    id='neuron-selector',
                    options=[{'label': neuron, 'value': neuron} for neuron in all_neurons],
                    value=[all_neurons[0]] if all_neurons else [],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Select Stimuli:"),
                dcc.Checklist(
                    id='stimuli-selector',
                    options=[{'label': get_stimulus_label(s, odor_information), 'value': s} for s in all_stimuli],
                    value=[all_stimuli[0]] if all_stimuli else [],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Display:"),
                dcc.RadioItems(
                    id='display-type-selector',
                    options=[
                        {'label': 'Individual Trials', 'value': 'individual'},
                        {'label': 'Mean ± SEM', 'value': 'mean_sem'}
                    ],
                    value='mean_sem',
                    inline=True
                ),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                dcc.Checklist(
                    id='combine-options',
                    options=[
                        {'label': 'Combine compounds (different dilutions)', 'value': 'combine_compounds'},
                        {'label': 'Combine neurons (L/R)', 'value': 'combine_neurons'}
                    ],
                    value=[],
                    labelStyle={'display': 'block', 'marginBottom': '5px'}
                ),
            ], style={'marginBottom': '15px'}),
            
            html.Button('Update Plot', id='update-plot-button', 
                       style={'marginTop': '10px', 'padding': '10px 20px'}),
            
        ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Add a fixed height container for the plot
        html.Div([
            dcc.Graph(id='response-plot', style={'height': '100%', 'width': '100%'})
        ], id='plot-container', style={'height': '800px', 'width': '100%'})
    ])
    
    @app.callback(
        [Output('response-plot', 'figure'),
         Output('plot-container', 'style')],
        Input('update-plot-button', 'n_clicks'),
        [State('neuron-selector', 'value'),
         State('stimuli-selector', 'value'),
         State('display-type-selector', 'value'),
         State('combine-options', 'value')]
    )
    def update_plot(n_clicks, selected_neurons, selected_stimuli, display_type, combine_options):
        if not selected_neurons or not selected_stimuli:
            return go.Figure(), {'height': '800px', 'width': '100%'}
        
        # Process combination options
        combine_compounds = 'combine_compounds' in combine_options
        combine_neurons = 'combine_neurons' in combine_options
        
        # Apply neuron combination if needed
        processed_neuron_dict = neuron_segments_dict
        if combine_neurons:
            processed_neuron_dict = combine_lr_neurons(neuron_segments_dict)
            # Update selected neurons list based on combined neurons
            # First create a mapping from original neuron names to combined names
            neuron_mapping = create_neuron_mapping(neuron_segments_dict)
            
            # Then update the selected neurons list
            new_selected_neurons = []
            for neuron in selected_neurons:
                if neuron in neuron_mapping:
                    if neuron_mapping[neuron] not in new_selected_neurons:
                        new_selected_neurons.append(neuron_mapping[neuron])
                elif neuron in processed_neuron_dict:
                    new_selected_neurons.append(neuron)
            
            selected_neurons = new_selected_neurons if new_selected_neurons else [list(processed_neuron_dict.keys())[0]]
        
        # Process stimuli combination if needed
        grouped_stimuli = {}
        compound_to_stimuli = {}  # For legend creation - maps compound to list of stimuli

        if combine_compounds:
            # Group stimuli by compound type (e.g., c1_1, c1_2 → c1)
            for stim in selected_stimuli:
                compound = stim.split('_')[0]
                if compound not in grouped_stimuli:
                    grouped_stimuli[compound] = []
                    compound_to_stimuli[compound] = []
                grouped_stimuli[compound].append(stim)
                compound_to_stimuli[compound].append(stim)
        else:
            # Use stimuli as is
            for stim in selected_stimuli:
                grouped_stimuli[stim] = [stim]
                compound_to_stimuli[stim] = [stim]
        
        # Calculate dynamic plot dimensions based on number of neurons
        # Each neuron gets a fixed height regardless of total number
        fixed_height_per_neuron = 200  # Fixed height for each neuron row
        total_height = fixed_height_per_neuron * len(selected_neurons)
        
        # Calculate container height to fit all rows
        container_height = max(800, total_height + 50)  # Add some padding
        
        # Adjust width based on number of stimuli
        base_width_per_stimulus = 400
        min_total_width = 1400
        total_width = max(min_total_width, len(grouped_stimuli) * base_width_per_stimulus)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=len(selected_neurons),
            cols=len(grouped_stimuli),
            shared_yaxes=False,  # Allow different y-ranges per neuron
            horizontal_spacing=0.02,
            vertical_spacing=0.05,  # Reduced spacing to maximize plot area
            subplot_titles=[]  # We'll add custom titles
        )
        
        # Calculate y-ranges for each neuron
        neuron_y_ranges = {}
        for row_idx, neuron in enumerate(selected_neurons, 1):
            min_values = []
            max_values = []
            
            for group_key, stim_list in grouped_stimuli.items():
                # Collect all data for this neuron across all selected stimuli
                for stim in stim_list:
                    if neuron in processed_neuron_dict and stim in processed_neuron_dict[neuron]:
                        for seg in processed_neuron_dict[neuron][stim]:
                            values = seg['deltaFoverF_0']
                            min_values.append(np.min(values))
                            max_values.append(np.max(values))
            
            if min_values and max_values:
                # Calculate appropriate y-range with buffer
                y_min = min(min_values)
                y_max = max(max_values)
                # Add 15% buffer on each side
                buffer = (y_max - y_min) * 0.15
                neuron_y_ranges[neuron] = [y_min - buffer, y_max + buffer]
            else:
                # Default range if no data
                neuron_y_ranges[neuron] = [-0.2, 0.5]

        # Add an empty domain to the left for neuron labels
        # This creates space for the neuron labels
        fig.update_layout(
            grid={
                'rows': len(selected_neurons),
                'columns': len(grouped_stimuli),
                'xgap': 0.005,
                'ygap': 0.01
            }
        )
        
        # Create legend entries - one per grouped stimulus
        legend_traces = []
        for group_key, stim_list in compound_to_stimuli.items():
            # Determine color for the compound
            if combine_compounds:
                # Get color from first stimulus in group
                color = stimulus_color_map.get(stim_list[0], 'gray')
                legend_name = get_compound_name(group_key, odor_information)
            else:
                color = stimulus_color_map.get(group_key, 'gray')
                legend_name = get_stimulus_label(group_key, odor_information)
                
            # Create invisible trace just for legend
            legend_traces.append(
                go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=legend_name,
                    showlegend=True
                )
            )
        
        # Plot each neuron
        for row_idx, neuron in enumerate(selected_neurons, 1):
            # Use neuron-specific y-range
            y_range = neuron_y_ranges[neuron]
            
            # Add neuron label on the left side of each row
            if row_idx == 1:
                # Add a global y-axis title for the first row
                fig.update_layout(
                    yaxis_title="ΔF/F0"
                )
            
            # Add neuron name as y-axis title for each row
            fig.update_yaxes(
                title=dict(
                    text=f"<b>{neuron}</b>",
                    font=dict(size=12),
                    standoff=15  # Space between axis and title
                ),
                row=row_idx, col=1
            )
            
            for col_idx, (group_key, stim_list) in enumerate(grouped_stimuli.items(), 1):
                # Get color for stimulus group
                if combine_compounds:
                    # Use the color of the first stimulus in the group
                    highlight_color = stimulus_color_map.get(stim_list[0], 'gray')
                else:
                    highlight_color = stimulus_color_map.get(group_key, 'gray')
                
                # Add stimulus highlighting
                fig.add_shape(
                    type="rect",
                    x0=0, x1=50,
                    y0=y_range[0], y1=y_range[1],  # Use neuron-specific y-range
                    fillcolor=highlight_color,
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                    row=row_idx, col=col_idx
                )
                
                # Collect all segments for this neuron and stimulus group
                all_segments = []
                for stim in stim_list:
                    if neuron in processed_neuron_dict and stim in processed_neuron_dict[neuron]:
                        all_segments.extend(processed_neuron_dict[neuron][stim])
                
                if not all_segments:
                    continue
                    
                if display_type == 'individual':
                    # Plot individual traces
                    for seg in all_segments:
                        values = seg['deltaFoverF_0']
                        x_values = np.arange(len(values)) - 30
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=values,
                                mode='lines',
                                line=dict(width=1, color=highlight_color),
                                opacity=0.4,
                                showlegend=False
                            ),
                            row=row_idx, col=col_idx
                        )
                else:
                    # Calculate and plot mean ± SEM
                    all_data = np.array([seg['deltaFoverF_0'] for seg in all_segments])
                    # Make sure all arrays have the same length
                    min_len = min(len(data) for data in all_data)
                    all_data = np.array([data[:min_len] for data in all_data])
                    
                    mean_data = np.mean(all_data, axis=0)
                    sem_data = stats.sem(all_data, axis=0)
                    x_values = np.arange(min_len) - 30
                    
                    # Plot mean line
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=mean_data,
                            mode='lines',
                            line=dict(color=highlight_color, width=2),
                            showlegend=False
                        ),
                        row=row_idx, col=col_idx
                    )
                    
                    # Plot SEM band
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x_values, x_values[::-1]]),
                            y=np.concatenate([mean_data + sem_data, (mean_data - sem_data)[::-1]]),
                            fill='toself',
                            fillcolor=f'rgba{tuple(int(highlight_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            hoverinfo='none'
                        ),
                        row=row_idx, col=col_idx
                    )
                    
                # Set neuron-specific y-range
                fig.update_yaxes(
                    range=y_range, 
                    row=row_idx, 
                    col=col_idx, 
                    showgrid=False,
                    # Only show tick labels on the leftmost column
                    showticklabels=(col_idx == 1)
                )
                
                # Update x-axes properties
                fig.update_xaxes(
                    showgrid=False, 
                    row=row_idx, 
                    col=col_idx,
                    # Only show tick labels on the bottom row
                    showticklabels=(row_idx == len(selected_neurons))
                )
                
                # Only add x-label on bottom row
                if row_idx == len(selected_neurons):
                    fig.update_xaxes(
                        title_text="Time (frames)", 
                        row=row_idx, 
                        col=col_idx
                    )
        
        # Add legend traces to the figure (in first subplot)
        for trace in legend_traces:
            fig.add_trace(trace, row=1, col=1)
            
        # Update layout to maintain fixed heights
        fig.update_layout(
            height=total_height,
            width=total_width,
            margin=dict(l=80, r=100, t=40, b=60),  # Increased right margin for legend
            template="plotly_white",
            hovermode="closest",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=1.1,
                title="Stimulus",
                bordercolor="Black",
                borderwidth=1
            )
        )
        
        # Return both the figure and updated container style
        return fig, {'height': f'{container_height}px', 'width': '100%'}
    
    return app

def generate_compound_color_scheme(stimuli):
    """
    Generate a color scheme for compounds automatically, where the same types
    of compounds have similar colors, and different dilutions have similar but not identical colors.
    """
    import colorsys

    # Extract unique compound types (c1, c2, c3, etc.)
    compound_types = set()
    dilution_dict = {}
    for stim in stimuli:
        if '_' in stim:
            compound, dilution = stim.split('_')
            compound_types.add(compound)
            dilution_dict.setdefault(compound, set()).add(dilution)
    compound_types = sorted(list(compound_types))

    # Create base colors for each compound type
    base_colors = {}
    for i, compound in enumerate(compound_types):
        hue = i / len(compound_types)
        saturation = 0.8
        value = 0.85
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_int = tuple(int(255 * c) for c in rgb)
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_int)
        base_colors[compound] = (hue, saturation, value)

    # Create variations for each dilution in a compound series
    color_map = {}
    for stim in stimuli:
        if '_' in stim:
            compound, dilution = stim.split('_')
            if compound in base_colors:
                h, s, v = base_colors[compound]
                # Sort dilutions for consistent color steps
                dilution_list = sorted(list(dilution_dict[compound]), key=lambda x: int(x))
                dilution_idx = dilution_list.index(dilution)
                n_dilutions = len(dilution_list)
                # Adjust value (lightness) for each dilution
                # Lower dilution_idx = darker, higher = lighter
                v_adjust = v + (dilution_idx - n_dilutions // 2) * (0.18 / max(n_dilutions-1, 1))
                v_adjust = min(1.0, max(0.55, v_adjust))  # Clamp to [0.55, 1.0]
                # Optionally, adjust saturation a bit too
                s_adjust = s - (dilution_idx) * (0.15 / max(n_dilutions-1, 1))
                s_adjust = min(1.0, max(0.45, s_adjust))
                rgb = colorsys.hsv_to_rgb(h, s_adjust, v_adjust)
                rgb_int = tuple(int(255 * c) for c in rgb)
                color_map[stim] = '#{:02x}{:02x}{:02x}'.format(*rgb_int)
            else:
                color_map[stim] = '#808080'
        else:
            color_map[stim] = '#808080'
    return color_map

def get_compound_name(compound_key, odor_information=None):
    """
    Get a descriptive name for a compound group, possibly using odor information.
    
    Parameters:
    -----------
    compound_key : str
        Compound key (e.g., 'c1')
    odor_information : dict, optional
        Dictionary mapping stimulus keys to descriptions
    
    Returns:
    --------
    str
        Descriptive name for the compound
    """
    if odor_information:
        # Look for any stimulus from this compound group in the odor information
        matching_stimuli = [stim for stim in odor_information if stim.startswith(f"{compound_key}_")]
        if matching_stimuli:
            # Use the description of the first matching stimulus
            return f"{compound_key} ({odor_information[matching_stimuli[0]].split(' ')[0]})"
    
    return compound_key

def create_neuron_mapping(neuron_segments_dict):
    """
    Create a mapping from original neuron names to combined names.
    For L/R pairs, maps to base name (e.g., ADLL -> ADL, ADLR -> ADL).
    For others, maps to original name.
    """
    mapping = {}
    
    # Find all neurons that might form L/R pairs
    lr_candidates = {}
    for neuron in neuron_segments_dict.keys():
        if neuron.endswith('L') or neuron.endswith('R'):
            base_name = neuron[:-1]  # Remove the L or R suffix
            if base_name not in lr_candidates:
                lr_candidates[base_name] = []
            lr_candidates[base_name].append(neuron)
    
    # Create mappings for neurons that have both L and R versions
    for base_name, neurons in lr_candidates.items():
        if len(neurons) == 2:
            has_left = any(n.endswith('L') for n in neurons)
            has_right = any(n.endswith('R') for n in neurons)
            
            if has_left and has_right:
                for neuron in neurons:
                    mapping[neuron] = base_name
    
    return mapping

def combine_lr_neurons(neuron_segments_dict):
    """
    Combine left and right neuron pairs (e.g., ADLL and ADLR become ADL).
    Only combines neurons that have both L and R versions.
    Returns a new dictionary with combined neurons.
    """
    combined_dict = {}
    
    # Find neurons with L/R suffix
    neuron_groups = {}
    for neuron in neuron_segments_dict:
        if neuron.endswith('L') or neuron.endswith('R'):
            base_name = neuron[:-1]  # Remove the L or R suffix
            if base_name not in neuron_groups:
                neuron_groups[base_name] = []
            neuron_groups[base_name].append(neuron)
    
    # Combine neuron pairs
    for base_name, neurons in neuron_groups.items():
        if len(neurons) == 2:  # If we have both L and R versions
            # Verify one ends with L and one with R
            has_left = any(n.endswith('L') for n in neurons)
            has_right = any(n.endswith('R') for n in neurons)
            
            if has_left and has_right:
                left = next((n for n in neurons if n.endswith('L')), None)
                right = next((n for n in neurons if n.endswith('R')), None)
                
                if left and right:
                    # Create new entry for the combined neuron
                    combined_dict[base_name] = {}
                    
                    # Find common stimuli
                    left_stimuli = set(neuron_segments_dict[left].keys())
                    right_stimuli = set(neuron_segments_dict[right].keys())
                    common_stimuli = left_stimuli.intersection(right_stimuli)
                    
                    # Process each stimulus
                    for stim in common_stimuli:
                        combined_dict[base_name][stim] = []
                        
                        # Get segments from both neurons
                        left_segments = neuron_segments_dict[left][stim]
                        right_segments = neuron_segments_dict[right][stim]
                        
                        # Take the minimum number of segments
                        num_segments = min(len(left_segments), len(right_segments))
                        
                        # Average corresponding segments
                        for i in range(num_segments):
                            left_seg = left_segments[i]
                            right_seg = right_segments[i]
                            
                            # Create a new segment with averaged data
                            combined_seg = {}
                            
                            # Copy metadata from either neuron
                            for key in ['worm_key', 'date', 'segment_index']:
                                combined_seg[key] = left_seg.get(key, right_seg.get(key, 'unknown'))
                            
                            # Average the time series data
                            if 'deltaFoverF_0' in left_seg and 'deltaFoverF_0' in right_seg:
                                left_data = left_seg['deltaFoverF_0']
                                right_data = right_seg['deltaFoverF_0']
                                
                                # Make sure they're the same length
                                min_len = min(len(left_data), len(right_data))
                                left_data = left_data[:min_len]
                                right_data = right_data[:min_len]
                                
                                # Average the data
                                combined_seg['deltaFoverF_0'] = np.mean([left_data, right_data], axis=0)
                                
                                combined_dict[base_name][stim].append(combined_seg)
                    
                    # Add any stimuli that only one neuron has
                    all_stimuli = left_stimuli.union(right_stimuli)
                    for stim in all_stimuli - common_stimuli:
                        source_neuron = left if stim in left_stimuli else right
                        combined_dict[base_name][stim] = neuron_segments_dict[source_neuron][stim]
            else:
                # Keep individual neurons that don't have a pair
                for neuron in neurons:
                    combined_dict[neuron] = neuron_segments_dict[neuron]
        else:
            # Keep individual neurons that don't have a pair
            for neuron in neurons:
                combined_dict[neuron] = neuron_segments_dict[neuron]
    
    # Include neurons that don't have L/R suffix
    for neuron in neuron_segments_dict:
        if not (neuron.endswith('L') or neuron.endswith('R')):
            combined_dict[neuron] = neuron_segments_dict[neuron]
    
    return combined_dict