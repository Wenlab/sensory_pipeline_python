#%%
# Import necessary libraries
import dash
from dash import dcc, html, Input, Output, State, callback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
import os
import json
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.HDF5Toolkit import load_h5file

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


def extract_metadata_from_dict(neuron_segments_dict):
    """
    Extract stim_name and stim_color mappings from the trial data embedded in the dictionary.
    
    Parameters:
    -----------
    neuron_segments_dict : dict
        Dictionary with structure {neuron_group: {stimulus_type: [trial data]}}
        Each trial should have 'stim_name' and 'stim_color' keys.
    
    Returns:
    --------
    odor_information : dict
        Mapping from stimulus_type to stim_name
    stimulus_color_map : dict
        Mapping from stimulus_type to stim_color
    """
    odor_information = {}
    stimulus_color_map = {}
    
    for neuron, stimuli in neuron_segments_dict.items():
        for stim_type, segments in stimuli.items():
            if segments:
                first_seg = segments[0]
                # Extract stim_name if not already present
                if stim_type not in odor_information:
                    odor_information[stim_type] = first_seg.get('stim_name', stim_type)
                # Extract stim_color if not already present
                if stim_type not in stimulus_color_map:
                    stimulus_color_map[stim_type] = first_seg.get('stim_color', '#808080')
    
    return odor_information, stimulus_color_map


def create_neuronal_dashboard(neuron_segments_dict, odor_information=None, stimulus_color_map=None):
    """
    Create a streamlined Dash app for visualizing neuronal responses.
    
    Parameters:
    -----------
    neuron_segments_dict : dict or pd.DataFrame
        Dictionary with structure {neuron_group: {stimulus_type: [trial data]}}
        or a DataFrame with columns 'neuron', 'stimulus', 'time_point', 'delta_F_over_F0', etc.
    odor_information : dict, optional
        Dictionary mapping stimulus codes to descriptions. If None, will try to extract from trial data.
    stimulus_color_map : dict, optional
        Dictionary mapping stimulus codes to color strings. If None, will try to extract from trial data.
    """
    
    # Check if input is a DataFrame
    is_dataframe = isinstance(neuron_segments_dict, pd.DataFrame)
    
    if is_dataframe:
        # Convert DataFrame to dict format for existing logic (temporary)
        # TODO: Implement native DataFrame support in generate_response_figure
        df = neuron_segments_dict
        all_neurons = sorted(df['neuron'].unique().tolist())
        all_stimuli = sorted(df['stimulus'].unique().tolist())
        
        # Extract metadata from DataFrame
        if odor_information is None:
            odor_information = df.drop_duplicates('stimulus').set_index('stimulus')['stim_name'].to_dict() if 'stim_name' in df.columns else {}
        if stimulus_color_map is None:
            stimulus_color_map = df.drop_duplicates('stimulus').set_index('stimulus')['stim_color'].to_dict() if 'stim_color' in df.columns else {}
    else:
        # Standard dict input
        all_neurons = sorted(neuron_segments_dict.keys())
        
        # Extract all unique stimuli
        all_stimuli = set()
        for neuron in neuron_segments_dict:
            all_stimuli.update(neuron_segments_dict[neuron].keys())
        all_stimuli = sorted(all_stimuli)
        
        # Extract metadata from embedded trial data if not provided externally
        if odor_information is None or stimulus_color_map is None:
            extracted_odor_info, extracted_color_map = extract_metadata_from_dict(neuron_segments_dict)
            if odor_information is None:
                odor_information = extracted_odor_info
            if stimulus_color_map is None:
                stimulus_color_map = extracted_color_map
    
    # Generate colors automatically for all stimuli if still empty
    if not stimulus_color_map:
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
                        {'label': 'Combine neurons (L/R)', 'value': 'combine_neurons'},
                        {'label': 'Show date difference', 'value': 'show_date_difference'}
                    ],
                    value=[],
                    labelStyle={'display': 'block', 'marginBottom': '5px'}
                ),
            ], style={'marginBottom': '15px'}),
            
            html.Button('Update Plot', id='update-plot-button', 
                       style={'marginTop': '10px', 'padding': '10px 20px'}),
            html.Div([
                html.Label("Save Path:", style={'margin-right': '10px'}),
                dcc.Input(
                    id='save-path-input',
                    type='text',
                    value='my_neuron_plot.html',
                    style={'width': '300px', 'margin-right': '10px'}
                ),
                html.Button('Save Plot as HTML', id='save-html-button'),
                html.Span(id='save-feedback', style={'margin-left': '10px'})
            ], style={'marginTop': '15px'}),

        ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Add a fixed height container for the plot
        html.Div([
            dcc.Graph(id='response-plot', style={'height': '100%', 'width': '100%'}, responsive=True),
        ], id='plot-container', style={'height': '800px', 'width': '100%'})
    ])
    
    def generate_response_figure(selected_neurons, selected_stimuli, display_type, combine_options):
        """
        Generates the plotly figure and container style based on user selections.
        This function contains the core plotting logic extracted from update_plot.
        """
        if not selected_neurons or not selected_stimuli:
            return go.Figure(), {'height': '800px', 'width': '100%'}
        
        combine_compounds = 'combine_compounds' in combine_options
        combine_neurons = 'combine_neurons' in combine_options
        show_date_difference = 'show_date_difference' in combine_options
        processed_neuron_dict = neuron_segments_dict
        
        # Get unique dates and define dash styles for date differentiation
        all_dates = set()
        if show_date_difference:
            for neuron in neuron_segments_dict:
                for stim in neuron_segments_dict[neuron]:
                    for seg in neuron_segments_dict[neuron][stim]:
                        if 'date' in seg:
                            all_dates.add(seg['date'])
        all_dates = sorted(list(all_dates))
        
        # Define dash styles for different dates
        dash_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
        date_dash_map = {date: dash_styles[i % len(dash_styles)] for i, date in enumerate(all_dates)}
        if combine_neurons:
            processed_neuron_dict = combine_lr_neurons(neuron_segments_dict)
            neuron_mapping = create_neuron_mapping(neuron_segments_dict)
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
            horizontal_spacing=0.00,
            vertical_spacing=0.00,  # Reduced spacing to maximize plot area
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
                y_min = np.percentile(min_values, 5)  # 5th percentile for lower bound
                y_max = np.percentile(max_values, 95)  # 95th percentile for upper bound
                # Add 15% buffer on each side
                buffer = (y_max - y_min) * 0.15
                neuron_y_ranges[neuron] = [y_min - buffer, y_max + buffer]
            else:
                # Default range if no data
                neuron_y_ranges[neuron] = [-0.5, 0.5]

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
                    legendgroup=group_key,
                    showlegend=True
                )
            )
        
        # Plot each neuron
        for row_idx, neuron in enumerate(selected_neurons, 1):
            # Use neuron-specific y-range
            y_range = neuron_y_ranges[neuron]
            
            # # Add neuron label on the left side of each row
            # if row_idx == 1:
            #     # Add a global y-axis title for the first row
            #     fig.update_layout(
            #         yaxis_title="ΔF/F0",
            #     )
            
            # # Add neuron name as y-axis title for each row
            # fig.update_yaxes(
            #     title=dict(
            #         text=f"<b>{neuron}</b>",
            #         font=dict(size=12),
            #         standoff=15  # Space between axis and title
            #     ),
            #     row=row_idx, col=1
            # )
            n_rows = len(selected_neurons)
            y_pos = 1 - (row_idx - 0.5) / n_rows
            fig.add_annotation(
                text=f"<b>{neuron}</b>",
                xref="x domain", yref="y domain",
                x=-0.02, y=0.5,
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                row=row_idx, col=1,
                font=dict(size=12, color="black")
            )

            if row_idx == 1:
                fig.add_annotation(
                    text="ΔF/F0",
                    xref="paper", yref="paper",
                    x=-0.08, y=1.05, # Position above the first row
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )

            fig.update_yaxes(
                showline=True,
                visible=True,
                showticklabels=True,
                row=row_idx, col=1
            )
            
            for col_idx, (group_key, stim_list) in enumerate(grouped_stimuli.items(), 1):
                # Collect all segments for this neuron and stimulus group
                all_segments = []
                for stim in stim_list:
                    if neuron in processed_neuron_dict and stim in processed_neuron_dict[neuron]:
                        all_segments.extend(processed_neuron_dict[neuron][stim])
                
                if not all_segments:
                    continue
                start_time = all_segments[0].get("start_time", 5)
                endtime = all_segments[0].get("end_time", 14)
                # Get color for stimulus group
                if combine_compounds:
                    # Use the color of the first stimulus in the group
                    highlight_color = stimulus_color_map.get(stim_list[0], 'gray')
                else:
                    highlight_color = stimulus_color_map.get(group_key, 'gray')
                
                # Add stimulus highlighting
                fig.add_shape(
                    type="rect",
                    x0=0, x1=endtime-start_time,
                    y0=y_range[0], y1=y_range[1],  # Use neuron-specific y-range
                    fillcolor=highlight_color,
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                    row=row_idx, col=col_idx
                )

                # Determine legend group name for this stimulus
                legend_group_name = group_key
                
                if display_type == 'individual':
                    # Plot individual traces
                    for seg in all_segments:
                        values = seg['deltaFoverF_0']
                        # start_time = seg.get('start_time', 6)
                        x_values = np.arange(len(values)) - start_time
                        # Create hover text with worm_key and date
                        seg_date = seg.get('date', 'unknown')
                        hover_text = f"{seg.get('worm_key', '')}_{seg.get('segment_index', '')}_{seg_date}"
                        
                        # Use date-based legendgroup if show_date_difference is enabled
                        trace_legendgroup = f"date_{seg_date}" if show_date_difference else legend_group_name
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=values,
                                mode='lines',
                                line=dict(width=1, color=highlight_color),
                                opacity=0.4,
                                showlegend=False,
                                legendgroup=trace_legendgroup,
                                hovertemplate=(
                                    f"{hover_text}<br>"
                                    f"x: %{{x}}<br>"
                                    f"y: %{{y:.3f}}<br>"
                                    f"N: {len(all_segments)}<br>"
                                    f"{neuron} - {get_stimulus_label(group_key, odor_information)}"
                                ),
                            ),
                            row=row_idx, col=col_idx
                        )
                else:
                    # Calculate and plot mean ± SEM
                    if show_date_difference and all_dates:
                        # Group segments by date and plot separately
                        segments_by_date = {}
                        for seg in all_segments:
                            date = seg.get('date', 'unknown')
                            if date not in segments_by_date:
                                segments_by_date[date] = []
                            segments_by_date[date].append(seg)
                        
                        for date in all_dates:
                            if date not in segments_by_date:
                                continue
                            date_segments = segments_by_date[date]
                            all_data_raw = [seg['deltaFoverF_0'] for seg in date_segments]
                            if not all_data_raw:
                                continue
                            min_len = min(len(data) for data in all_data_raw)
                            all_data_truncated = [data[:min_len] for data in all_data_raw]
                            all_data = np.array(all_data_truncated)
                            
                            mean_data = np.mean(all_data, axis=0)
                            sem_data = stats.sem(all_data, axis=0)
                            x_values = np.arange(min_len) - start_time
                            
                            dash_style = date_dash_map.get(date, 'solid')
                            hover_text = f"{neuron} - {get_stimulus_label(group_key, odor_information)} ({date})"
                            
                            # Legend group for date-based traces
                            date_legend_group = f"date_{date}"
                            
                            # Plot mean line with date-specific dash style
                            fig.add_trace(
                                go.Scatter(
                                    x=x_values,
                                    y=mean_data,
                                    mode='lines',
                                    line=dict(color=highlight_color, width=2, dash=dash_style),
                                    showlegend=False,
                                    legendgroup=date_legend_group,
                                    hovertemplate=(
                                        f"{hover_text}<br>"
                                        f"x: %{{x}}<br>"
                                        f"y: %{{y:.3f}}<br>"
                                        f"N: {len(date_segments)}"
                                    ),
                                ),
                                row=row_idx, col=col_idx
                            )
                            
                            # Plot SEM band
                            fig.add_trace(
                                go.Scatter(
                                    x=np.concatenate([x_values, x_values[::-1]]),
                                    y=np.concatenate([mean_data + sem_data, (mean_data - sem_data)[::-1]]),
                                    fill='toself',
                                    fillcolor=f'rgba{tuple(int(highlight_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    showlegend=False,
                                    legendgroup=date_legend_group,
                                    hoverinfo='none'
                                ),
                                row=row_idx, col=col_idx
                            )
                    else:
                        # Original behavior: aggregate all dates together
                        all_data_raw = [seg['deltaFoverF_0'] for seg in all_segments]
                        if not all_data_raw:
                            continue
                        # Make sure all arrays have the same length
                        min_len = min(len(data) for data in all_data_raw)
                        # Truncate all arrays to the same length
                        all_data_truncated = [data[:min_len] for data in all_data_raw]
                        all_data = np.array(all_data_truncated)
                        # start_time = all_segments[0].get('start_time', 6)

                        mean_data = np.mean(all_data, axis=0)
                        sem_data = stats.sem(all_data, axis=0)
                        x_values = np.arange(min_len) - start_time
                        # Create hover text
                        hover_text = f"{neuron} - {get_stimulus_label(group_key, odor_information)}"
                        # Plot mean line
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=mean_data,
                                mode='lines',
                                line=dict(color=highlight_color, width=2),
                                showlegend=False,
                                legendgroup=legend_group_name,
                                hovertemplate=(
                                        f"{hover_text}<br>"
                                        f"x: %{{x}}<br>"
                                        f"y: %{{y:.3f}}<br>"
                                        f"N: {len(all_segments)}"
                                    ),
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
                                legendgroup=legend_group_name,
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
                        title_text="Time(s)", 
                        row=row_idx, 
                        col=col_idx
                    )
        
        # Add legend traces to the figure (in first subplot)
        for trace in legend_traces:
            fig.add_trace(trace, row=1, col=1)
        
        # Add date legend traces if show_date_difference is enabled
        if show_date_difference and all_dates:
            for date in all_dates:
                dash_style = date_dash_map.get(date, 'solid')
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='lines',
                        line=dict(color='gray', width=2, dash=dash_style),
                        name=f"Date: {date}",
                        legendgroup=f"date_{date}",
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
        # Update layout to maintain fixed heights
        fig.update_layout(
            height=total_height,
            width=total_width,
            margin=dict(l=80, r=20, t=80, b=60),
            template="plotly_white",
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0,
                title="Stimulus / Date",
                bordercolor="White",
                borderwidth=0.5,
                itemclick="toggle",
                itemdoubleclick="toggleothers"
            )
        )
        
        return fig, {'height': f'{container_height}px', 'width': '100%'}

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
        fig, container_style = generate_response_figure(
            selected_neurons, selected_stimuli, display_type, combine_options
        )
        return fig, container_style
    
    @app.callback(
        Output('save-feedback', 'children'),
        Input('save-html-button', 'n_clicks'),
        [State('save-path-input', 'value'),      # 获取自定义路径
         State('neuron-selector', 'value'),      # 获取所有绘图选项
         State('stimuli-selector', 'value'),
         State('display-type-selector', 'value'),
         State('combine-options', 'value')]
    )
    def save_plot_html(n_clicks, save_path, selected_neurons, selected_stimuli, display_type, combine_options):
        if not n_clicks:
            # 防止在应用加载时触发
            raise dash.exceptions.PreventUpdate

        if not save_path:
            return "Error: Please provide a save path."
        
        if not selected_neurons or not selected_stimuli:
            return "Error: No plot data selected to save."

        try:
            # 1. 使用完全相同的逻辑重新生成图形
            # 我们只需要 fig 对象，所以用 _ 忽略 container_style
            fig, _ = generate_response_figure(
                selected_neurons, selected_stimuli, display_type, combine_options
            )
            
            # 2. 将图形保存到指定的 HTML 文件
            fig.write_html(save_path)
            
            # 3. 向用户返回成功消息
            return f"Plot successfully saved to {save_path}"
            
        except Exception as e:
            # 4. 返回错误消息
            return f"Error saving plot: {str(e)}"
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
        if base_name == 'ASE':
            continue
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
        # Special handling for ASE - do not combine
        if base_name == 'ASE':
            for neuron in neurons:
                combined_dict[neuron] = neuron_segments_dict[neuron]
            continue

        if len(neurons) == 2:  # If we have both L and R versions
            # Verify one ends with L and one with R
            has_left = any(n.endswith('L') for n in neurons)
            has_right = any(n.endswith('R') for n in neurons)
            
            if has_left and has_right:
                # Create new entry for the combined neuron
                combined_dict[base_name] = {}
                
                # Collect all stimuli from both neurons
                all_stimuli = set()
                for neuron in neurons:
                    all_stimuli.update(neuron_segments_dict[neuron].keys())
                
                for stim in all_stimuli:
                    combined_dict[base_name][stim] = []
                    for neuron in neurons:
                        if stim in neuron_segments_dict[neuron]:
                            # Extend the list with trials from this neuron
                            combined_dict[base_name][stim].extend(neuron_segments_dict[neuron][stim])
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

def run_neuron_dashboard(neuron_segments_dict, odor_information=None, stimulus_color_map=None, port=8050):
    """
    Main function to create and run the neuron activity visualization dashboard.
    
    Parameters:
    -----------
    neuron_segments_dict : dict
        Dictionary with structure {neuron_group: {stimulus_type: [trial data]}}
    odor_information : dict, optional
        Dictionary mapping stimulus codes to descriptions
    """
    app = create_neuronal_dashboard(neuron_segments_dict, odor_information, stimulus_color_map)

    # Run the app
    app.run(host="0.0.0.0", port= port, debug=True, jupyter_mode='external')

    return app

if __name__ == "__main__":
    # Load data and odor information
    # neuron_segments_dict = load_h5file(data_path, root_name='neuron_segments_dict')
    neuron_segments_dict = load_h5file(r"I:\WJH\flavor\Albert_data\neuron_segments_dict.h5", root_name='neuron_segments_dict')
    # with open(info_path, 'r', encoding='utf-8') as f:
    #     odor_information = json.load(f)
    
    
    # Start the visualization web app
    app = create_neuronal_dashboard(neuron_segments_dict, odor_information=None)
    app.run(host="0.0.0.0", port= 8056)