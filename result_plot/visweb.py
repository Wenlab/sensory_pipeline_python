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
from data_load.process_worm_data import transfer_dict2dataframe, transfer_dataframe2dict

#%%
def get_stimulus_label(stimulus_code, odor_information=None):
    """Convert stimulus code to readable label using odor_information if available."""
    if odor_information and stimulus_code in odor_information:
        return f"{stimulus_code}: {odor_information[stimulus_code]}"
    return stimulus_code

def convert_to_df(data):
    """
    Convert input data (dict or DataFrame) to a standardized DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        return transfer_dict2dataframe(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Must be dict or pd.DataFrame.")

def create_neuronal_dashboard(neuron_segments_data, odor_information=None, stimulus_color_map=None):
    """
    Create a streamlined Dash app for visualizing neuronal responses.
    
    Parameters:
    -----------
    neuron_segments_data : dict or pd.DataFrame
        Dictionary with structure {neuron_group: {stimulus_type: [trial data]}}
        or a DataFrame with columns 'neuron', 'stimulus', 'time_point', 'delta_F_over_F0', etc.
    odor_information : dict, optional
        Dictionary mapping stimulus codes to descriptions. If None, will try to extract from data.
    stimulus_color_map : dict, optional
        Dictionary mapping stimulus codes to color strings. If None, will try to extract from data.
    """
    
    # Standardize to DataFrame
    df_all = convert_to_df(neuron_segments_data)
    
    # Extract unique values
    all_neurons = sorted(df_all['neuron'].unique().tolist())
    all_stimuli = sorted(df_all['stimulus'].unique().tolist())
    
    # Extract metadata from DataFrame if not provided
    if odor_information is None:
        if 'stim_name' in df_all.columns:
            odor_information = df_all.drop_duplicates('stimulus').set_index('stimulus')['stim_name'].to_dict()
        else:
            odor_information = {s: s for s in all_stimuli}

    if stimulus_color_map is None:
        if 'stim_color' in df_all.columns:
            stimulus_color_map = df_all.drop_duplicates('stimulus').set_index('stimulus')['stim_color'].to_dict()
        else:
            stimulus_color_map = generate_compound_color_scheme(all_stimuli)
            
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
        Generates the plotly figure using DataFrame operations.
        """
        if not selected_neurons or not selected_stimuli:
            return go.Figure(), {'height': '800px', 'width': '100%'}
        
        combine_compounds = 'combine_compounds' in combine_options
        combine_neurons = 'combine_neurons' in combine_options
        show_date_difference = 'show_date_difference' in combine_options
        
        # 1. Filter and prepare local DataFrame copy
        dff = df_all[df_all['neuron'].isin(selected_neurons) & df_all['stimulus'].isin(selected_stimuli)].copy()
        if dff.empty:
            return go.Figure(), {'height': '800px', 'width': '100%'}
            
        dff['rel_time'] = dff['time_point'] - dff['start_time']
        
        # 2. Handle neuron combination (L/R)
        if combine_neurons:
            mapping = create_neuron_mapping(dff['neuron'].unique())
            dff['neuron_display'] = dff['neuron'].replace(mapping)
            # Update selected_neurons used for plotting order
            seen = set()
            new_selected_neurons = []
            for n in selected_neurons:
                combined_name = mapping.get(n, n)
                if combined_name not in seen:
                    new_selected_neurons.append(combined_name)
                    seen.add(combined_name)
            plot_neurons = new_selected_neurons
        else:
            dff['neuron_display'] = dff['neuron']
            plot_neurons = [n for n in selected_neurons if n in dff['neuron_display'].unique()]

        # 3. Handle stimulus groups
        grouped_stimuli = {}
        compound_to_stimuli = {}
        if combine_compounds:
            for stim in selected_stimuli:
                compound = stim.split('_')[0]
                if compound not in grouped_stimuli:
                    grouped_stimuli[compound] = []
                    compound_to_stimuli[compound] = []
                grouped_stimuli[compound].append(stim)
                compound_to_stimuli[compound].append(stim)
            dff['stimulus_group'] = dff['stimulus'].apply(lambda x: x.split('_')[0])
        else:
            for stim in selected_stimuli:
                grouped_stimuli[stim] = [stim]
                compound_to_stimuli[stim] = [stim]
            dff['stimulus_group'] = dff['stimulus']
        
        plot_groups = [g for g in grouped_stimuli.keys() if g in dff['stimulus_group'].unique()]

        # 4. Calculate plot dimensions
        fixed_height_per_neuron = 200
        total_height = fixed_height_per_neuron * len(plot_neurons)
        container_height = max(800, total_height + 50)
        base_width_per_stimulus = 400
        min_total_width = 1400
        total_width = max(min_total_width, len(plot_groups) * base_width_per_stimulus)
        
        # 5. Get date info and dash styles
        all_dates = sorted(dff['date'].unique().tolist())
        dash_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
        date_dash_map = {date: dash_styles[i % len(dash_styles)] for i, date in enumerate(all_dates)}

        # 6. Create subplots
        fig = make_subplots(
            rows=len(plot_neurons),
            cols=len(plot_groups),
            shared_yaxes=False,
            horizontal_spacing=0.00,
            vertical_spacing=0.00
        )
        
        # Calculate y-ranges per neuron display
        neuron_y_ranges = {}
        for neuron in plot_neurons:
            vals = dff[dff['neuron_display'] == neuron]['delta_F_over_F0']
            if not vals.empty:
                y_min = vals.quantile(0.05)
                y_max = vals.quantile(0.95)
                buffer = (y_max - y_min) * 0.15
                neuron_y_ranges[neuron] = [y_min - buffer, y_max + buffer]
            else:
                neuron_y_ranges[neuron] = [-0.5, 0.5]

        # 7. Add legend entries
        for group_key, stim_list in compound_to_stimuli.items():
            if combine_compounds:
                color = stimulus_color_map.get(stim_list[0], 'gray')
                legend_name = get_compound_name(group_key, odor_information)
            else:
                color = stimulus_color_map.get(group_key, 'gray')
                legend_name = get_stimulus_label(group_key, odor_information)
                
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color=color, width=2),
                name=legend_name, legendgroup=group_key, showlegend=True
            ), row=1, col=1)

        # 8. Plot data
        for row_idx, neuron in enumerate(plot_neurons, 1):
            y_range = neuron_y_ranges[neuron]
            
            # Neuron Label
            fig.add_annotation(
                text=f"<b>{neuron}</b>", xref="x domain", yref="y domain",
                x=-0.02, y=0.5, showarrow=False, xanchor="right", yanchor="middle",
                row=row_idx, col=1, font=dict(size=12, color="black")
            )
            if row_idx == 1:
                fig.add_annotation(
                    text="ΔF/F0", xref="paper", yref="paper",
                    x=-0.08, y=1.05, showarrow=False, font=dict(size=10, color="gray")
                )

            for col_idx, group_key in enumerate(plot_groups, 1):
                # Data for this cell
                cell_df = dff[(dff['neuron_display'] == neuron) & (dff['stimulus_group'] == group_key)]
                if cell_df.empty:
                    continue
                
                # Stimulus Highlight
                first_row = cell_df.iloc[0]
                start_time_rel = 0
                end_time_rel = first_row.get('end_time', 15) - first_row.get('start_time', 5)
                
                if combine_compounds:
                    highlight_color = stimulus_color_map.get(grouped_stimuli[group_key][0], 'gray')
                else:
                    highlight_color = stimulus_color_map.get(group_key, 'gray')
                
                fig.add_shape(
                    type="rect", x0=start_time_rel, x1=end_time_rel,
                    y0=y_range[0], y1=y_range[1],
                    fillcolor=highlight_color, opacity=0.15, layer="below", line_width=0,
                    row=row_idx, col=col_idx
                )

                if display_type == 'individual':
                    # Group by trial
                    trial_cols = ['worm_key', 'segment_index', 'date']
                    for trial_info, trial_data in cell_df.groupby(trial_cols):
                        worm, seg, date = trial_info
                        trace_legendgroup = f"date_{date}" if show_date_difference else group_key
                        fig.add_trace(go.Scatter(
                            x=trial_data['rel_time'], y=trial_data['delta_F_over_F0'],
                            mode='lines', line=dict(width=1, color=highlight_color),
                            opacity=0.4, showlegend=False, legendgroup=trace_legendgroup,
                            hovertemplate=(f"{worm}_{seg}_{date}<br>x: %{{x}}<br>y: %{{y:.3f}}<br>"
                                           f"{neuron} - {get_stimulus_label(group_key, odor_information)}")
                        ), row=row_idx, col=col_idx)
                else:
                    # Mean ± SEM
                    group_cols = ['neuron_display', 'stimulus_group', 'rel_time']
                    if show_date_difference:
                        group_cols.append('date')
                    
                    # Calculate stats
                    stats_df = cell_df.groupby(group_cols)['delta_F_over_F0'].agg(['mean', 'sem', 'count']).reset_index()
                    
                    if show_date_difference:
                        for date in all_dates:
                            date_stats = stats_df[stats_df['date'] == date]
                            if date_stats.empty: continue
                            
                            dash_style = date_dash_map.get(date, 'solid')
                            date_legend_group = f"date_{date}"
                            
                            fig.add_trace(go.Scatter(
                                x=date_stats['rel_time'], y=date_stats['mean'],
                                mode='lines', line=dict(color=highlight_color, width=2, dash=dash_style),
                                showlegend=False, legendgroup=date_legend_group,
                                hovertemplate=(f"{neuron} - {get_stimulus_label(group_key, odor_information)} ({date})<br>"
                                               f"x: %{{x}}<br>y: %{{y:.3f}}<br>N: %{{customdata}}"),
                                customdata=date_stats['count']
                            ), row=row_idx, col=col_idx)
                            
                            fig.add_trace(go.Scatter(
                                x=np.concatenate([date_stats['rel_time'], date_stats['rel_time'][::-1]]),
                                y=np.concatenate([date_stats['mean'] + date_stats['sem'], (date_stats['mean'] - date_stats['sem'])[::-1]]),
                                fill='toself',
                                fillcolor=f'rgba{tuple(int(highlight_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}',
                                line=dict(color='rgba(255,255,255,0)'),
                                showlegend=False, legendgroup=date_legend_group, hoverinfo='none'
                            ), row=row_idx, col=col_idx)
                    else:
                        fig.add_trace(go.Scatter(
                            x=stats_df['rel_time'], y=stats_df['mean'],
                            mode='lines', line=dict(color=highlight_color, width=2),
                            showlegend=False, legendgroup=group_key,
                            hovertemplate=(f"{neuron} - {get_stimulus_label(group_key, odor_information)}<br>"
                                           f"x: %{{x}}<br>y: %{{y:.3f}}<br>N: %{{customdata}}"),
                            customdata=stats_df['count']
                        ), row=row_idx, col=col_idx)
                        
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([stats_df['rel_time'], stats_df['rel_time'][::-1]]),
                            y=np.concatenate([stats_df['mean'] + stats_df['sem'], (stats_df['mean'] - stats_df['sem'])[::-1]]),
                            fill='toself',
                            fillcolor=f'rgba{tuple(int(highlight_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False, legendgroup=group_key, hoverinfo='none'
                        ), row=row_idx, col=col_idx)

                # Axes formatting
                fig.update_yaxes(range=y_range, row=row_idx, col=col_idx, 
                                 showgrid=False, showticklabels=(col_idx == 1))
                fig.update_xaxes(showgrid=False, row=row_idx, col=col_idx,
                                 showticklabels=(row_idx == len(plot_neurons)))
                if row_idx == len(plot_neurons):
                    fig.update_xaxes(title_text="Time(s)", row=row_idx, col=col_idx)

        # 9. Global legend adjustments
        if show_date_difference:
            for date in all_dates:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color='gray', width=2, dash=date_dash_map.get(date, 'solid')),
                    name=f"Date: {date}", legendgroup=f"date_{date}", showlegend=True
                ), row=1, col=1)
        
        fig.update_layout(
            height=total_height, width=total_width,
            margin=dict(l=80, r=20, t=80, b=60),
            template="plotly_white", hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title="Stimulus / Date")
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

def create_neuron_mapping(neuron_data):
    """
    Create a mapping from original neuron names to combined names.
    For L/R pairs, maps to base name (e.g., ADLL -> ADL, ADLR -> ADL).
    For others, maps to original name.
    
    Parameters:
    -----------
    neuron_data : dict or list or np.ndarray
        Either the neuron segments dict or a list of neuron names.
    """
    mapping = {}
    
    if isinstance(neuron_data, dict):
        neuron_names = neuron_data.keys()
    else:
        neuron_names = neuron_data
    
    # Find all neurons that might form L/R pairs
    lr_candidates = {}
    for neuron in neuron_names:
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
    neuron_segments_dict = load_h5file(r"./data/bacteria/20260121_merged_bacteria_segments.h5", root_name='neuron_segments')
    # with open(info_path, 'r', encoding='utf-8') as f:
    #     odor_information = json.load(f)
    
    
    # Start the visualization web app
    app = create_neuronal_dashboard(neuron_segments_dict, odor_information=None)
    app.run(host="0.0.0.0", port= 8056)