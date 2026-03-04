# Import necessary libraries
import sys
import os

import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_load.process_worm_data import transfer_dict2dataframe

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
    
    label_style = {
        'fontWeight': '600',
        'color': '#475569',
        'marginBottom': '8px',
        'display': 'block',
        'fontSize': '14px',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px'
    }
    
    # Create app layout inline styles
    glass_card_style = {
        'background': 'rgba(255, 255, 255, 0.7)',
        'backdropFilter': 'blur(16px)',
        'WebkitBackdropFilter': 'blur(16px)',
        'borderRadius': '16px',
        'border': '1px solid rgba(255, 255, 255, 0.8)',
        'boxShadow': '0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01)',
        'padding': '24px',
        'marginBottom': '24px'
    }
    
    btn_primary_style = {
        'background': 'linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%)',
        'color': 'white',
        'border': 'none',
        'borderRadius': '8px',
        'padding': '14px',
        'fontWeight': '600',
        'cursor': 'pointer',
        'transition': 'all 0.2s ease',
        'boxShadow': '0 4px 6px -1px rgba(79, 70, 229, 0.3)',
        'width': '100%',
        'fontSize': '15px',
        'marginBottom': '20px'
    }
    
    btn_secondary_style = {
        'background': 'white',
        'color': '#475569',
        'border': '1px solid #cbd5e1',
        'borderRadius': '8px',
        'padding': '12px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.2s ease',
        'width': '100%'
    }
    
    btn_secondary_small_style = {
        'background': 'white',
        'color': '#475569',
        'border': '1px solid #cbd5e1',
        'borderRadius': '6px',
        'padding': '6px 10px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.2s ease',
        'fontSize': '12px',
        'flex': '1'
    }
    
    custom_input_style = {
        'width': '100%',
        'boxSizing': 'border-box',
        'padding': '12px',
        'borderRadius': '8px',
        'border': '1px solid #e2e8f0',
        'marginBottom': '12px',
        'fontFamily': '"Inter", sans-serif',
        'fontSize': '14px',
        'transition': 'outline 0.2s'
    }
    
    section_title_style = {
        'marginTop': '0',
        'fontSize': '18px',
        'fontWeight': '700',
        'color': '#0f172a',
        'borderBottom': '2px solid #f1f5f9',
        'paddingBottom': '12px',
        'marginBottom': '20px'
    }

    app.layout = html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1("Neuron Activity Dashboard", 
                        style={'margin': '0', 'color': '#0f172a', 'fontSize': '32px', 'fontWeight': '800', 'letterSpacing': '-0.5px'}),
                html.P("Explore and analyze neuronal responses to various stimuli", 
                       style={'margin': '8px 0 0 0', 'color': '#64748b', 'fontSize': '16px'})
            ]),
            html.Button('Toggle Controls', id='toggle-sidebar-btn', style={**btn_secondary_small_style, 'flex': '0 0 auto', 'padding': '10px 16px', 'fontSize': '14px', 'fontWeight': '600'})
        ], style={**glass_card_style, 'background': 'linear-gradient(135deg, #e0e7ff 0%, #f8fafc 100%)', 'padding': '32px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
        
        html.Div([
            # Sidebar / Controls
            html.Div(id='sidebar-container', children=[
                # Card 1: Data Selection (Shear Plate)
                html.Div([
                    html.H3("Data Selection", style=section_title_style),
                    
                    html.Label("Neurons", style=label_style),
                    dcc.Dropdown(
                        id='neuron-dropdown',
                        options=[{'label': n, 'value': n} for n in all_neurons],
                        value=[],
                        multi=True,
                        placeholder="Pick neurons to add...",
                        style={'marginBottom': '10px'}
                    ),
                    html.Div([
                        html.Button('All', id='btn-neurons-all', style=btn_secondary_small_style),
                        html.Button('None', id='btn-neurons-none', style=btn_secondary_small_style),
                    ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '6px'}),
                    dcc.Textarea(
                        id='neuron-textarea',
                        value=', '.join(all_neurons[:5]) if len(all_neurons) > 5 else ', '.join(all_neurons),
                        style={**custom_input_style, 'height': '90px', 'resize': 'vertical'},
                    ),
                    
                    html.Label("Stimuli", style={**label_style, 'marginTop': '12px'}),
                    dcc.Dropdown(
                        id='stimuli-dropdown',
                        options=[{'label': get_stimulus_label(s, odor_information), 'value': s} for s in all_stimuli],
                        value=[],
                        multi=True,
                        placeholder="Pick stimuli to add...",
                        style={'marginBottom': '10px'}
                    ),
                    html.Div([
                        html.Button('All', id='btn-stimuli-all', style=btn_secondary_small_style),
                        html.Button('None', id='btn-stimuli-none', style=btn_secondary_small_style),
                        html.Button('Toggle Labels', id='btn-stimuli-toggle', style=btn_secondary_small_style),
                    ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '6px'}),
                    dcc.Textarea(
                        id='stimuli-textarea',
                        value=', '.join(all_stimuli),
                        style={**custom_input_style, 'height': '90px', 'resize': 'vertical'},
                    ),
                    dcc.Store(id='stimuli-format-state', data='codes'),
                ], style=glass_card_style),
                
                # Card 2: Display Settings
                html.Div([
                    html.H3("Display Settings", style=section_title_style),
                    
                    html.Label("Visualization Mode", style=label_style),
                    dcc.RadioItems(
                        id='display-type-selector',
                        options=[
                            {'label': ' Individual Trials', 'value': 'individual'},
                            {'label': ' Mean ± SEM', 'value': 'mean_sem'}
                        ],
                        value='mean_sem',
                        labelStyle={'display': 'flex', 'alignItems': 'center', 'margin': '10px 0', 'color': '#334155', 'fontWeight': '500'},
                        inputStyle={'marginRight': '10px', 'transform': 'scale(1.2)'}
                    ),
                    
                    html.Div(style={'height': '16px'}),
                    
                    html.Label("Grouping Options", style=label_style),
                    dcc.Checklist(
                        id='combine-options',
                        options=[
                            {'label': ' Combine L/R Neurons', 'value': 'combine_neurons'},
                            {'label': ' Compare by Date', 'value': 'show_date_difference'},
                            {'label': ' Cluster Mode', 'value': 'cluster_mode'}
                        ],
                        value=[],
                        labelStyle={'display': 'flex', 'alignItems': 'center', 'margin': '10px 0', 'color': '#334155', 'fontWeight': '500'},
                        inputStyle={'marginRight': '10px', 'transform': 'scale(1.2)'}
                    ),
                ], style=glass_card_style),
                
                # Card 3: Actions
                html.Div([
                    html.H3("Actions", style=section_title_style),
                    
                    html.Button('Update Plot', id='update-plot-button', style=btn_primary_style),
                               
                    html.Label("Export HTML Path", style=label_style),
                    dcc.Input(
                        id='save-path-input',
                        type='text',
                        value='my_neuron_plot.html',
                        style=custom_input_style
                    ),
                    html.Button('Export Plot', id='save-html-button', style=btn_secondary_style),
                    html.Div(id='save-feedback', style={'marginTop': '12px', 'fontSize': '14px', 'fontWeight': '500', 'color': '#059669', 'minHeight': '20px'})
                ], style=glass_card_style),
                
            ], style={'width': '380px', 'minWidth': '380px', 'marginRight': '24px'}),
            
            # Main Plot Area
            html.Div([
                dcc.Loading(
                    id="loading-plot",
                    type="circle",
                    color="#4f46e5",
                    children=[
                        html.Div([
                            dcc.Graph(id='response-plot', style={'height': '100%', 'width': '100%'}, config={'displayModeBar': True, 'scrollZoom': False}),
                        ], id='plot-container', style={'height': '800px', 'width': '100%', 'transition': 'height 0.3s ease'})
                    ]
                )
            ], style={**glass_card_style, 'flex': '1', 'minWidth': '0', 'padding': '16px', 'display': 'flex', 'flexDirection': 'column'})
            
        ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start', 'flexWrap': 'wrap'})
        
    ], style={'padding': '32px', 'backgroundColor': '#f8fafc', 'minHeight': '100vh', 'fontFamily': "'Inter', sans-serif", 'backgroundImage': 'radial-gradient(circle at top right, #e0e7ff, transparent 30%), radial-gradient(circle at bottom left, #e2e8f0, transparent 40%)'})
    
    def generate_response_figure(selected_neurons_text, selected_stimuli_text, display_type, combine_options):
        """
        Generates the plotly figure using DataFrame operations.
        """
        def parse_text(text):
            if not text: return []
            return [x.strip() for x in text.replace('\n', ',').split(',') if x.strip()]
            
        parsed_neurons = parse_text(selected_neurons_text)
        
        # map stimuli labels back to codes
        label_to_code = {get_stimulus_label(s, odor_information): s for s in all_stimuli}
        selected_stimuli = []
        cluster_stimuli_map = {} # cluster_name -> [stimulus_codes]
        cluster_names_ordered = []
        
        cluster_mode = 'cluster_mode' in combine_options
        if cluster_mode:
            import re
            # Find all Cluster definitions: Cluster X: stim1, stim2...
            matches = re.findall(r"(Cluster\s+\d+):\s+([\w\s,]+)", selected_stimuli_text)
            for cluster_name, stim_str in matches:
                cluster_names_ordered.append(cluster_name)
                cluster_stimuli_map[cluster_name] = []
                stim_items = parse_text(stim_str)
                for item in stim_items:
                    code = item
                    if item in label_to_code:
                        code = label_to_code[item]
                    
                    if code in all_stimuli:
                        cluster_stimuli_map[cluster_name].append(code)
                        if code not in selected_stimuli:
                            selected_stimuli.append(code)
        else:
            parsed_stimuli = parse_text(selected_stimuli_text)
            for s in parsed_stimuli:
                if s in all_stimuli:
                    selected_stimuli.append(s)
                elif s in label_to_code:
                    selected_stimuli.append(label_to_code[s])
                    
        selected_neurons = []
        # Support combined neuron names in user text
        mapping = create_neuron_mapping(df_all['neuron'].unique())
        # Add original valid neurons
        valid_neurons = set(df_all['neuron'].unique())
        valid_combined = set(mapping.values())
        for n in parsed_neurons:
            if n in valid_neurons or n in valid_combined:
                selected_neurons.append(n)
                
        if not selected_neurons or not selected_stimuli:
            return go.Figure(), {'height': '800px', 'width': '100%'}
        
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
        for stim in selected_stimuli:
            grouped_stimuli[stim] = [stim]
            compound_to_stimuli[stim] = [stim]
        dff['stimulus_group'] = dff['stimulus']
        
        plot_groups = [g for g in grouped_stimuli.keys() if g in dff['stimulus_group'].unique()]

        # 4. Calculate plot dimensions
        fixed_height_per_neuron = 150
        total_height = max(500, fixed_height_per_neuron * len(plot_neurons) + 100)
        container_height = total_height + 50
        
        # Determine total_width dynamically
        # Let's not restrict the width to a hardcoded minimum, but rather rely on Plotly's autosize
        # We can still provide a calculation if needed, but we'll remove it from update_layout
        
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
                
                highlight_color = stimulus_color_map.get(group_key, 'gray')
                
                fig.add_shape(
                    type="rect", x0=start_time_rel, x1=end_time_rel,
                    y0=y_range[0], y1=y_range[1],
                    fillcolor=highlight_color, opacity=0.15, layer="below", line_width=0,
                    row=row_idx, col=col_idx
                )

                if display_type == 'individual':
                    # Group by trial AND original neuron to prevent connecting separate neurons (e.g. L/R)
                    trial_cols = ['worm_key', 'segment_index', 'date', 'neuron']
                    
                    for trial_info, trial_data in cell_df.groupby(trial_cols):
                        # 解包增加一项 original_neuron_name
                        worm, seg, date, original_neuron_name = trial_info
                        
                        trace_legendgroup = f"date_{date}" if show_date_difference else group_key
                        
                        # 在 hovertemplate 中增加原始神经元名称显示，方便调试区分
                        fig.add_trace(go.Scatter(
                            x=trial_data['rel_time'], y=trial_data['delta_F_over_F0'],
                            mode='lines', line=dict(width=1, color=highlight_color),
                            opacity=0.4, showlegend=False, legendgroup=trace_legendgroup,
                            hovertemplate=(f"{worm}_{seg}_{date}<br>"
                                           f"Original: {original_neuron_name}<br>"  # 显示原始名称 (如 ADLL)
                                           f"x: %{{x}}<br>y: %{{y:.3f}}<br>"
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
                                 showgrid=False, showticklabels=(col_idx == 1), fixedrange=True)
                fig.update_xaxes(showgrid=False, row=row_idx, col=col_idx,
                                 showticklabels=False, fixedrange=True)
                if row_idx == len(plot_neurons):
                    fig.update_xaxes(title_text="", row=row_idx, col=col_idx, fixedrange=True)

        # 9. Global legend adjustments
        if show_date_difference:
            for date in all_dates:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color='gray', width=2, dash=date_dash_map.get(date, 'solid')),
                    name=f"Date: {date}", legendgroup=f"date_{date}", showlegend=True
                ), row=1, col=1)
        
        fig.update_layout(
            height=total_height, # Keep height explicitly so it scrolls vertically
            autosize=True,       # Self-adapting width
            margin=dict(l=80, r=20, t=80, b=60),
            template="plotly_white", hovermode="closest",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="'Inter', sans-serif", color="#334155", size=13),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, 
                title="Stimulus / Date", 
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=12)
            )
        )
        return fig, {'height': f'{container_height}px', 'width': '100%'}

    @app.callback(
        Output('sidebar-container', 'style'),
        Input('toggle-sidebar-btn', 'n_clicks'),
        State('sidebar-container', 'style'),
        prevent_initial_call=True
    )
    def toggle_sidebar(n_clicks, current_style):
        if not current_style:
            current_style = {'width': '380px', 'minWidth': '380px', 'marginRight': '24px'}
            
        if current_style.get('display') == 'none':
            current_style['display'] = 'block'
        else:
            current_style['display'] = 'none'
        return current_style

    @app.callback(
        [Output('neuron-textarea', 'value'),
         Output('neuron-dropdown', 'value')],
        [Input('btn-neurons-all', 'n_clicks'),
         Input('btn-neurons-none', 'n_clicks'),
         Input('neuron-dropdown', 'value')],
        State('neuron-textarea', 'value'),
        prevent_initial_call=True
    )
    def update_neurons_textarea(all_clicks, none_clicks, dropdown_val, current_val):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_val, []
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'btn-neurons-all':
            return ', '.join(all_neurons), []
        elif button_id == 'btn-neurons-none':
            return '', []
        elif button_id == 'neuron-dropdown' and dropdown_val:
            # Append dropdown selections to current textarea
            existing = [x.strip() for x in current_val.replace('\n', ',').split(',') if x.strip()] if current_val else []
            for v in dropdown_val:
                if v not in existing:
                    existing.append(v)
            return ', '.join(existing), []
        return current_val, []

    @app.callback(
        [Output('stimuli-textarea', 'value'), Output('stimuli-format-state', 'data'),
         Output('stimuli-dropdown', 'value')],
        [Input('btn-stimuli-all', 'n_clicks'),
         Input('btn-stimuli-none', 'n_clicks'),
         Input('btn-stimuli-toggle', 'n_clicks'),
         Input('stimuli-dropdown', 'value')],
        [State('stimuli-textarea', 'value'),
         State('stimuli-format-state', 'data')],
        prevent_initial_call=True
    )
    def update_stimuli_textarea(all_clicks, none_clicks, toggle_clicks, dropdown_val, current_val, format_state):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_val, format_state, []
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'btn-stimuli-all':
            items = all_stimuli
            if format_state == 'labels':
                items = [get_stimulus_label(s, odor_information) for s in items]
            return ', '.join(items), format_state, []
        elif button_id == 'btn-stimuli-none':
            return '', format_state, []
        elif button_id == 'stimuli-dropdown' and dropdown_val:
            # Append dropdown selections to current textarea
            existing = [x.strip() for x in current_val.replace('\n', ',').split(',') if x.strip()] if current_val else []
            for v in dropdown_val:
                display_v = get_stimulus_label(v, odor_information) if format_state == 'labels' else v
                if display_v not in existing:
                    existing.append(display_v)
            return ', '.join(existing), format_state, []
        elif button_id == 'btn-stimuli-toggle':
            items = [x.strip() for x in current_val.replace('\n', ',').split(',') if x.strip()]
            new_state = 'labels' if format_state == 'codes' else 'codes'
            new_items = []
            
            label_to_code = {get_stimulus_label(s, odor_information): s for s in all_stimuli}
            
            for item in items:
                if new_state == 'labels':
                    if item in all_stimuli:
                        new_items.append(get_stimulus_label(item, odor_information))
                    else:
                        new_items.append(item)
                else:
                    if item in label_to_code:
                        new_items.append(label_to_code[item])
                    else:
                        new_items.append(item)
            return ', '.join(new_items), new_state, []
            
        return current_val, format_state, []

    @app.callback(
        [Output('response-plot', 'figure'),
         Output('plot-container', 'style')],
        Input('update-plot-button', 'n_clicks'),
        [State('neuron-textarea', 'value'),
         State('stimuli-textarea', 'value'),
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
        [State('save-path-input', 'value'),
         State('neuron-textarea', 'value'),
         State('stimuli-textarea', 'value'),
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
            fig, _ = generate_response_figure(
                selected_neurons, selected_stimuli, display_type, combine_options
            )
            fig.layout.width = None
            fig.layout.height = None
            # Update layout for better standalone viewing
            fig.update_layout(
                autosize=True, 
                template='plotly_white',
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            fig.write_html(
                save_path,
                include_plotlyjs='cdn',
                full_html=True,
                config={'responsive': True, 'scrollZoom': True}
            )
            
            return f"Plot successfully saved to {save_path}"
            
        except Exception as e:
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