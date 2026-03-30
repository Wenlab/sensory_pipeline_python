import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_metabolism_dashboard(pca_df: pd.DataFrame, pca_model):
    """
    Create a Dash app for visualizing PCA clustering results.
    """
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Extract variance explained (max 20)
    var_exp = pca_model.explained_variance_ratio_[:20]
    components = [f"PC{i+1}" for i in range(len(var_exp))]
    cumulative_variance = np.cumsum(var_exp)
    
    # Create Variance Bar Chart with Cumulative Line
    from plotly.subplots import make_subplots
    var_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    var_fig.add_trace(
        go.Bar(
            x=components, 
            y=var_exp,
            name='Individual Variance',
            text=[f"{v:.1%}" for v in var_exp],
            textposition='auto',
            marker_color='#93c5fd'
        ),
        secondary_y=False
    )
    var_fig.add_trace(
        go.Scatter(
            x=components,
            y=cumulative_variance,
            name='Cumulative Variance',
            mode='lines+markers',
            marker_color='#1e3a8a',
            line=dict(shape='spline', smoothing=0)
        ),
        secondary_y=True
    )
    
    var_fig.update_layout(
        title="PCA Variance Explained (First 20 PCs)",
        margin=dict(l=20, r=40, t=40, b=20),
        height=350,
        showlegend=False
    )
    var_fig.update_yaxes(title_text="Individual Variance", secondary_y=False)
    var_fig.update_yaxes(title_text="Cumulative Variance", secondary_y=True, range=[0, 1.05])
    
    # Base 3D Scatter Plot (before callbacks)
    if 'PC3' not in pca_df.columns:
        pca_df['PC3'] = 0.0
        
    scatter_fig = px.scatter_3d(
        pca_df,
        x='PC1', y='PC2', z='PC3',
        color_discrete_sequence=['#4f46e5'],
        text='Bacteria',
        title="3D Metabolic Space"
    )
    scatter_fig.update_traces(
        mode='markers', 
        marker=dict(size=3.5, opacity=0.6, line=dict(width=1, color='#334155'))
    )
    scatter_fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=800)
    
    app.layout = html.Div(
        style={'display': 'flex', 'flexDirection': 'row', 'height': '100vh', 'fontFamily': 'Arial, sans-serif'},
        children=[
            # Left Sidebar
            html.Div(
                style={'width': '30%', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRight': '1px solid #dee2e6'},
                children=[
                    html.H2("Metabolism PCA Dashboard"),
                    html.Label("Search Bacteria:"),
                    dcc.Input(
                        id='search-input',
                        type='text',
                        placeholder='Type to highlight bacteria...',
                        style={'width': '100%', 'padding': '10px', 'marginBottom': '20px', 'boxSizing': 'border-box'}
                    ),
                    dcc.Graph(
                        id='variance-plot',
                        figure=var_fig
                    )
                ]
            ),
            # Right Main Area
            html.Div(
                style={'width': '70%', 'padding': '20px'},
                children=[
                    dcc.Graph(
                        id='scatter-3d-plot',
                        figure=scatter_fig,
                        style={'height': '100%'}
                    )
                ]
            )
        ]
    )
    
    @app.callback(
        dash.Output('scatter-3d-plot', 'figure'),
        [dash.Input('search-input', 'value')]
    )
    def update_scatter(search_value):
        # We process opacity mapping sequentially over px traces 
        fig = px.scatter_3d(
            pca_df,
            x='PC1', y='PC2', z='PC3',
            color_discrete_sequence=['#4f46e5'],
            text='Bacteria',
            title="3D Metabolic Space"
        )
        # Force markers mode to hide text annotations on the plot, while preserving them for hover
        fig.update_traces(mode='markers')
        
        if search_value:
            search_lower = search_value.lower()
            # Split by comma to support multiple keywords at once
            search_tokens = [tok.strip() for tok in search_lower.split(',') if tok.strip()]
            
            new_traces = []
            for trace in fig.data:
                if trace.text is not None:
                    # trace.text holds the Bacteria label list
                    mask = np.array([
                        any(token in str(t).lower() for token in search_tokens) 
                        for t in trace.text
                    ])
                    
                    if mask.all():
                        trace.marker.opacity = 0.85
                        trace.marker.size = 5
                        trace.marker.line = dict(width=1.5, color='#1e293b')
                        trace.mode = 'markers+text'
                        trace.textfont = dict(size=10, color='#1e293b')
                        trace.textposition = 'top right'
                    elif not mask.any():
                        trace.marker.opacity = 0.1
                        trace.marker.size = 2
                        trace.marker.line = dict(width=0)
                        trace.mode = 'markers'
                    else:
                        # Split trace since 3D scatter doesn't support list of opacities
                        trace_dim = go.Scatter3d(
                            x=np.array(trace.x)[~mask],
                            y=np.array(trace.y)[~mask],
                            z=np.array(trace.z)[~mask],
                            mode='markers',
                            marker=dict(
                                color=trace.marker.color,
                                size=2,
                                opacity=0.1,
                                line=dict(width=0)
                            ),
                            text=np.array(trace.text)[~mask],
                            name=trace.name,
                            legendgroup=trace.legendgroup or trace.name,
                            showlegend=False,
                            hovertemplate=trace.hovertemplate
                        )
                        if hasattr(trace, 'customdata') and trace.customdata is not None:
                            trace_dim.customdata = np.array(trace.customdata)[~mask]
                            trace.customdata = np.array(trace.customdata)[mask]
                            
                        # Keep matched points in original trace
                        trace.x = np.array(trace.x)[mask]
                        trace.y = np.array(trace.y)[mask]
                        trace.z = np.array(trace.z)[mask]
                        trace.text = np.array(trace.text)[mask]
                        trace.legendgroup = trace.legendgroup or trace.name
                        trace.marker.opacity = 0.85
                        trace.marker.size = 5
                        trace.marker.line = dict(width=1.5, color='#1e293b')
                        trace.mode = 'markers+text'
                        trace.textfont = dict(size=10, color='#1e293b')
                        trace.textposition = 'top right'
                        new_traces.append(trace_dim)
            if new_traces:
                fig.add_traces(new_traces)
        else:
            for trace in fig.data:
                trace.marker.opacity = 0.6
                trace.marker.size = 3.5
                trace.marker.line = dict(width=1, color='#334155')
                trace.mode = 'markers'
                
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=800)
        return fig
        
    return app
