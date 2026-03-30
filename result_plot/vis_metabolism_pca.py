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
    
    # Extract variance explained
    var_exp = pca_model.explained_variance_ratio_
    components = [f"PC{i+1}" for i in range(len(var_exp))]
    
    # Create Variance Bar Chart
    var_fig = go.Figure(data=[
        go.Bar(
            x=components, 
            y=var_exp,
            text=[f"{v:.1%}" for v in var_exp],
            textposition='auto'
        )
    ])
    var_fig.update_layout(
        title="PCA Variance Explained",
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained Ratio",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    
    # Base 3D Scatter Plot (before callbacks)
    if 'PC3' not in pca_df.columns:
        pca_df['PC3'] = 0.0
        
    scatter_fig = px.scatter_3d(
        pca_df,
        x='PC1', y='PC2', z='PC3',
        color='Cluster',
        text='Bacteria',
        title="3D Metabolic Space",
        hover_data=['Cluster']
    )
    scatter_fig.update_traces(marker=dict(size=5, opacity=1.0))
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
            color='Cluster',
            text='Bacteria',
            title="3D Metabolic Space",
            hover_data=['Cluster']
        )
        
        if search_value:
            search_lower = search_value.lower()
            for trace in fig.data:
                if trace.text is not None:
                    # trace.text holds the Bacteria label list
                    mask = np.array([search_lower in str(t).lower() for t in trace.text])
                    trace.marker.opacity = np.where(mask, 1.0, 0.1)
                    trace.marker.size = np.where(mask, 6, 3)
        else:
            for trace in fig.data:
                trace.marker.opacity = 1.0
                trace.marker.size = 5
                
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=800)
        return fig
        
    return app
