import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.projections.polar import PolarAxes
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import plotly.graph_objects as go

#%%
def compare_compounds_and_dilutions(neuron_segments_dict, odor_information, output_folder="./visualization_output",  if_combine=False, interactive=False):
    """
    Create radar charts to compare:
    1. Responses to different compounds
    2. Responses to different dilutions of the same compound
    
    Parameters:
    neuron_segments_dict (dict): Dictionary with structure {neuron: {stimulus: [segments,...]}}
    odor_information (dict): Dictionary mapping stimulus codes to descriptions
    output_folder (str): Folder path to save visualization outputs
    if_combine (bool): If True, combine left and right neuron pairs by averaging their responses
    interactive (bool): If True, also create interactive HTML plots using Plotly

    Returns:
    tuple: (radar_fig_compounds, radar_figs_dilutions)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    if interactive:
        # Create a subfolder for interactive plots
        interactive_folder = os.path.join(output_folder, "interactive")
        os.makedirs(interactive_folder, exist_ok=True)
    # Step 1: Extract compound types from odor_information
    compound_types = {}
    for stim_code, description in odor_information.items():
        compound_name = description.split()[0]  # Get first word (compound name)
        compound_code = stim_code.split('_')[0]  # Get compound code (e.g., c1, c2)
        
        if compound_code not in compound_types:
            compound_types[compound_code] = compound_name
    
    # Step 2: Calculate maximum responses for each neuron to each stimulus
    response_data = extract_peak_responses(neuron_segments_dict)

    # create a radar response data using the absolute value of the response
    response_data_abs = response_data.abs()

    # If if_combine is True, combine left and right neuron pairs
    if if_combine:
        response_data_abs = combine_lr_neurons(response_data_abs)
        print(f"After combining L/R neurons: {len(response_data_abs)} neurons")

    # Step 3: Create dendrogram to visualize neuron clustering
    visualize_neuron_clustering(
        response_data_abs,
        os.path.join(output_folder, 'neuron_clustering_dendrogram.png')
    )

    # Step 4: Create compound comparison radar chart
    radar_fig_compounds = compare_compounds(
        response_data_abs, 
        compound_types, 
        os.path.join(output_folder, 'compound_comparison.svg')
    )
    
    # Step 4: Create dilution comparison radar charts (one for each compound)
    radar_figs_dilutions = {}
    interactive_figs = {'compounds': None, 'dilutions': {}}
    for compound_code, compound_name in compound_types.items():
        radar_figs_dilutions[compound_code] = compare_dilutions(
            response_data_abs, 
            compound_code,
            compound_name,
            odor_information,
            os.path.join(output_folder, f'{compound_name}_dilution_comparison.svg')
        )
    # Create interactive plots if requested
    if interactive:
        # For compound comparison
        compound_responses = pd.DataFrame(index=response_data_abs.index)
        
        for compound_code, compound_name in compound_types.items():
            compound_columns = [col for col in response_data_abs.columns if col.startswith(compound_code)]
            
            if compound_columns:
                compound_responses[compound_name] = response_data_abs[compound_columns].mean(axis=1)
        
        compound_responses = compound_responses.dropna(how='any')
        if not compound_responses.empty:
            interactive_figs['compounds'] = create_interactive_radar_chart(
                compound_responses,
                title="Neuronal Responses to Different Compounds",
                output_path=os.path.join(interactive_folder, 'compound_comparison_interactive.html'),
                cluster_neurons=True
            )
        
        # For dilution comparisons
        for compound_code, compound_name in compound_types.items():
            dilution_columns = [col for col in response_data_abs.columns if col.startswith(compound_code)]
            
            if not dilution_columns:
                continue
            
            dilution_responses = pd.DataFrame(index=response_data_abs.index)
            
            for col in dilution_columns:
                if col in odor_information:
                    description = odor_information[col]
                    concentration = description.split(' ', 1)[1]
                    dilution_responses[concentration] = response_data_abs[col]
            
            dilution_responses = dilution_responses.dropna(how='any')
            if not dilution_responses.empty:
                interactive_figs['dilutions'][compound_name] = create_interactive_radar_chart(
                    dilution_responses,
                    title=f"Neuronal Responses to {compound_name} Dilutions",
                    output_path=os.path.join(interactive_folder, f'{compound_name}_dilution_comparison_interactive.html')
                )

    return radar_fig_compounds, radar_figs_dilutions, interactive_figs if interactive else None

def extract_peak_responses(neuron_segments_dict):
    """
    Extract the peak responses during stimulus periods for each neuron and stimulus,
    preserving the sign (positive or negative) of the response.
    """
    # Initialize dictionary to store max responses
    neurons = list(neuron_segments_dict.keys())
    stimulus_types = set()
    
    # Find all unique stimulus types across all neurons
    for neuron, stim_data in neuron_segments_dict.items():
        stimulus_types.update(stim_data.keys())
    
    stimulus_types = sorted(list(stimulus_types))
    
    # Create a pandas DataFrame to store the max responses
    response_df = pd.DataFrame(index=neurons, columns=stimulus_types,dtype=np.float32)
    
    # For each neuron and stimulus type, calculate the maximum response
    for neuron, stim_data in neuron_segments_dict.items():
        for stim_type, segments in stim_data.items():
            # max_responses = []
            peak_responses = []
            
            for segment in segments:
                # Get calcium signal time series
                time_series = segment['deltaFoverF_0']
                
                # Define stimulus period (starts at index 30 and lasts 50 frames)
                stim_start = 30
                stim_end = stim_start + 50
                buffer_end = stim_end + 120
                
                # Extract the stimulus period
                stim_period = time_series[stim_start:stim_end]
                cal_period = time_series[stim_start:buffer_end]
                
                # # Calculate maximum absolute response
                # max_abs_response = np.max(np.abs(stim_period))
                # max_responses.append(max_abs_response)

                # Calculate maximum response (preserving sign)
                max_postive_response = np.max(cal_period)
                max_negative_response = np.min(cal_period)

                if abs(max_postive_response) > abs(max_negative_response):
                    peak_response = max_postive_response
                else:
                    peak_response = max_negative_response
                
                peak_responses.append(peak_response)
            # Use the average of peak responses across segments
            if peak_responses:
                response_df.loc[neuron, stim_type] = np.mean(peak_responses)
            else:
                response_df.loc[neuron, stim_type] = 0
    
    # Fill NaN values with 0
    response_df.fillna(0, inplace=True)
    
    return response_df

def compare_compounds(response_df, compound_types, output_path):
    """
    Create a radar chart comparing different compounds regardless of dilution.
    Each axis is a neuron, each line is a compound type.
    """
    # Aggregate responses by compound type
    compound_responses = pd.DataFrame(index=response_df.index)
    
    for compound_code, compound_name in compound_types.items():
        # Find all columns that start with this compound code
        compound_columns = [col for col in response_df.columns if col.startswith(compound_code)]
        
        if compound_columns:
            # Calculate mean response across all dilutions of this compound
            compound_responses[compound_name] = response_df[compound_columns].mean(axis=1)
    compound_responses = compound_responses.dropna(how='any')
    if compound_responses.empty:
        print("No common neurons across all compounds")
        return None
    
    # Create radar chart with neuron clustering enabled
    fig = create_radar_chart(
        compound_responses, 
        title="Neuronal Responses to Different Compounds",
        output_path=output_path,
        cluster_neurons=True  # Enable clustering
    )
    
    return fig

def compare_dilutions(response_df, compound_code, compound_name, odor_information, output_path):
    """
    Create a radar chart comparing different dilutions of the same compound.
    Each axis is a neuron, each line is a dilution level.
    """
    # Find all stimuli for this compound
    dilution_columns = [col for col in response_df.columns if col.startswith(compound_code)]
    
    if not dilution_columns:
        print(f"No data found for compound {compound_name}")
        return None
    
    # Create a new DataFrame for the dilutions
    dilution_responses = pd.DataFrame(index=response_df.index)
    
    # Add each dilution as a column with the actual concentration in the label
    for col in dilution_columns:
        # Get the dilution description from odor_information
        if col in odor_information:
            description = odor_information[col]
            # Extract just the concentration part
            concentration = description.split(' ', 1)[1]
            dilution_responses[concentration] = response_df[col]
    
    dilution_responses = dilution_responses.dropna(how='any')
    if dilution_responses.empty:
        print(f"No common neurons for all dilutions of {compound_name}")
        return None
    
    # Create radar chart
    fig = create_radar_chart(
        dilution_responses, 
        title=f"Neuronal Responses to {compound_name} Dilutions",
        output_path=output_path
    )
    
    return fig

def create_radar_chart(data_df, title, output_path, cluster_neurons=True):
    """
    Create a radar chart from the provided DataFrame.
    Each row is a neuron (axis), each column is a category (line).
    """
    # Get neurons and categories
    neurons = data_df.index.tolist()
    categories = data_df.columns.tolist()

    # Cluster neurons by response pattern similarity
    if cluster_neurons and len(neurons) > 2:
        try:
            # Handle potential NaN or Inf values
            clean_data = data_df.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Check if we have enough variation to compute correlation
            # If any row has zero variance, clustering by correlation won't work
            row_stds = np.std(clean_data.values, axis=1)
            if np.all(row_stds > 0):
                # Compute distance matrix between neuron response profiles
                distance_matrix = pdist(clean_data.values, 'correlation')
                
                if np.all(np.isfinite(distance_matrix)):
                    Z = hierarchy.linkage(distance_matrix, 'average')
                    
                    # Get the optimized leaf ordering
                    ordered_indices = hierarchy.leaves_list(
                        hierarchy.optimal_leaf_ordering(Z, distance_matrix)
                    )
                    
                    # Reorder neurons based on clustering
                    neurons = [neurons[i] for i in ordered_indices]
                    data_df = data_df.iloc[ordered_indices]
                else:
                    print(f"Warning: Non-finite values in distance matrix. Skipping clustering for {title}")
            else:
                print(f"Warning: Zero variance in some rows. Skipping clustering for {title}")
        except Exception as e:
            print(f"Clustering error in {title}: {str(e)}. Continuing without clustering.")

    # Number of variables (neurons)
    N = len(neurons)
    
    # Create angles for each neuron
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Complete the loop
    theta = np.append(theta, theta[0])
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Find the maximum value to set the scale
    max_value = data_df.max().max()
    
    # Draw circular grid lines
    grid_levels = 5
    grid_vals = np.linspace(0, max_value*1.1, grid_levels)
    
    # Set radial grid
    ax.set_ylim(0, max_value*1.1)
    ax.set_rticks(grid_vals)
    
    # Set the angle of the first axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)  # Clockwise
    
    # Draw the axes and add neuron labels
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(neurons, fontsize=9)
    
    # Use a nice color palette
    colors = plt.cm.Dark2(np.linspace(0, 1, len(categories)))
    
    # Plot each category and add to legend
    legend_elements = []
    for i, category in enumerate(categories):
        values = data_df[category].values.tolist()
        # Repeat the first value to close the loop
        values = values + [values[0]]
        
        # Plot the category line
        ax.plot(theta, values, 'o-', linewidth=2, color=colors[i], markersize=6)
        ax.fill(theta, values, color=colors[i], alpha=0.25)
        
        # Add to legend
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, marker='o', 
                                     markersize=6, label=category))
    
    # Add legend with custom position
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.2, 0.2),
              fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    
    # Add title
    plt.title(title, size=16, y=1.1, fontweight='bold')
    
    # # Add a subtitle explaining the metric
    # plt.figtext(0.5, 0.01, 'Values represent maximum absolute response during stimulus period',
    #            ha='center', fontsize=10, style='italic')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    return fig

def create_interactive_radar_chart(data_df, title, output_path, cluster_neurons=True):
    """
    Create an interactive radar chart using Plotly.
    Each axis is a neuron, each line is a compound/dilution.
    
    Parameters:
    data_df (DataFrame): DataFrame with neurons as index and categories as columns
    title (str): Chart title
    output_path (str): Path to save the HTML file
    cluster_neurons (bool): Whether to cluster neurons by response pattern similarity
    
    Returns:
    plotly.graph_objects.Figure: The created figure
    """
    # Get neurons and categories
    neurons = data_df.index.tolist()
    categories = data_df.columns.tolist()

    # Cluster neurons by response pattern similarity
    if cluster_neurons and len(neurons) > 2:
        try:
            # Handle potential NaN or Inf values
            clean_data = data_df.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Check if we have enough variation to compute correlation
            row_stds = np.std(clean_data.values, axis=1)
            if np.all(row_stds > 0):
                # Compute distance matrix between neuron response profiles
                distance_matrix = pdist(clean_data.values, 'correlation')
                
                if np.all(np.isfinite(distance_matrix)):
                    Z = hierarchy.linkage(distance_matrix, 'average')
                    
                    # Get the optimized leaf ordering
                    ordered_indices = hierarchy.leaves_list(
                        hierarchy.optimal_leaf_ordering(Z, distance_matrix)
                    )
                    
                    # Reorder neurons based on clustering
                    neurons = [neurons[i] for i in ordered_indices]
                    data_df = data_df.iloc[ordered_indices]
                else:
                    print(f"Warning: Non-finite values in distance matrix. Skipping clustering for {title}")
            else:
                print(f"Warning: Zero variance in some rows. Skipping clustering for {title}")
        except Exception as e:
            print(f"Clustering error in {title}: {str(e)}. Continuing without clustering.")
    
    # Create figure
    fig = go.Figure()
    
    # Find the maximum value to set the scale
    max_value = data_df.max().max()
    
    # Plot each category
    for i, category in enumerate(categories):
        values = data_df[category].values.tolist()
        
        # We need to close the loop for the radar chart
        theta_values = neurons + [neurons[0]]
        r_values = values + [values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            mode='lines+markers',
            name=category,
            fill='toself',
            opacity=0.7
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value * 1.1]
            )
        ),
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        height=800,
        width=800
    )
    
    # Add annotation explaining the metric
    fig.add_annotation(
        text='Values represent peak response during stimulus period',
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10, color="gray")
    )
    
    # Save as HTML
    fig.write_html(output_path)
    
    return fig

def visualize_neuron_clustering(data_df, output_path):
    """Create a dendrogram showing neuron clusters based on response patterns."""
    plt.figure(figsize=(10, 6))
    
    # Compute linkage matrix
    Z = hierarchy.linkage(pdist(data_df.values, 'correlation'), 'average')
    
    # Plot dendrogram
    dn = hierarchy.dendrogram(
        Z,
        labels=data_df.index,
        orientation='right',
        leaf_font_size=10
    )
    
    plt.title('Neuron Response Similarity')
    plt.xlabel('Distance (correlation)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return Z

def create_heatmap_comparison(response_df, compound_types, odor_information, output_folder, if_combine=False):
    """
    Create heatmap visualizations for comparing compounds and dilutions
    """
    os.makedirs(output_folder, exist_ok=True)

    # If if_combine is True, combine left and right neuron pairs
    if if_combine:
        response_df = combine_lr_neurons(response_df)
        print(f"After combining L/R neurons: {len(response_df)} neurons")
    # Create a custom diverging colormap: pink -> white -> green
    pink = '#FF9AA2'  # Light pink
    white = '#FFFFFF' # White
    green = '#006400' # Dark green
    cmap = mcolors.LinearSegmentedColormap.from_list('pink_white_green', 
                                                     [pink, white, green], N=256)
    
    # Compound comparison heatmap
    compound_responses = pd.DataFrame(index=response_df.index)
    
    for compound_code, compound_name in compound_types.items():
        # Find all columns that start with this compound code
        compound_columns = [col for col in response_df.columns if col.startswith(compound_code)]
        
        if compound_columns:
            # Calculate mean response across all dilutions of this compound
            compound_responses[compound_name] = response_df[compound_columns].mean(axis=1)

    # Determine the color scale limits based on data
    vmax = max(abs(compound_responses.max().max()), abs(compound_responses.min().min()))
    vmin = -vmax  # Symmetric limits for diverging colormap

    # Create compound heatmap
    plt.figure(figsize=(10, 12))
    sns.heatmap(compound_responses, cmap=cmap, annot=False,
                linewidths=0.5, center=0, vmin=vmin, vmax= vmax, cbar_kws={'label': 'Peak Response'})
    plt.title('Neuronal Responses', size=15)
    plt.xlabel('Compound', size=12)
    plt.ylabel('Neuron', size=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'compound_heatmap.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create dilution heatmaps for each compound
    for compound_code, compound_name in compound_types.items():
        # Find all stimuli for this compound
        dilution_columns = [col for col in response_df.columns if col.startswith(compound_code)]
        
        if not dilution_columns:
            continue
        
        # Create a new DataFrame for the dilutions
        dilution_responses = pd.DataFrame(index=response_df.index)
        
        # Add each dilution as a column with the actual concentration in the label
        for col in sorted(dilution_columns):
            # Get the dilution description from odor_information
            if col in odor_information:
                description = odor_information[col]
                # Extract just the concentration part
                concentration = description.split(' ', 1)[1]
                dilution_responses[concentration] = response_df[col]
        
        if dilution_responses.empty:
            continue
            

        # Determine color scale limits for this specific dilution set
        vmax_dil = max(abs(dilution_responses.max().max()), abs(dilution_responses.min().min()))
        vmin_dil = -vmax_dil
        
        # Create dilution heatmap
        plt.figure(figsize=(len(dilution_responses.columns)*1.5, 12))
        # sns.heatmap(dilution_responses, cmap='viridis', annot=False, 
        #             linewidths=0.5, cbar_kws={'label': 'Response'})
        sns.heatmap(dilution_responses, cmap=cmap, annot=False,
                    linewidths=0.5, center=0, vmin=vmin_dil, vmax=vmax_dil,
                    cbar_kws={'label': 'Peak Response'})
        plt.title(f'Neuronal Responses to {compound_name} Dilutions', size=15)
        plt.xlabel('Concentration', size=12)
        plt.ylabel('Neuron', size=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{compound_name}_dilution_heatmap.svg'), 
                    dpi=300, bbox_inches='tight')
        
        plt.close()
    
    return compound_responses

def combine_lr_neurons(df):
    """
    Combine left and right neuron pairs by averaging their responses.
    
    Parameters:
    df (DataFrame): DataFrame with neurons as index
    
    Returns:
    DataFrame: New DataFrame with combined L/R neuron pairs
    """
    # Group neurons by their base name (without L/R suffix)
    neuron_groups = {}
    
    for neuron in df.index:
        # Check if neuron name ends with 'L' or 'R'
        if neuron.endswith('L') or neuron.endswith('R'):
            base_name = neuron[:-1]  # Remove the last character (L or R)
            if base_name not in neuron_groups:
                neuron_groups[base_name] = []
            neuron_groups[base_name].append(neuron)
    
    # Create a new DataFrame for the combined neurons
    combined_df = pd.DataFrame(columns=df.columns)
    
    # Add neurons that can be paired
    for base_name, neurons in neuron_groups.items():
        if len(neurons) == 2:  # We have both L and R
            # Make sure they're actually an L/R pair
            if neurons[0].endswith('L') and neurons[1].endswith('R') or \
               neurons[0].endswith('R') and neurons[1].endswith('L'):
                # Average the responses
                combined_df.loc[base_name] = df.loc[neurons].mean()
        else:
            # Only one side exists, keep it with original name
            combined_df.loc[neurons[0]] = df.loc[neurons[0]]
    
    # Add neurons that don't follow the L/R pattern
    for neuron in df.index:
        if not (neuron.endswith('L') or neuron.endswith('R')):
            combined_df.loc[neuron] = df.loc[neuron]
    
    return combined_df

def heatmap_plot(neuron_segments_dict, odor_information, output_folder, if_combine=False):
    """
    Create heatmap visualizations for comparing compounds and dilutions
    """
    compound_types = {}
    for stim_code, description in odor_information.items():
        compound_name = description.split()[0]  # Get first word (compound name)
        compound_code = stim_code.split('_')[0]  # Get compound code (e.g., c1, c2)
        
        if compound_code not in compound_types:
            compound_types[compound_code] = compound_name
    
    create_heatmap_comparison(
    extract_peak_responses(neuron_segments_dict),
        compound_types,
        odor_information,
        output_folder,
        if_combine=if_combine
    )

    return None

#%%
if __name__ == "__main__":
    # Example usage
    neuron_segments_dict = {
        'Neuron1': {
            'c1_1': [{'deltaFoverF_0': np.random.rand(100)}],
            'c1_2': [{'deltaFoverF_0': np.random.rand(100)}]
        },
        'Neuron2': {
            'c1_1': [{'deltaFoverF_0': np.random.rand(100)}],
            'c1_3': [{'deltaFoverF_0': np.random.rand(100)}]
        }
    }
    
    odor_information = {
        'c1_1': 'Compound A 10uM',
        'c1_2': 'Compound A 20uM',
        'c1_3': 'Compound B 10uM'
    }
    
    output_folder = "./visualization_output"
    # compare_compounds_and_dilutions(neuron_segments_dict, odor_information, output_folder, if_combine=True, interactive=True)
    heatmap_plot(neuron_segments_dict, odor_information, output_folder, if_combine=True)