import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from dPCA import dPCA
import sys
if __name__ == "__main__":
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parse_stimulus_info import group_and_sort_stimuli
class SegmentdPCA:
    def __init__(self, neuron_segments_dict, compound_info=None, compound_color_scheme=None):
        """
        Initialize the SegmentdPCA class with neuron segments data.
        """
        self.data = neuron_segments_dict
        self.dpca_results = None
        self.arranged_data = None
        self.dpca_all = None
        self.neuron_index_map = {}
        self.stimulus_index_map = {}
        self.compound_info = compound_info
        self.compound_color_scheme = compound_color_scheme

    def get_neuron_stimuli_info(self):
        """
        Get the list of neuron names and stimuli from the data.
        
        :return: Tuple of (neuron_names, stimuli)
        """
        neuron_names = list(self.data.keys())
        stimuli = set()
        for neuron in neuron_names:
            stimuli.update(self.data[neuron].keys())
        return neuron_names, list(stimuli)

    def arrange_data(self):
        """
        Arrange data for dPCA analysis.
        
        :return: Averaged Trials in format (Neurons, Stimulus, Time).
        """
        neuron_names, stimuli = self.get_neuron_stimuli_info()
        num_neurons = len(neuron_names)
        num_stimuli = len(stimuli)

        # Assuming all trials have the same length
        time_points = len(self.data[neuron_names[0]][stimuli[0]][0]['deltaFoverF_0'])

        # Create mapping from names to indices
        neuron_index_map = {name: idx for idx, name in enumerate(neuron_names)}
        stimulus_index_map = {stimulus: idx for idx, stimulus in enumerate(stimuli)}

        # Initialize the trials array
        trials = np.zeros((num_neurons, num_stimuli, time_points))

        # Loop through neurons and stimuli to fill the trials array
        for neuron_name, stimuli_data in self.data.items():
            neuron_index = neuron_index_map[neuron_name]
            for stimulus, trials_data in stimuli_data.items():
                stimulus_idx = stimulus_index_map[stimulus]

                all_trials_for_condition = []
                # Collect trials for this specific stimulus
                for trial in trials_data:
                    all_trials_for_condition.append(trial['deltaFoverF_0'])

                if all_trials_for_condition:
                    # Calculate mean across trials for this neuron-stimulus pair
                    trials[neuron_index, stimulus_idx, :] = np.mean(all_trials_for_condition, axis=0)
        self.arranged_data = trials
        self.neuron_index_map = neuron_index_map
        self.stimulus_index_map = stimulus_index_map

    def fill_blank_neuron_stimuli_pair(self):
        """
        Fill in blank neuron-stimuli pair with neuron that is symmetric in anatomy, for example, AWCL and AWCR.
        """
        # find if there is a neuron_stimuli_pair has all zeros
        for neuron_idx in range(self.arranged_data.shape[0]):
            for stimulus_idx in range(self.arranged_data.shape[1]):
                if np.all(self.arranged_data[neuron_idx, stimulus_idx, :] == 0):
                    neuron_name = list(self.neuron_index_map.keys())[neuron_idx]
                    symmetric_neuron_index = self._find_symmetric_neuron_index(neuron_name)
                    if symmetric_neuron_index is not None:
                        self.arranged_data[neuron_idx, stimulus_idx, :] = self.arranged_data[symmetric_neuron_index, stimulus_idx, :]

    def _find_symmetric_neuron_index(self, neuron_name):
        """
        Find the index of the neuron that is symmetric to the given neuron.
        
        :param neuron_name: Name of the neuron.
        :return: Index of the symmetric neuron or None if not found.
        """
        if neuron_name.endswith('L'):
            symmetric_name = neuron_name[:-1] + 'R'
        elif neuron_name.endswith('R'):
            symmetric_name = neuron_name[:-1] + 'L'
        else:
            return None
        
        return self.neuron_index_map.get(symmetric_name, None)
   
    def perform_dpca(self, n_components=10):
        """
        Perform dPCA on the arranged data.
        
        :param n_components: Number of components to compute.
        :return: dPCA results.
        """        
        dpca = dPCA.dPCA(labels='st', n_components=n_components)
        dpca.protect = ['t']
        Z = dpca.fit_transform(self.arranged_data)
        self.dpca_results = Z
        self.dpca_all = dpca
        return self.dpca_results
    
    def plot_explained_variance(self, top_n=10, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,6))
        
        # get total explained variance for sorting
        total_val = self.dpca_all.explained_variance_ratio_
        sorted_indices = np.argsort(total_val)[::-1][:top_n]

        # get explained variance for each component type
        

    def plot_dpca_results(self, components=['t', 's', 'st'], component_indices=[0], 
                          figsize=(18, 6), show_legend=True, save_path=None):
        """
        Plot the results of dPCA analysis with improved customization.
        
        :param components: List of component types to plot ['t', 's', 'st']
        :param component_indices: List of component indices to plot (e.g., [0, 1] for 1st and 2nd components)
        :param figsize: Figure size tuple
        :param show_legend: Whether to show legend with compound names
        :param save_path: Path to save the figure (optional)
        :return: Matplotlib figure object.
        """
        if self.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")
        
        Z = self.dpca_results
        time = np.arange(Z['t'].shape[2])
        
        # Create reverse mapping from index to stimulus name
        index_to_stimulus = {idx: stimulus for stimulus, idx in self.stimulus_index_map.items()}
        
        # Set up the plot
        n_plots = len(components) * len(component_indices)
        n_cols = len(components)
        n_rows = len(component_indices)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        for comp_idx in component_indices:
            for comp_type in components:
                ax = axes[plot_idx] if n_plots > 1 else axes
                
                # Plot each stimulus
                for s in range(Z[comp_type].shape[1]):
                    stimulus_name = index_to_stimulus[s]
                    
                    # Get color for this stimulus
                    color = self._get_stimulus_color(stimulus_name)
                    
                    # Get compound name for legend
                    compound_name = self._get_compound_name(stimulus_name)
                    
                    ax.plot(time, Z[comp_type][comp_idx, s], 
                           color=color, label=compound_name, linewidth=2)
                
                # Customize the plot
                ax.set_title(f'{self._get_component_title(comp_type)} (Component {comp_idx + 1})', 
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Time Points')
                ax.set_ylabel('dPC Score')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add legend if requested and there's space
                if show_legend and len(self.stimulus_index_map) <= 20:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                
                plot_idx += 1
        
        plt.tight_layout()
        
        # Save if path provided(save as png and pdf)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if save_path.endswith('.png'):
                pdf_path = save_path[:-4] + '.pdf'
                plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            
        
        plt.show()
    
    def _get_stimulus_color(self, stimulus_name):
        """
        Get color for a stimulus based on compound_color_scheme.
        
        :param stimulus_name: Name of the stimulus
        :return: Color string or default color
        """
        if self.compound_color_scheme and stimulus_name in self.compound_color_scheme:
            return self.compound_color_scheme[stimulus_name]
        else:
            # Generate a default color based on hash
            import hashlib
            hash_object = hashlib.md5(stimulus_name.encode())
            hex_dig = hash_object.hexdigest()
            return f"#{hex_dig[:6]}"
    
    def _get_compound_name(self, stimulus_name):
        """
        Get compound name for a stimulus based on compound_info.
        
        :param stimulus_name: Name of the stimulus
        :return: Compound name or stimulus name if not found
        """
        if self.compound_info and stimulus_name in self.compound_info:
            return self.compound_info[stimulus_name]
        else:
            return stimulus_name
    
    def _get_component_title(self, comp_type):
        """
        Get a descriptive title for component type.
        
        :param comp_type: Component type ('t', 's', 'st')
        :return: Descriptive title
        """
        titles = {
            't': 'Time Component',
            's': 'Stimulus Component', 
            'st': 'Stimulus-Time Interaction'
        }
        return titles.get(comp_type, comp_type)     

    def plot_concentration_trends(self, compound_base_name, component_type='s', component_idx=0, 
                                figsize=(10, 6), save_path=None):
        """
        Plot concentration trends for a specific compound across different concentrations.
        
        :param compound_base_name: Base name of compound (e.g., 'c1' for EGCG, 'c2' for Quininic acid)
        :param component_type: Type of component to plot ('t', 's', 'st')
        :param component_idx: Index of component to plot
        :param figsize: Figure size tuple
        :param save_path: Path to save the figure (optional)
        """
        if self.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")
        
        Z = self.dpca_results
        time = np.arange(Z[component_type].shape[2])
        
        # Find all stimuli that match the compound base name
        matching_stimuli = []
        for stimulus in self.stimulus_index_map.keys():
            if stimulus.startswith(compound_base_name + '_'):
                matching_stimuli.append(stimulus)
        
        if not matching_stimuli:
            print(f"No stimuli found for compound base name: {compound_base_name}")
            return
        
        # Sort by concentration (assuming format like c1_1, c1_2, etc.)
        matching_stimuli.sort(key=lambda x: int(x.split('_')[1]))
        
        plt.figure(figsize=figsize)
        
        for stimulus in matching_stimuli:
            stimulus_idx = self.stimulus_index_map[stimulus]
            color = self._get_stimulus_color(stimulus)
            compound_name = self._get_compound_name(stimulus)
            
            plt.plot(time, Z[component_type][component_idx, stimulus_idx], 
                    color=color, label=compound_name, linewidth=2, marker='o', markersize=3)
        
        plt.title(f'{self._get_component_title(component_type)} - {compound_base_name.upper()} Concentration Series', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Time Points')
        plt.ylabel('dPC Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_compound_groups(self, component_type='s', component_idx=0, figsize=(15, 10), save_path=None):
        """
        Plot compounds grouped by their base type (e.g., all EGCG together, all caffeine together).
        
        :param component_type: Type of component to plot ('t', 's', 'st')
        :param component_idx: Index of component to plot
        :param figsize: Figure size tuple
        :param save_path: Path to save the figure (optional)
        """
        if self.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")
        
        Z = self.dpca_results
        time = np.arange(Z[component_type].shape[2])
        
        # Group stimuli by compound base name
        compound_groups = {}
        for stimulus in self.stimulus_index_map.keys():
            if '_' in stimulus:
                base_name = stimulus.split('_')[0]
                if base_name not in compound_groups:
                    compound_groups[base_name] = []
                compound_groups[base_name].append(stimulus)
        
        # Create subplots for each compound group
        n_groups = len(compound_groups)
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        for base_name, stimuli in compound_groups.items():
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            # Sort stimuli by concentration
            stimuli.sort(key=lambda x: int(x.split('_')[1]) if len(x.split('_')) > 1 else 0)
            
            for stimulus in stimuli:
                stimulus_idx = self.stimulus_index_map[stimulus]
                color = self._get_stimulus_color(stimulus)
                compound_name = self._get_compound_name(stimulus)
                
                ax.plot(time, Z[component_type][component_idx, stimulus_idx], 
                       color=color, label=compound_name, linewidth=2)
            
            ax.set_title(f'{base_name.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Points')
            ax.set_ylabel('dPC Score')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'{self._get_component_title(component_type)} by Compound Groups', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_compound_summary(self):
        """
        Print a summary of compounds and their concentrations in the dataset.
        """
        print("Compound Summary:")
        print("=" * 50)
        
        # Group by compound type
        compound_groups = {}
        for stimulus in self.stimulus_index_map.keys():
            if '_' in stimulus:
                base_name = stimulus.split('_')[0]
                if base_name not in compound_groups:
                    compound_groups[base_name] = []
                compound_groups[base_name].append(stimulus)
        
        for base_name, stimuli in sorted(compound_groups.items()):
            # Get the compound name from the first stimulus
            first_stimulus = stimuli[0]
            if self.compound_info and first_stimulus in self.compound_info:
                compound_type = self.compound_info[first_stimulus].split()[0]
                print(f"\n{base_name.upper()} ({compound_type}):")
            else:
                print(f"\n{base_name.upper()}:")
            
            stimuli.sort(key=lambda x: int(x.split('_')[1]) if len(x.split('_')) > 1 else 0)
            for stimulus in stimuli:
                compound_name = self._get_compound_name(stimulus)
                print(f"  - {stimulus}: {compound_name}")
    
    def plot_dpca_grid(self, component_idx=0, vertical_overlap=0.3, fig_width=12, fig_height_per_stimulus=0.8, save_folder=None):
        """
        Plots dPCA results in a grid, with stimuli as rows and component types as columns.
        Similar layout to draw_mean_signal_cluster.

        :param component_idx: Index of the dPC to plot for each component type. Default is 0 (the first component).
        :param vertical_overlap: Overlap between rows.
        :param fig_width: Total width of the figure.
        :param fig_height_per_stimulus: Height for each stimulus row.
        :param save_folder: Folder to save the figure.
        """
        if self.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")

        Z = self.dpca_results
        components_to_plot = ['t', 's', 'st']
        time = np.arange(Z['t'].shape[2])

        # 1. Get sorted and grouped stimuli list
        if self.compound_info:
            grouped_stimuli = group_and_sort_stimuli(self.compound_info)
            stimulus_order = [s for _, stimuli in grouped_stimuli for s in stimuli]
            
            # Filter to only include stimuli present in the analysis
            available_stimuli = set(self.stimulus_index_map.keys())
            stimulus_order = [s for s in stimulus_order if s in available_stimuli]
        else:
            stimulus_order = sorted(self.stimulus_index_map.keys())
            grouped_stimuli = [(s, [s]) for s in stimulus_order]

        n_stimuli = len(stimulus_order)
        n_components = len(components_to_plot)

        # 2. Calculate global y-axis limits for dPC scores
        all_scores = []
        for comp_type in components_to_plot:
            all_scores.append(Z[comp_type][component_idx, :, :])
        
        y_min = np.min(all_scores)
        y_max = np.max(all_scores)
        padding = (y_max - y_min) * 0.1
        y_lim = (y_min - padding, y_max + padding)

        # 3. Setup Figure using add_axes for precise control
        total_height = fig_height_per_stimulus * (1 + (n_stimuli - 1) * (1 - vertical_overlap))
        fig = plt.figure(figsize=(fig_width, total_height))
        
        # Define layout: main plot area and a left margin for stimulus group labels
        main_plot_width_ratio = 0.85
        left_margin_ratio = 1.0 - main_plot_width_ratio
        
        subplot_width = main_plot_width_ratio / n_components
        subplot_height = fig_height_per_stimulus / total_height

        # 4. Loop through stimuli and components to create subplots
        for i, stimulus in enumerate(stimulus_order):
            stimulus_idx = self.stimulus_index_map[stimulus]
            
            for j, comp_type in enumerate(components_to_plot):
                left = left_margin_ratio + j * subplot_width
                bottom = (n_stimuli - i - 1) * (subplot_height * (1 - vertical_overlap))
                
                ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
                ax.set_facecolor('none')

                # Plot the dPC trace
                trace = Z[comp_type][component_idx, stimulus_idx, :]
                color = self._get_stimulus_color(stimulus)
                ax.plot(time, trace, color=color, linewidth=1.5)
                
                # Style the axes
                ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(y_lim)
                ax.set_xlim(time[0], time[-1])

                # Add component titles at the top row
                if i == 0:
                    ax.set_title(self._get_component_title(comp_type), fontsize=12, pad=10)

        # 5. Add stimulus names and group brackets on the left margin
        label_ax = fig.add_axes([0, 0, left_margin_ratio, 1])
        label_ax.axis('off')

        for i, stimulus in enumerate(stimulus_order):
            y_pos = ((n_stimuli - i - 0.5) * (subplot_height * (1 - vertical_overlap)) + bottom * 0.5) / (1 + bottom * 0.5)
            full_name = self._get_compound_name(stimulus)
            label_ax.text(0.95, y_pos, full_name, ha='right', va='center', fontsize=8)

        # Add tree brackets for compound groups
        bracket_x = 0.1
        bracket_width = 0.1
        
        for compound_name, stimulus_codes in grouped_stimuli:
            group_stimuli_in_plot = [s for s in stimulus_codes if s in stimulus_order]
            if not group_stimuli_in_plot:
                continue

            start_idx = stimulus_order.index(group_stimuli_in_plot[0])
            end_idx = stimulus_order.index(group_stimuli_in_plot[-1])

            y_start_center = ((n_stimuli - start_idx - 0.5) * (subplot_height * (1 - vertical_overlap)) + bottom * 0.5) / (1 + bottom * 0.5)
            y_end_center = ((n_stimuli - end_idx - 0.5) * (subplot_height * (1 - vertical_overlap)) + bottom * 0.5) / (1 + bottom * 0.5)
            
            # Draw bracket
            label_ax.plot([bracket_x, bracket_x + bracket_width], [y_start_center, y_start_center], 'k-', linewidth=1)
            label_ax.plot([bracket_x, bracket_x + bracket_width], [y_end_center, y_end_center], 'k-', linewidth=1)
            label_ax.plot([bracket_x, bracket_x], [y_start_center, y_end_center], 'k-', linewidth=1)
            
            # Add compound name
            label_ax.text(bracket_x - 0.02, (y_start_center + y_end_center) / 2, compound_name, 
                         ha='right', va='center', fontsize=9, weight='bold')

        plt.tight_layout(rect=[left_margin_ratio, 0, 1, 1])
        
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            path = os.path.join(save_folder, f"dpca_grid_pc{component_idx+1}.svg")
            fig.savefig(path, bbox_inches='tight')
            print(f"Figure saved to {path}")

        plt.show()        

if __name__ == "__main__":
    import sys
    import os
    import json
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.HDF5Toolkit import load_h5file
    neuron_segments_dict = load_h5file(
            path = r"I:\WJH\flavor\neuron_segments_dict_filter.h5",
            root_name= 'neuron_segments_dict')
    from result_analysis.baseline_correction import BaselineCorrection
    neuron_segments_dict_correct = BaselineCorrection(neuron_segments_dict)
    neuron_segments_dict_correct.apply_baseline_correction()
    neuron_segments_dict = neuron_segments_dict_correct.corrected_data
    with open(r"H:\Process_temporary\WJH\sensory_pipeline_python\data_load\config\compound_info.json", 'r') as f:
        compound_info = json.load(f)
    with open(r"H:\Process_temporary\WJH\sensory_pipeline_python\data_load\config\compound_color_scheme.json", 'r') as f:
        compound_color_scheme = json.load(f)
    dpca_analyser = SegmentdPCA(neuron_segments_dict, 
                              compound_info=compound_info, 
                              compound_color_scheme=compound_color_scheme)
    dpca_analyser.arrange_data()
    dpca_analyser.fill_blank_neuron_stimuli_pair()
    dpca_analyser.perform_dpca()

    # plot
    dpca_analyser.plot_dpca_results(
    components=['t', 's', 'st'],  # Which components to plot
    component_indices=[0, 1],     # Plot 1st and 2nd components
    figsize=(20, 8),              # Custom figure size
    show_legend=True              # Show compound names
    )
    dpca_analyser.plot_concentration_trends('c1')  # c1 = EGCG
    dpca_analyser.plot_compound_groups()

    dpca_analyser.get_compound_summary()
