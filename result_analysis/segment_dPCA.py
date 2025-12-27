import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from dPCA import dPCA
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import hashlib
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import sem
from utils.parse_stimulus_info import group_and_sort_stimuli
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
        # elif neuron_name.endswith('ON'):
        #     symmetric_name = neuron_name[:-2] + "OFF"
        # elif neuron_name.endswith('OFF'):
        #     symmetric_name = neuron_name[:-3] + "ON"
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
    
    def plot_variance_explained(self, n_components_to_plot=None):
        """
        Plot histogram for variance explained for raw data.
        
        :param n_components_to_plot: top n Component, none for all components.
        """
        if self.dpca_all is None:
            print("Error: Please run perform_dpca() first.")
            return
            
        import matplotlib.pyplot as plt
        import numpy as np

        # get variance data
        var_dict = self.dpca_all.explained_variance_ratio_
        
        total_components = len(list(var_dict.values())[0])
        
        if n_components_to_plot is None:
            n_components_to_plot = total_components

        n_plot = min(n_components_to_plot, total_components)
        
        # set plot parameter
        ind = np.arange(n_plot) + 1
        width = 0.7
        
        # dPCA 标准配色 (Kobak et al.)
        # t (Time): 蓝色
        # s (Stimulus): 红色/橙色
        # st (Interaction): 黄色
        colors = {
            't': "#0F5685",  # Blue
            's': '#D95319',  # Orange
            'st': '#EDB120', # Yellow (Interaction)
            'dt': '#7E2F8E'  # Purple (如果将来有其他 label)
        }
        
        plt.figure(figsize=(12, 6))
        bottom_tracker = np.zeros(n_plot)
        
        # 3. 循环绘制堆叠图
        # 这里的顺序决定了堆叠的层级，通常按 t, s, st 顺序
        for key in ['t', 's', 'st']:
            if key in var_dict:
                # 获取该 key 对前 n_plot 个 component 的方差贡献
                values = var_dict[key][:n_plot]
                
                plt.bar(ind, values, width, bottom=bottom_tracker, 
                        label=f'Marginalization: {key}', 
                        color=colors.get(key, 'gray'))
                
                # 更新底部高度，以便下一层堆叠
                bottom_tracker += values

        # 4. 装饰图表
        plt.xlabel('Component Index')
        plt.ylabel('Proportion of Variance Explained')
        plt.title('dPCA Variance Explained by Component')
        plt.xticks(ind)
        plt.legend(loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 计算并显示总方差解释率
        total_var_explained = np.sum(bottom_tracker)
        plt.text(n_plot * 0.5, np.max(bottom_tracker)*0.9, 
                f'Total Variance Explained (Top {n_plot}): {total_var_explained:.1%}', 
                fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()
        

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
        time = np.arange(-5, Z['t'].shape[2]-5)
        
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
                ax.set_xlabel('Time(s)')
                ax.set_ylabel('dPC Score')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticks(np.arange(-5, time[-1] + 1, 5))
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
        
        :param compound_base_name: Base name of compound
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

    def plot_dpca_grid(self, component_idx=0, vertical_overlap=0.3, fig_width=12, fig_height_per_row=0.8, save_folder=None, group_by_concentration=False):
        """
        Plots dPCA results in a grid.
        
        :param component_idx: Index of the dPC to plot.
        :param vertical_overlap: Overlap between rows.
        :param fig_width: Total width of the figure.
        :param fig_height_per_row: Height for each row (stimulus or group).
        :param save_folder: Folder to save the figure.
        :param group_by_concentration: If True, rows are compound groups with concentration gradients. 
                                     If False, rows are individual stimuli.
        """
        if self.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")

        Z = self.dpca_results
        components_to_plot = ['t', 's', 'st']
        time = np.arange(Z['t'].shape[2])

        # --- 1. Prepare Rows Data ---
        # Structure: list of tuples (row_label, [stimulus_list])
        rows_data = []
        
        if group_by_concentration:
            if not self.compound_info:
                print("No compound info found for grouping. Cannot plot by concentration.")
                return
            
            grouped = group_and_sort_stimuli(self.compound_info)
            for group_name, stimulus_codes in grouped:
                valid_stimuli = [s for s in stimulus_codes if s in self.stimulus_index_map]
                if valid_stimuli:
                    rows_data.append((group_name, valid_stimuli))
        else:
            # Original logic for individual stimuli order
            if self.compound_info:
                grouped = group_and_sort_stimuli(self.compound_info)
                stimulus_order = [s for _, stimuli in grouped for s in stimuli]
                available_stimuli = set(self.stimulus_index_map.keys())
                stimulus_order = [s for s in stimulus_order if s in available_stimuli]
            else:
                stimulus_order = sorted(self.stimulus_index_map.keys())
            
            for s in stimulus_order:
                # Label logic from original plot_dpca_grid
                label = self._get_compound_name(s)
                rows_data.append((label, [s]))

        if not rows_data:
            print("No data to plot.")
            return

        n_rows = len(rows_data)
        n_components = len(components_to_plot)

        # --- 2. Calculate Y-Limits ---
        all_vals = []
        for _, stimuli in rows_data:
            indices = [self.stimulus_index_map[s] for s in stimuli]
            for comp_type in components_to_plot:
                traces = Z[comp_type][component_idx, indices, :]
                all_vals.extend(traces.flatten())
        
        if not all_vals:
             print("No data found for limits.")
             return

        y_min, y_max = np.min(all_vals), np.max(all_vals)
        padding = (y_max - y_min) * 0.1
        y_lim = (y_min - padding, y_max + padding)

        # --- 3. Setup Figure ---
        total_height = fig_height_per_row * (1 + (n_rows - 1) * (1 - vertical_overlap))
        fig = plt.figure(figsize=(fig_width, total_height))
        
        main_plot_width_ratio = 0.85 if not group_by_concentration else 0.8
        left_margin_ratio = 1.0 - main_plot_width_ratio if not group_by_concentration else 0.15
        right_margin_ratio = 0.05 if group_by_concentration else 0.0 # Space for legend
        
        subplot_width = (1 - left_margin_ratio - right_margin_ratio) / n_components
        subplot_height = fig_height_per_row / total_height
        
        # --- 4. Plotting Loop ---
        for i, (row_label, stimuli) in enumerate(rows_data):
            # Prepare colors
            if group_by_concentration:
                base_color = self._get_stimulus_color(stimuli[0])
                num_concentrations = len(stimuli)
                colors = [mcolors.to_rgb(base_color)]
                hsv = mcolors.rgb_to_hsv(colors[0])
                hsv_lighter = hsv.copy()
                hsv_lighter[1] *= 0.2 
                hsv_lighter[2] = 1.0
                cmap_local = mcolors.LinearSegmentedColormap.from_list(
                    "custom_cmap", [mcolors.hsv_to_rgb(hsv_lighter), colors[0]], N=num_concentrations
                )
                line_colors = [cmap_local(k) for k in range(num_concentrations)]
            else:
                # Single stimulus
                line_colors = [self._get_stimulus_color(stimuli[0])]

            for j, comp_type in enumerate(components_to_plot):
                left = left_margin_ratio + j * subplot_width
                bottom = (n_rows - i - 1) * (subplot_height * (1 - vertical_overlap))
                
                ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
                ax.set_facecolor('none')

                for k, s in enumerate(stimuli):
                    stim_idx = self.stimulus_index_map[s]
                    trace = Z[comp_type][component_idx, stim_idx, :]
                    ax.plot(time, trace, color=line_colors[k], linewidth=1.5)

                # Styling
                ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(y_lim)
                ax.set_xlim(time[0], time[-1])
                ax.axvline(x=5, color='#E6A61C', linestyle='--', linewidth=1.5, zorder=5)
                ax.axvline(x=15, color='#76642E', linestyle='--', linewidth=1.5, zorder=5)

                if i == 0:
                    ax.set_title(f"{self._get_component_title(comp_type)} {component_idx+1}", fontsize=12, pad=10)

        # --- 5. Labels ---
        label_ax = fig.add_axes([0, 0, left_margin_ratio, 1])
        label_ax.axis('off')
        for i, (row_label, _) in enumerate(rows_data):
            y_pos_norm = ((n_rows - i - 1) * (subplot_height * (1 - vertical_overlap))) + (subplot_height / 2)
            font_weight = 'bold' if group_by_concentration else 'normal'
            label_ax.text(0.95, y_pos_norm, row_label, ha='right', va='center', 
                        fontsize=10 if group_by_concentration else 8, 
                        weight=font_weight, transform=label_ax.transAxes)

        # --- 6. Legend (Concentration only) ---
        if group_by_concentration:
            legend_ax = fig.add_axes([1 - right_margin_ratio, 0.7, 0.015, 0.2])
            cmap = mcolors.LinearSegmentedColormap.from_list("grad_legend", ["lightgray", "black"])
            norm = mcolors.Normalize(vmin=0, vmax=1)
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=legend_ax, orientation='vertical')
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'])
            cb.set_label('Concentration', size=10, labelpad=5)
            cb.outline.set_visible(False)

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            suffix = "by_conc" if group_by_concentration else "grid"
            path = os.path.join(save_folder, f"dpca_{suffix}_pc{component_idx+1}.svg")
            fig.savefig(path, bbox_inches='tight')
            print(f"Figure saved to {path}")
        
        plt.show()

    def plot_trajectory_grid(self, pc_x=0, pc_y=1, vertical_overlap=0.3, fig_width=12, fig_height_per_stimulus=2.0, save_folder=None):
        """
        Plots dPCA trajectories in a grid, with stimuli as rows and component types as columns.
        
        Args:
            pc_x (int): Index of the dPC for the x-axis.
            pc_y (int): Index of the dPC for the y-axis.
            vertical_overlap (float): Overlap between rows.
            fig_width (int): Total width of the figure.
            fig_height_per_stimulus (float): Height for each stimulus row.
            save_folder (str): Folder to save the figure.
        """
        if self.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")

        Z = self.dpca_results
        components_to_plot = ['t', 's', 'st']
        time = np.arange(Z['t'].shape[-1])

        # 1. Get sorted and grouped stimuli list
        if self.compound_info:
            grouped_stimuli = group_and_sort_stimuli(self.compound_info)
            stimulus_order = [s for _, stimuli in grouped_stimuli for s in stimuli]
            available_stimuli = set(self.stimulus_index_map.keys())
            stimulus_order = [s for s in stimulus_order if s in available_stimuli]
        else:
            stimulus_order = sorted(self.stimulus_index_map.keys())
            grouped_stimuli = [(s, [s]) for s in stimulus_order]

        n_stimuli = len(stimulus_order)
        n_components = len(components_to_plot)

        # 2. Calculate global axis limits for all trajectories
        all_points_x, all_points_y = [], []
        for comp_type in ['t', 'st']:
            all_points_x.append(Z[comp_type][pc_x, ...])
            all_points_y.append(Z[comp_type][pc_y, ...])
        all_points_x.append(Z['s'][pc_x, :])
        all_points_y.append(Z['s'][pc_y, :])
        
        x_min, x_max = np.min(all_points_x), np.max(all_points_x)
        y_min, y_max = np.min(all_points_y), np.max(all_points_y)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_lim = (x_min - x_padding, x_max + x_padding)
        y_lim = (y_min - y_padding, y_max + y_padding)

        # 3. Setup Figure
        total_height = fig_height_per_stimulus * (1 + (n_stimuli - 1) * (1 - vertical_overlap))
        fig = plt.figure(figsize=(fig_width, total_height))
        
        main_plot_width_ratio = 0.8
        left_margin_ratio = 0.15
        colorbar_width_ratio = 0.05

        subplot_width = main_plot_width_ratio / n_components
        subplot_height = fig_height_per_stimulus / total_height
        bottom_pos_list = [(n_stimuli - i - 1) * (subplot_height * (1 - vertical_overlap)) for i in range(n_stimuli)]

        # 4. Loop through stimuli and components to create subplots
        for i, stimulus in enumerate(stimulus_order):
            stimulus_idx = self.stimulus_index_map[stimulus]
            
            for j, comp_type in enumerate(components_to_plot):
                left = left_margin_ratio + j * subplot_width
                bottom = bottom_pos_list[i]
                
                ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
                
                if comp_type == 's': # Plot a single point for stimulus component
                    x_val = Z[comp_type][pc_x, stimulus_idx]
                    y_val = Z[comp_type][pc_y, stimulus_idx]

                    mean_x, mean_y = np.mean(x_val), np.mean(y_val)
                    std_x, std_y = np.std(x_val), np.std(y_val)

                    color = self._get_stimulus_color(stimulus)
                    ax.plot(mean_x, mean_y, 'o', color=color, markersize=3)
                    # ax.plot(x_val, y_val, '-', color=color, linewidth=1.5, alpha=0.7)
                    # ax.plot(x_val[0], y_val[0], 'o', color=color, markersize=3, markeredgecolor='k', mew=0.5)
                    # ax.plot(x_val[-1], y_val[-1], 's', color=color, markersize=3, markeredgecolor='k', mew=0.5)
                    # Add a translucent ellipse representing one standard deviation
                    ellipse = Ellipse((mean_x, mean_y), width=std_x*2, height=std_y*2,
                                    facecolor=color, alpha=0.3)
                    ax.add_patch(ellipse)
                else: # Plot a trajectory for time and interaction
                    if comp_type == 't':
                        # Time component is stimulus-independent
                        traj_x = Z[comp_type][pc_x, stimulus_idx, :]
                        traj_y = Z[comp_type][pc_y, stimulus_idx, :]
                    else: # 'st'
                        traj_x = Z[comp_type][pc_x, stimulus_idx, :]
                        traj_y = Z[comp_type][pc_y, stimulus_idx, :]

                    points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    norm = plt.Normalize(time.min(), time.max())
                    lc = LineCollection(segments, cmap='viridis', norm=norm)
                    lc.set_array(time)
                    lc.set_linewidth(2)
                    ax.add_collection(lc)

                # Style the axes
                ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
                ax.axvline(0, color='k', linestyle=':', linewidth=0.5)
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                if i == 0:
                    title = f"{self._get_component_title(comp_type)}"
                    ax.set_title(title, fontsize=12, pad=15)

        # 5. Add stimulus names and group brackets
        label_ax = fig.add_axes([0, 0, left_margin_ratio, 1])
        label_ax.axis('off')
        for i, stimulus in enumerate(stimulus_order):
            y_pos = bottom_pos_list[i] + subplot_height / 2
            full_name = self._get_compound_name(stimulus).split(" ")[-1]
            if len(full_name) > 3:
                full_name = full_name[:3] + "."
            label_ax.text(0.95, y_pos, full_name, ha='right', va='center', fontsize=9, transform=label_ax.transAxes)

        bracket_x = 0.6
        bracket_width = 0.05
        for compound_name, stimulus_codes in grouped_stimuli:
            group_stimuli_in_plot = [s for s in stimulus_codes if s in stimulus_order]
            if not group_stimuli_in_plot: continue
            start_idx, end_idx = stimulus_order.index(group_stimuli_in_plot[0]), stimulus_order.index(group_stimuli_in_plot[-1])
            y_start_center, y_end_center = bottom_pos_list[start_idx] + subplot_height / 2, bottom_pos_list[end_idx] + subplot_height / 2
            label_ax.plot([bracket_x, bracket_x + bracket_width], [y_start_center, y_start_center], 'k-', lw=1, transform=label_ax.transAxes)
            label_ax.plot([bracket_x, bracket_x + bracket_width], [y_end_center, y_end_center], 'k-', lw=1, transform=label_ax.transAxes)
            label_ax.plot([bracket_x, bracket_x], [y_start_center, y_end_center], 'k-', lw=1, transform=label_ax.transAxes)
            label_ax.text(bracket_x - 0.02, (y_start_center + y_end_center) / 2, compound_name, ha='right', va='center', fontsize=10, weight='bold', transform=label_ax.transAxes, rotation=90)

        # 6. Add a colorbar
        cbar_ax = fig.add_axes([1 - colorbar_width_ratio - 0.02, 0.1, colorbar_width_ratio, 0.8])
        norm = plt.Normalize(time.min(), time.max())
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Time (s)', fontsize=12)

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            path = os.path.join(save_folder, f"dpca_trajectory_grid_pc{pc_x+1}v{pc_y+1}.svg")
            fig.savefig(path, bbox_inches='tight')
            print(f"Figure saved to {path}")

        plt.show()


    def plot_component(self, component_to_plot, pc_x=0, pc_y=1, figsize=(7, 6), save_folder=None):
        """
        根据选择的dPCA成分（'t', 's', 或 'st'）绘制单张投影图。

        Args:
            component_to_plot (str): 要绘制的成分。必须是 't', 's', 或 'st' 之一。
            pc_x (int): 用于x轴的dPC索引。
            pc_y (int): 用于y轴的dPC索引。
            figsize (tuple): Matplotlib图窗的大小。
            save_folder (str): 保存图像的文件夹。如果为None，则不保存。
        """
        if self.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")

        valid_components = ['t', 's', 'st']
        if component_to_plot not in valid_components:
            raise ValueError(f"component_to_plot must be one of {valid_components}")

        Z = self.dpca_results
        time = np.arange(Z['t'].shape[-1])

        # 1. 获取排序后的刺激列表
        if self.compound_info:
            # 假设 group_and_sort_stimuli 在别处已定义
            grouped_stimuli = group_and_sort_stimuli(self.compound_info)
            stimulus_order = [s for _, stimuli in grouped_stimuli for s in stimuli]
            available_stimuli = set(self.stimulus_index_map.keys())
            stimulus_order = [s for s in stimulus_order if s in available_stimuli]
        else:
            stimulus_order = sorted(self.stimulus_index_map.keys())

        # 2. 仅为选定的 component 计算坐标轴范围
        points_x = Z[component_to_plot][pc_x, ...]
        points_y = Z[component_to_plot][pc_y, ...]
        x_min, x_max = np.min(points_x), np.max(points_x)
        y_min, y_max = np.min(points_y), np.max(points_y)
        
        x_padding = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 0.1
        y_padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
        x_lim = (x_min - x_padding, x_max + x_padding)
        y_lim = (y_min - y_padding, y_max + y_padding)
        # x_lim = (x_min*0.8, x_max*0.8 )
        # y_lim = (y_min*0.4, y_max*0.7)

        # 3. 设置单张图
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # 4. 根据选择的 component 执行相应的绘图逻辑
        # --- A. 绘制时间成分 ('t') ---
        if component_to_plot == 't':
            mean_time_trajectory = np.mean(Z['t'], axis=1)
            
            # 2. 从平均轨迹中提取 x 和 y 坐标
            traj_x = mean_time_trajectory[pc_x, :]
            traj_y = mean_time_trajectory[pc_y, :]
            
            # 3. 后续绘图代码保持不变
            points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            norm = plt.Normalize(time.min(), time.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(time)
            lc.set_linewidth(8)
            ax.add_collection(lc)


        # --- B. 绘制刺激成分 ('s') ---
        elif component_to_plot == 's':
            # ax.set_title(f'Stimulus Component (dPC {pc_x+1} vs dPC {pc_y+1})')
            for stimulus in stimulus_order:
                stimulus_idx = self.stimulus_index_map[stimulus]
                color = self._get_stimulus_color(stimulus)
                x_vals, y_vals = Z['s'][pc_x, stimulus_idx], Z['s'][pc_y, stimulus_idx]

                if hasattr(x_vals, "__len__") and len(x_vals) > 1:
                    mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
                    std_x, std_y = np.std(x_vals), np.std(y_vals)
                else:
                    mean_x, mean_y, std_x, std_y = x_vals, y_vals, 0, 0
                
                ax.plot(mean_x, mean_y, 'o', color=color, markersize=15, 
                        markeredgecolor='k', mew=0.5, label=self._get_compound_name(stimulus))

                if std_x > 0 and std_y > 0:
                    ellipse = Ellipse((mean_x, mean_y), width=std_x, height=std_y,
                                    facecolor=color, alpha=0.25)
                    ax.add_patch(ellipse)
            ax.legend(title="Stimuli", loc="best", markerscale=0.5)

        # --- C. 绘制交互成分 ('st') ---
        elif component_to_plot == 'st':
            # ax.set_title(f'Stimulus-Time Interaction (dPC {pc_x+1} vs dPC {pc_y+1})')
            for stimulus in stimulus_order:
                stimulus_idx = self.stimulus_index_map[stimulus]
                traj_x = Z['st'][pc_x, stimulus_idx, :]
                traj_y = Z['st'][pc_y, stimulus_idx, :]
                
                points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                base_color = self._get_stimulus_color(stimulus)
                # cmap = LinearSegmentedColormap.from_list('custom_map', [(1, 1, 1), base_color])
                # norm = plt.Normalize(time.min(), time.max())
                
                # lc = LineCollection(segments, cmap=cmap, norm=norm)
                # lc.set_array(time)
                # lc.set_linewidth(2.5)
                # ax.add_collection(lc)

                ax.plot(traj_x, traj_y, '-', color=base_color, linewidth=2.5, alpha=0.7)
                # ax.plot(traj_x[0], traj_y[0], 'o', markersize=5, markeredgecolor='k', color=base_color, alpha=0.7)
                # ax.plot(traj_x[-1], traj_y[-1], 's', markersize=5, markeredgecolor='k', color=base_color)

            # 为 'st' 图创建图例
            legend_elements = [Line2D([0], [0], color=self._get_stimulus_color(s), lw=2, 
                                    label=self._get_compound_name(s)) for s in stimulus_order]
            ax.legend(handles=legend_elements, title="Stimuli", loc="best")

        # 5. 通用样式设置
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axhline(0, color='grey', linestyle='--', linewidth=3)
        ax.axvline(0, color='grey', linestyle='--', linewidth=3)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # 6. 保存和显示
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            path = os.path.join(save_folder, f"dpca_{component_to_plot}_pc{pc_x+1}v{pc_y+1}.svg")
            fig.savefig(path, bbox_inches='tight')
            print(f"Figure saved to {path}")

        plt.show()