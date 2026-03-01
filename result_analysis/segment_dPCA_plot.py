import numpy as np
from utils.parse_stimulus_info import group_and_sort_stimuli
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import os
import hashlib

class SegmentdPCAVisualizer:
    def __init__(self, dpca_model):
        self.model = dpca_model
        if self.model.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() on the model first.")

    def _get_stimulus_color(self, stimulus_name):
        if self.model.compound_color_scheme and stimulus_name in self.model.compound_color_scheme:
            return self.model.compound_color_scheme[stimulus_name]
        hash_object = hashlib.md5(stimulus_name.encode())
        return f"#{hash_object.hexdigest()[:6]}"
        
    def _get_compound_name(self, stimulus_name):
        if self.model.compound_info and stimulus_name in self.model.compound_info:
            return self.model.compound_info[stimulus_name]
        return stimulus_name
        
    def _get_component_title(self, comp_type):
        return {'t': 'Time Component', 's': 'Stimulus Component', 'st': 'Stimulus-Time Interaction'}.get(comp_type, comp_type)

    def _style_axis(self, ax, x_lim=None, y_lim=None, no_spines=True, hide_ticks=False, draw_zero_lines=False):
        if no_spines:
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        if draw_zero_lines:
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
            ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
            
        if hide_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            
        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)

    def _get_axis_limits(self, points_x, points_y, padding_ratio=0.1):
        x_min, x_max = np.min(points_x), np.max(points_x)
        y_min, y_max = np.min(points_y), np.max(points_y)
        
        x_pad = (x_max - x_min) * padding_ratio if (x_max - x_min) > 0 else padding_ratio
        y_pad = (y_max - y_min) * padding_ratio if (y_max - y_min) > 0 else padding_ratio
        
        return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)

    def plot_variance_explained(self, n_components_to_plot=None):
        if self.model.dpca_all is None:
            print("Error: Please run perform_dpca() on the model first.")
            return

        var_dict = self.model.dpca_all.explained_variance_ratio_
        total_components = len(list(var_dict.values())[0])
        if n_components_to_plot is None:
            n_components_to_plot = total_components
        n_plot = min(n_components_to_plot, total_components)
        
        ind = np.arange(n_plot) + 1
        width = 0.7
        colors = {'t': "#0F5685", 's': '#D95319', 'st': '#EDB120', 'dt': '#7E2F8E'}
        
        plt.figure(figsize=(12, 6))
        bottom_tracker = np.zeros(n_plot)
        for key in ['t', 's', 'st']:
            if key in var_dict:
                values = var_dict[key][:n_plot]
                plt.bar(ind, values, width, bottom=bottom_tracker, label=f'Marginalization: {key}', color=colors.get(key, 'gray'))
                bottom_tracker += values

        plt.xlabel('Component Index')
        plt.ylabel('Proportion of Variance Explained')
        plt.title('dPCA Variance Explained by Component')
        plt.xticks(ind)
        plt.legend(loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        total_var_explained = np.sum(bottom_tracker)
        plt.text(n_plot * 0.5, np.max(bottom_tracker)*0.9, f'Total Variance Explained (Top {n_plot}): {total_var_explained:.1%}', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.show()

    def plot_dpca_results(self, components=['t', 's', 'st'], component_indices=[0], figsize=(18, 6), show_legend=True, save_path=None):
        Z = self.model.dpca_results
        time = np.arange(-5, Z['t'].shape[2]-5)
        index_to_stimulus = {idx: stimulus for stimulus, idx in self.model.stimulus_index_map.items()}
        
        n_plots = len(components) * len(component_indices)
        n_cols = len(components)
        n_rows = len(component_indices)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
        axes = [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        
        plot_idx = 0
        for comp_idx in component_indices:
            for comp_type in components:
                ax = axes[plot_idx] if n_plots > 1 else axes[0]
                
                for s in range(Z[comp_type].shape[1]):
                    stimulus_name = index_to_stimulus[s]
                    color = self._get_stimulus_color(stimulus_name)
                    compound_name = self._get_compound_name(stimulus_name)
                    ax.plot(time, Z[comp_type][comp_idx, s], color=color, label=compound_name, linewidth=2)
                
                ax.set_title(f'{self._get_component_title(comp_type)} (Component {comp_idx + 1})', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time(s)')
                ax.set_ylabel('dPC Score')
                self._style_axis(ax, no_spines=False)
                ax.set_xticks(np.arange(-5, time[-1] + 1, 5))

                if show_legend and len(self.model.stimulus_index_map) <= 20:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                plot_idx += 1
                
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if save_path.endswith('.png'):
                plt.savefig(save_path[:-4] + '.pdf', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_concentration_trends(self, compound_base_name, component_type='s', component_idx=0, figsize=(10, 6), save_path=None):
        Z = self.model.dpca_results
        time = np.arange(Z[component_type].shape[2])
        
        matching_stimuli = [s for s in self.model.stimulus_index_map.keys() if s.startswith(compound_base_name + '_')]
        if not matching_stimuli:
            print(f"No stimuli found for compound base name: {compound_base_name}")
            return
            
        matching_stimuli.sort(key=lambda x: int(x.split('_')[1]))
        
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        for stimulus in matching_stimuli:
            stimulus_idx = self.model.stimulus_index_map[stimulus]
            color = self._get_stimulus_color(stimulus)
            compound_name = self._get_compound_name(stimulus)
            ax.plot(time, Z[component_type][component_idx, stimulus_idx], color=color, label=compound_name, linewidth=2, marker='o', markersize=3)
        
        ax.set_title(f'{self._get_component_title(component_type)} - {compound_base_name.upper()} Concentration Series', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('dPC Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_compound_groups(self, component_type='s', component_idx=0, figsize=(15, 10), save_path=None):
        Z = self.model.dpca_results
        time = np.arange(Z[component_type].shape[2])
        
        compound_groups = {}
        for stimulus in self.model.stimulus_index_map.keys():
            if '_' in stimulus:
                base_name = stimulus.split('_')[0]
                compound_groups.setdefault(base_name, []).append(stimulus)
                
        n_groups = len(compound_groups)
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        plot_idx = 0
        for base_name, stimuli in compound_groups.items():
            row, col = plot_idx // n_cols, plot_idx % n_cols
            ax = axes[row, col]
            
            stimuli.sort(key=lambda x: int(x.split('_')[1]) if len(x.split('_')) > 1 else 0)
            for stimulus in stimuli:
                stimulus_idx = self.model.stimulus_index_map[stimulus]
                color = self._get_stimulus_color(stimulus)
                compound_name = self._get_compound_name(stimulus)
                ax.plot(time, Z[component_type][component_idx, stimulus_idx], color=color, label=compound_name, linewidth=2)
            
            ax.set_title(f'{base_name.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Points')
            ax.set_ylabel('dPC Score')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            plot_idx += 1
            
        for i in range(plot_idx, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)
            
        fig.suptitle(f'{self._get_component_title(component_type)} by Compound Groups', fontsize=16, fontweight='bold')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_dpca_grid(self, component_idx=0, vertical_overlap=0.3, fig_width=12, fig_height_per_row=0.8, save_folder=None, group_by_concentration=False):
        if self.model.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")

        Z = self.model.dpca_results
        components_to_plot = ['t', 's', 'st']
        time = np.arange(Z['t'].shape[2])

        rows_data = []
        if group_by_concentration:
            if not self.model.compound_info:
                print("No compound info found for grouping. Cannot plot by concentration.")
                return
            grouped = group_and_sort_stimuli(self.model.compound_info)
            for group_name, stimulus_codes in grouped:
                valid_stimuli = [s for s in stimulus_codes if s in self.model.stimulus_index_map]
                if valid_stimuli:
                    rows_data.append((group_name, valid_stimuli))
        else:
            if self.model.compound_info:
                grouped = group_and_sort_stimuli(self.model.compound_info)
                stimulus_order = [s for _, stimuli in grouped for s in stimuli]
                available_stimuli = set(self.model.stimulus_index_map.keys())
                stimulus_order = [s for s in stimulus_order if s in available_stimuli]
            else:
                stimulus_order = sorted(self.model.stimulus_index_map.keys())
            for s in stimulus_order:
                label = self._get_compound_name(s)
                rows_data.append((label, [s]))

        if not rows_data:
            print("No data to plot.")
            return

        n_rows = len(rows_data)
        n_components = len(components_to_plot)

        all_vals = []
        for _, stimuli in rows_data:
            indices = [self.model.stimulus_index_map[s] for s in stimuli]
            for comp_type in components_to_plot:
                traces = Z[comp_type][component_idx, indices, :]
                all_vals.extend(traces.flatten())
                
        if not all_vals: return
        
        y_lim = self._get_axis_limits([0], all_vals, padding_ratio=0.1)[1]

        total_height = fig_height_per_row * (1 + (n_rows - 1) * (1 - vertical_overlap))
        fig = plt.figure(figsize=(fig_width, total_height))
        
        main_plot_width_ratio = 0.85 if not group_by_concentration else 0.8
        left_margin_ratio = 1.0 - main_plot_width_ratio if not group_by_concentration else 0.15
        right_margin_ratio = 0.05 if group_by_concentration else 0.0 
        
        subplot_width = (1 - left_margin_ratio - right_margin_ratio) / n_components
        subplot_height = fig_height_per_row / total_height
        
        for i, (row_label, stimuli) in enumerate(rows_data):
            if group_by_concentration:
                base_color = self._get_stimulus_color(stimuli[0])
                num_concentrations = len(stimuli)
                hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(base_color))
                hsv_lighter = hsv.copy()
                hsv_lighter[1] *= 0.2 
                hsv_lighter[2] = 1.0
                cmap_local = mcolors.LinearSegmentedColormap.from_list(
                    "custom_cmap", [mcolors.hsv_to_rgb(hsv_lighter), mcolors.to_rgb(base_color)], N=num_concentrations
                )
                line_colors = [cmap_local(k) for k in range(num_concentrations)]
            else:
                line_colors = [self._get_stimulus_color(stimuli[0])]

            for j, comp_type in enumerate(components_to_plot):
                left = left_margin_ratio + j * subplot_width
                bottom = (n_rows - i - 1) * (subplot_height * (1 - vertical_overlap))
                
                ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
                ax.set_facecolor('none')

                for k, s in enumerate(stimuli):
                    stim_idx = self.model.stimulus_index_map[s]
                    trace = Z[comp_type][component_idx, stim_idx, :]
                    ax.plot(time, trace, color=line_colors[k], linewidth=1.5)

                self._style_axis(ax, x_lim=(time[0], time[-1]), y_lim=y_lim, hide_ticks=True, draw_zero_lines=False)
                ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
                ax.axvline(x=5, color='#E6A61C', linestyle='--', linewidth=1.5, zorder=5)
                ax.axvline(x=15, color='#76642E', linestyle='--', linewidth=1.5, zorder=5)

                if i == 0:
                    ax.set_title(f"{self._get_component_title(comp_type)} {component_idx+1}", fontsize=12, pad=10)

        label_ax = fig.add_axes([0, 0, left_margin_ratio, 1])
        label_ax.axis('off')
        for i, (row_label, _) in enumerate(rows_data):
            y_pos_norm = ((n_rows - i - 1) * (subplot_height * (1 - vertical_overlap))) + (subplot_height / 2)
            font_weight = 'bold' if group_by_concentration else 'normal'
            label_ax.text(0.95, y_pos_norm, row_label, ha='right', va='center', 
                        fontsize=10 if group_by_concentration else 8, 
                        weight=font_weight, transform=label_ax.transAxes)

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
        plt.show()

    def plot_trajectory_grid(self, pc_x=0, pc_y=1, vertical_overlap=0.3, fig_width=12, fig_height_per_stimulus=2.0, save_folder=None):
        if self.model.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")

        Z = self.model.dpca_results
        components_to_plot = ['t', 's', 'st']
        time = np.arange(Z['t'].shape[-1])

        if self.model.compound_info:
            grouped_stimuli = group_and_sort_stimuli(self.model.compound_info)
            stimulus_order = [s for _, stimuli in grouped_stimuli for s in stimuli]
            available_stimuli = set(self.model.stimulus_index_map.keys())
            stimulus_order = [s for s in stimulus_order if s in available_stimuli]
        else:
            stimulus_order = sorted(self.model.stimulus_index_map.keys())
            grouped_stimuli = [(s, [s]) for s in stimulus_order]

        n_stimuli = len(stimulus_order)
        n_components = len(components_to_plot)

        all_points_x, all_points_y = [], []
        for comp_type in ['t', 'st']:
            all_points_x.append(Z[comp_type][pc_x, ...])
            all_points_y.append(Z[comp_type][pc_y, ...])
        all_points_x.append(Z['s'][pc_x, :])
        all_points_y.append(Z['s'][pc_y, :])
        
        x_lim, y_lim = self._get_axis_limits(all_points_x, all_points_y, padding_ratio=0.1)

        total_height = fig_height_per_stimulus * (1 + (n_stimuli - 1) * (1 - vertical_overlap))
        fig = plt.figure(figsize=(fig_width, total_height))
        
        main_plot_width_ratio = 0.8
        left_margin_ratio = 0.15
        colorbar_width_ratio = 0.05

        subplot_width = main_plot_width_ratio / n_components
        subplot_height = fig_height_per_stimulus / total_height
        bottom_pos_list = [(n_stimuli - i - 1) * (subplot_height * (1 - vertical_overlap)) for i in range(n_stimuli)]

        for i, stimulus in enumerate(stimulus_order):
            stimulus_idx = self.model.stimulus_index_map[stimulus]
            for j, comp_type in enumerate(components_to_plot):
                left = left_margin_ratio + j * subplot_width
                bottom = bottom_pos_list[i]
                
                ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
                
                if comp_type == 's':
                    x_val, y_val = Z[comp_type][pc_x, stimulus_idx], Z[comp_type][pc_y, stimulus_idx]
                    mean_x, mean_y = np.mean(x_val), np.mean(y_val)
                    std_x, std_y = np.std(x_val), np.std(y_val)

                    color = self._get_stimulus_color(stimulus)
                    ax.plot(mean_x, mean_y, 'o', color=color, markersize=3)
                    ellipse = Ellipse((mean_x, mean_y), width=std_x*2, height=std_y*2, facecolor=color, alpha=0.3)
                    ax.add_patch(ellipse)
                else: 
                    traj_x = Z[comp_type][pc_x, stimulus_idx, :]
                    traj_y = Z[comp_type][pc_y, stimulus_idx, :]
                    points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    norm = plt.Normalize(time.min(), time.max())
                    lc = LineCollection(segments, cmap='viridis', norm=norm)
                    lc.set_array(time)
                    lc.set_linewidth(2)
                    ax.add_collection(lc)

                self._style_axis(ax, x_lim=x_lim, y_lim=y_lim, hide_ticks=True, draw_zero_lines=True)
                if i == 0:
                    ax.set_title(f"{self._get_component_title(comp_type)}", fontsize=12, pad=15)

        label_ax = fig.add_axes([0, 0, left_margin_ratio, 1])
        label_ax.axis('off')
        for i, stimulus in enumerate(stimulus_order):
            y_pos = bottom_pos_list[i] + subplot_height / 2
            full_name = self._get_compound_name(stimulus).split(" ")[-1]
            if len(full_name) > 3: full_name = full_name[:3] + "."
            label_ax.text(0.95, y_pos, full_name, ha='right', va='center', fontsize=9, transform=label_ax.transAxes)

        bracket_x, bracket_width = 0.6, 0.05
        for compound_name, stimulus_codes in grouped_stimuli:
            group_stimuli_in_plot = [s for s in stimulus_codes if s in stimulus_order]
            if not group_stimuli_in_plot: continue
            start_idx, end_idx = stimulus_order.index(group_stimuli_in_plot[0]), stimulus_order.index(group_stimuli_in_plot[-1])
            y_start_center = bottom_pos_list[start_idx] + subplot_height / 2
            y_end_center = bottom_pos_list[end_idx] + subplot_height / 2
            label_ax.plot([bracket_x, bracket_x + bracket_width], [y_start_center, y_start_center], 'k-', lw=1, transform=label_ax.transAxes)
            label_ax.plot([bracket_x, bracket_x + bracket_width], [y_end_center, y_end_center], 'k-', lw=1, transform=label_ax.transAxes)
            label_ax.plot([bracket_x, bracket_x], [y_start_center, y_end_center], 'k-', lw=1, transform=label_ax.transAxes)
            label_ax.text(bracket_x - 0.02, (y_start_center + y_end_center) / 2, compound_name, ha='right', va='center', fontsize=10, weight='bold', transform=label_ax.transAxes, rotation=90)

        cbar_ax = fig.add_axes([1 - colorbar_width_ratio - 0.02, 0.1, colorbar_width_ratio, 0.8])
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(time.min(), time.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Time (s)', fontsize=12)

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            path = os.path.join(save_folder, f"dpca_trajectory_grid_pc{pc_x+1}v{pc_y+1}.svg")
            fig.savefig(path, bbox_inches='tight')
        plt.show()

    def plot_component(self, component_to_plot, pc_x=0, pc_y=1, figsize=(7, 6), save_folder=None):
        if self.model.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")

        valid_components = ['t', 's', 'st']
        if component_to_plot not in valid_components:
            raise ValueError(f"component_to_plot must be one of {valid_components}")

        Z = self.model.dpca_results
        time = np.arange(Z['t'].shape[-1])

        if self.model.compound_info:
            grouped_stimuli = group_and_sort_stimuli(self.model.compound_info)
            stimulus_order = [s for _, stimuli in grouped_stimuli for s in stimuli]
            available_stimuli = set(self.model.stimulus_index_map.keys())
            stimulus_order = [s for s in stimulus_order if s in available_stimuli]
        else:
            stimulus_order = sorted(self.model.stimulus_index_map.keys())

        points_x = Z[component_to_plot][pc_x, ...]
        points_y = Z[component_to_plot][pc_y, ...]
        
        x_lim, y_lim = self._get_axis_limits(points_x, points_y, padding_ratio=0.1)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if component_to_plot == 't':
            mean_time_trajectory = np.mean(Z['t'], axis=1)
            traj_x = mean_time_trajectory[pc_x, :]
            traj_y = mean_time_trajectory[pc_y, :]
            
            points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            norm = plt.Normalize(time.min(), time.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(time)
            lc.set_linewidth(8)
            ax.add_collection(lc)

        elif component_to_plot == 's':
            for stimulus in stimulus_order:
                stimulus_idx = self.model.stimulus_index_map[stimulus]
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
                    ellipse = Ellipse((mean_x, mean_y), width=std_x, height=std_y, facecolor=color, alpha=0.25)
                    ax.add_patch(ellipse)
            ax.legend(title="Stimuli", loc="best", markerscale=0.5)

        elif component_to_plot == 'st':
            for stimulus in stimulus_order:
                stimulus_idx = self.model.stimulus_index_map[stimulus]
                traj_x = Z['st'][pc_x, stimulus_idx, :]
                traj_y = Z['st'][pc_y, stimulus_idx, :]
                
                base_color = self._get_stimulus_color(stimulus)
                ax.plot(traj_x, traj_y, '-', color=base_color, linewidth=2.5, alpha=0.7)

            legend_elements = [Line2D([0], [0], color=self._get_stimulus_color(s), lw=2, 
                                    label=self._get_compound_name(s)) for s in stimulus_order]
            ax.legend(handles=legend_elements, title="Stimuli", loc="best")

        self._style_axis(ax, x_lim=x_lim, y_lim=y_lim, hide_ticks=True, draw_zero_lines=False)
        ax.axhline(0, color='grey', linestyle='--', linewidth=3)
        ax.axvline(0, color='grey', linestyle='--', linewidth=3)
        fig.tight_layout()
        
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            path = os.path.join(save_folder, f"dpca_{component_to_plot}_pc{pc_x+1}v{pc_y+1}.svg")
            fig.savefig(path, bbox_inches='tight')
        plt.show()
