import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import os

class EpochPCA:
    def __init__(self, neuron_segments_dict, config_path=None):
        self.data = neuron_segments_dict
        self.pca_results = {}
        self.arranged_data = None
        self.stimulus_labels = None
        self.neuron_names = None
        self.time_points = None
        
        # Load compound information and color scheme
        if config_path is None:
            # Default path relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', 'data_load', 'config')
        
        self.compound_info = self._load_compound_info(config_path)
        self.color_scheme = self._load_color_scheme(config_path)
        
    def _load_compound_info(self, config_path):
        """Load compound information from JSON file."""
        try:
            compound_info_path = os.path.join(config_path, 'compound_info.json')
            with open(compound_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load compound_info.json: {e}")
            return {}
    
    def _load_color_scheme(self, config_path):
        """Load color scheme from JSON file."""
        try:
            color_scheme_path = os.path.join(config_path, 'compound_color_scheme.json')
            with open(color_scheme_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load compound_color_scheme.json: {e}")
            return {}
        
    def arrange_data(self, use_zscore=False, time_window=None, average_trials=True):
        """
        Arrange data for PCA analysis.
        
        Parameters:
        -----------
        use_zscore : bool, default=False
            Whether to use z-scored data or deltaFoverF_0
        time_window : tuple, default=None
            Time window (start, end) in seconds relative to stimulus onset.
            If None, uses the full trial duration.
        average_trials : bool, default=False
            If True, average all trials for each stimulus to get one sample per stimulus.
            If False, keep each trial as a separate sample.
            
        Returns:
        --------
        arranged_data : np.ndarray
            Shape: (n_samples, n_features) where:
            - If average_trials=False: n_samples = total trials across all stimuli
            - If average_trials=True: n_samples = number of stimuli
            - n_features = n_neurons * n_timepoints
        stimulus_labels : list
            List of stimulus names corresponding to each sample
        """
        
        # Get all neuron names and stimulus names
        all_neurons = list(self.data.keys())
        all_stimuli = set()
        for neuron_data in self.data.values():
            all_stimuli.update(neuron_data.keys())
        all_stimuli = sorted(list(all_stimuli))
        
        print(f"Found {len(all_neurons)} neurons and {len(all_stimuli)} stimuli")
        
        # Collect all trials and determine time grid
        all_trials = []
        time_grids = []
        
        for neuron_name in all_neurons:
            for stimulus_name in all_stimuli:
                if stimulus_name in self.data[neuron_name]:
                    trials = self.data[neuron_name][stimulus_name]
                    for trial in trials:
                        # Get the appropriate data field
                        if use_zscore and trial.get("z_scored") is not None:
                            trace_data = trial["z_scored"]
                        else:
                            trace_data = trial["deltaFoverF_0"]
                        
                        # Create time grid for this trial
                        start_time = 0
                        end_time = len(trace_data)-1
                        n_points = len(trace_data)
                        time_grid = np.linspace(start_time, end_time, n_points)
                        time_grids.append(time_grid)
        
        # Determine common time grid
        if time_window is not None:
            # Use specified time window
            min_time, max_time = time_window
        else:
            # Use the intersection of all time ranges
            min_time = max([tg[0] for tg in time_grids])
            max_time = min([tg[-1] for tg in time_grids])
        
        # Find the maximum number of time points across trials
        max_points = 0
        for tg in time_grids:
            valid_indices = (tg >= min_time) & (tg <= max_time)
            max_points = max(max_points, np.sum(valid_indices))
        
        # Create uniform time grid
        self.time_points = np.linspace(min_time, max_time, max_points)
        print(f"Using time window: {min_time:.2f} to {max_time:.2f} seconds with {max_points} time points")
        
        # Collect and interpolate data
        samples = []
        labels = []
        
        for stimulus_name in all_stimuli:
            stimulus_samples = []
            
            # Collect all trials for this stimulus across all neurons
            for neuron_name in all_neurons:
                if stimulus_name in self.data[neuron_name]:
                    trials = self.data[neuron_name][stimulus_name]
                    
                    for trial in trials:
                        # Get the appropriate data field
                        if use_zscore and trial.get("z_scored") is not None:
                            trace_data = trial["z_scored"]
                        else:
                            trace_data = trial["deltaFoverF_0"]
                        
                        # Create time grid for this trial
                        start_time = 0
                        end_time = len(trace_data) - 1
                        n_points = len(trace_data)
                        trial_time_grid = np.linspace(start_time, end_time, n_points)
                        
                        # Find the trial identifier
                        trial_id = f"{trial['worm_key']}_{trial['segment_index']}_{trial['date']}"
                        
                        # Check if we already have this trial
                        existing_trial = None
                        for sample in stimulus_samples:
                            if sample['trial_id'] == trial_id:
                                existing_trial = sample
                                break
                        
                        if existing_trial is None:
                            # Create new trial entry
                            trial_sample = {
                                'trial_id': trial_id,
                                'neuron_data': {},
                                'stimulus': stimulus_name
                            }
                            stimulus_samples.append(trial_sample)
                            existing_trial = trial_sample
                        
                        # Interpolate trace to common time grid
                        # Only use points within the time window
                        valid_indices = (trial_time_grid >= min_time) & (trial_time_grid <= max_time)
                        valid_time = trial_time_grid[valid_indices]
                        valid_trace = np.array(trace_data)[valid_indices]
                        
                        if len(valid_time) > 1:
                            interpolated_trace = np.interp(self.time_points, valid_time, valid_trace)
                        else:
                            # If only one or no valid points, use the mean or zero
                            interpolated_trace = np.full(len(self.time_points), 
                                                        np.mean(valid_trace) if len(valid_trace) > 0 else 0)
                        
                        existing_trial['neuron_data'][neuron_name] = interpolated_trace
            
            # Convert to feature vectors (concatenate all neurons for each trial)
            if average_trials:
                # Average all trials for this stimulus
                if stimulus_samples:
                    # Collect all feature vectors for this stimulus
                    stimulus_feature_vectors = []
                    for trial_sample in stimulus_samples:
                        feature_vector = []
                        for neuron_name in all_neurons:
                            if neuron_name in trial_sample['neuron_data']:
                                feature_vector.extend(trial_sample['neuron_data'][neuron_name])
                            else:
                                # Fill with zeros if neuron data is missing for this trial
                                feature_vector.extend(np.zeros(len(self.time_points)))
                        stimulus_feature_vectors.append(feature_vector)
                    
                    # Average across all trials for this stimulus
                    if stimulus_feature_vectors:
                        mean_feature_vector = np.mean(stimulus_feature_vectors, axis=0)
                        samples.append(mean_feature_vector)
                        labels.append(stimulus_name)
            else:
                # Keep individual trials as separate samples
                for trial_sample in stimulus_samples:
                    feature_vector = []
                    for neuron_name in all_neurons:
                        if neuron_name in trial_sample['neuron_data']:
                            feature_vector.extend(trial_sample['neuron_data'][neuron_name])
                        else:
                            # Fill with zeros if neuron data is missing for this trial
                            feature_vector.extend(np.zeros(len(self.time_points)))
                    
                    samples.append(feature_vector)
                    labels.append(stimulus_name)
        
        self.arranged_data = np.array(samples)
        self.stimulus_labels = labels
        self.neuron_names = all_neurons
        
        print(f"Arranged data shape: {self.arranged_data.shape}")
        if average_trials:
            print(f"Data represents averaged responses for {len(set(labels))} stimuli")
            print(f"Stimuli: {sorted(set(labels))}")
        else:
            print(f"Number of samples per stimulus: {dict(pd.Series(labels).value_counts())}")
        
        return self.arranged_data, self.stimulus_labels
    
    def perform_pca(self, n_components=None, standardize=True):
        """
        Perform PCA on the arranged data.
        
        Parameters:
        -----------
        n_components : int, default=None
            Number of principal components to compute. If None, compute all.
        standardize : bool, default=True
            Whether to standardize features before PCA
            
        Returns:
        --------
        pca_result : dict
            Dictionary containing PCA results
        """
        
        if self.arranged_data is None:
            raise ValueError("Data not arranged yet. Call arrange_data() first.")
        
        # Standardize data if requested
        if standardize:
            scaler = StandardScaler()
            data_for_pca = scaler.fit_transform(self.arranged_data)
            self.scaler = scaler
        else:
            data_for_pca = self.arranged_data
            self.scaler = None
        
        # Perform PCA
        if n_components is None:
            n_components = min(data_for_pca.shape) - 1
        
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data_for_pca)
        
        # Store results
        self.pca_results = {
            'pca_model': pca,
            'transformed_data': transformed_data,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'n_components': n_components,
            'standardized': standardize
        }
        
        print(f"PCA completed with {n_components} components")
        print(f"Explained variance ratio (first 5 PCs): {pca.explained_variance_ratio_[:5]}")
        print(f"Cumulative explained variance (first 5 PCs): {np.cumsum(pca.explained_variance_ratio_)[:5]}")
        
        return self.pca_results
    
    def plot_explained_variance(self, n_components_to_show=20):
        """
        Plot explained variance ratio and cumulative explained variance.
        """
        if not self.pca_results:
            raise ValueError("PCA not performed yet. Call perform_pca() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot explained variance ratio
        n_show = min(n_components_to_show, len(self.pca_results['explained_variance_ratio']))
        ax1.bar(range(1, n_show + 1), self.pca_results['explained_variance_ratio'][:n_show])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Principal Component')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative explained variance
        ax2.plot(range(1, n_show + 1), self.pca_results['cumulative_variance_ratio'][:n_show], 'o-')
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% variance')
        ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% variance')
        ax2.set_xlabel('Number of Principal Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_pca_scatter(self, pc_x=1, pc_y=2, color_by_stimulus=True, figsize=(12, 9), 
                        use_compound_names=True, show_legend=True):
        """
        Plot PCA scatter plot with compound information and color scheme.
        
        Parameters:
        -----------
        pc_x : int, default=1
            Principal component for x-axis (1-indexed)
        pc_y : int, default=2
            Principal component for y-axis (1-indexed)
        color_by_stimulus : bool, default=True
            Whether to color points by stimulus type
        figsize : tuple, default=(12, 9)
            Figure size
        use_compound_names : bool, default=True
            Whether to use compound names from compound_info.json for labels
        show_legend : bool, default=True
            Whether to show the legend
        """
        if not self.pca_results:
            raise ValueError("PCA not performed yet. Call perform_pca() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        transformed_data = self.pca_results['transformed_data']
        pc_x_idx = pc_x - 1
        pc_y_idx = pc_y - 1
        
        if color_by_stimulus:
            unique_stimuli = list(set(self.stimulus_labels))
            
            for stimulus in unique_stimuli:
                mask = np.array(self.stimulus_labels) == stimulus
                
                # Get compound name and color
                if use_compound_names and stimulus in self.compound_info:
                    label = self.compound_info[stimulus]
                else:
                    label = stimulus
                
                # Get color from color scheme, fallback to default colors
                if stimulus in self.color_scheme:
                    color = self.color_scheme[stimulus]
                else:
                    # Fallback to matplotlib's default color cycle
                    color_idx = len([s for s in unique_stimuli[:unique_stimuli.index(stimulus)]]) % 10
                    color = plt.cm.tab10(color_idx)
                
                ax.scatter(transformed_data[mask, pc_x_idx], 
                          transformed_data[mask, pc_y_idx],
                          c=color, label=label, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            if show_legend:
                # Adjust legend position and style
                legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                                 fontsize=8, frameon=True, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
        else:
            ax.scatter(transformed_data[:, pc_x_idx], transformed_data[:, pc_y_idx], 
                      alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        var_x = self.pca_results['explained_variance_ratio'][pc_x_idx] * 100
        var_y = self.pca_results['explained_variance_ratio'][pc_y_idx] * 100
        
        ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}% variance)', fontsize=12)
        ax.set_title('PCA Scatter Plot - Tea Compound Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_compound_summary(self):
        """
        Get a summary of compounds and their information.
        
        Returns:
        --------
        summary : pd.DataFrame
            DataFrame with compound codes, names, and colors
        """
        if not self.compound_info:
            print("No compound information loaded.")
            return None
        
        summary_data = []
        for code, name in self.compound_info.items():
            color = self.color_scheme.get(code, '#000000')  # Default to black if no color
            summary_data.append({
                'Compound_Code': code,
                'Compound_Name': name,
                'Color': color
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def plot_compound_color_reference(self, figsize=(12, 8)):
        """
        Plot a reference chart showing compound codes, names, and colors.
        """
        if not self.compound_info:
            print("No compound information loaded.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        compounds = list(self.compound_info.keys())
        y_positions = range(len(compounds))
        
        for i, compound in enumerate(compounds):
            name = self.compound_info[compound]
            color = self.color_scheme.get(compound, '#000000')
            
            # Plot colored rectangle
            ax.barh(i, 1, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add compound code and name
            ax.text(0.5, i, f'{compound}: {name}', 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        ax.set_xlim(0, 1)
        ax.set_xlabel('')
        ax.set_title('Compound Color Reference', fontsize=14, fontweight='bold')
        
        # Remove spines and ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_component_heatmap(self, components_to_plot=None, figsize=(15, 10)):
        """
        Plot heatmap of principal components showing contribution of each neuron over time.
        
        Parameters:
        -----------
        components_to_plot : list, default=None
            List of component indices to plot (0-indexed). If None, plot first 5 components.
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if not self.pca_results:
            raise ValueError("PCA not performed yet. Call perform_pca() first.")
        
        if components_to_plot is None:
            components_to_plot = list(range(min(5, self.pca_results['components'].shape[0])))
        
        n_neurons = len(self.neuron_names)
        n_timepoints = len(self.time_points)
        n_components = len(components_to_plot)
        
        fig, axes = plt.subplots(n_components, 1, figsize=figsize, 
                                sharex=True, sharey=False)
        if n_components == 1:
            axes = [axes]
        
        for i, comp_idx in enumerate(components_to_plot):
            # Reshape component to (neurons, timepoints)
            component_matrix = self.pca_results['components'][comp_idx].reshape(n_neurons, n_timepoints)
            
            im = axes[i].imshow(component_matrix, aspect='auto', cmap='RdBu_r', 
                               extent=[self.time_points[0], self.time_points[-1], 
                                      n_neurons-0.5, -0.5])
            
            axes[i].set_ylabel('Neuron Index')
            axes[i].set_yticks(range(n_neurons))
            axes[i].set_yticklabels(self.neuron_names, fontsize=8)
            
            var_explained = self.pca_results['explained_variance_ratio'][comp_idx] * 100
            axes[i].set_title(f'PC{comp_idx+1} ({var_explained:.1f}% variance)')
            
            plt.colorbar(im, ax=axes[i], label='Component Weight')
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_component_contributions(self, component_idx=0, top_n=10):
        """
        Get the top contributing features (neuron-timepoint pairs) for a specific component.
        
        Parameters:
        -----------
        component_idx : int, default=0
            Index of the component to analyze (0-indexed)
        top_n : int, default=10
            Number of top contributing features to return
            
        Returns:
        --------
        contributions : list
            List of tuples (neuron_name, timepoint, weight)
        """
        if not self.pca_results:
            raise ValueError("PCA not performed yet. Call perform_pca() first.")
        
        component = self.pca_results['components'][component_idx]
        n_neurons = len(self.neuron_names)
        n_timepoints = len(self.time_points)
        
        # Get absolute weights and their indices
        abs_weights = np.abs(component)
        top_indices = np.argsort(abs_weights)[-top_n:][::-1]
        
        contributions = []
        for idx in top_indices:
            neuron_idx = idx // n_timepoints
            time_idx = idx % n_timepoints
            
            neuron_name = self.neuron_names[neuron_idx]
            timepoint = self.time_points[time_idx]
            weight = component[idx]
            
            contributions.append((neuron_name, timepoint, weight))
        
        return contributions
    
    def save_results(self, filepath):
        """
        Save PCA results to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the results
        """
        if not self.pca_results:
            raise ValueError("PCA not performed yet. Call perform_pca() first.")
        
        results_to_save = {
            'arranged_data': self.arranged_data,
            'stimulus_labels': self.stimulus_labels,
            'neuron_names': self.neuron_names,
            'time_points': self.time_points,
            'pca_results': {
                'transformed_data': self.pca_results['transformed_data'],
                'explained_variance_ratio': self.pca_results['explained_variance_ratio'],
                'cumulative_variance_ratio': self.pca_results['cumulative_variance_ratio'],
                'components': self.pca_results['components'],
                'n_components': self.pca_results['n_components'],
                'standardized': self.pca_results['standardized']
            }
        }
        
        np.savez_compressed(filepath, **results_to_save)
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath):
        """
        Load PCA results from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the results from
        """
        loaded = np.load(filepath, allow_pickle=True)
        
        self.arranged_data = loaded['arranged_data']
        self.stimulus_labels = loaded['stimulus_labels'].tolist()
        self.neuron_names = loaded['neuron_names'].tolist()
        self.time_points = loaded['time_points']
        
        pca_results_loaded = loaded['pca_results'].item()
        self.pca_results = pca_results_loaded
        
        print(f"Results loaded from {filepath}")
        print(f"Data shape: {self.arranged_data.shape}")
        print(f"Number of components: {self.pca_results['n_components']}")
    
    def get_stimulus_averaged_responses(self, use_zscore=False, time_window=None, 
                                      include_stats=True):
        """
        Get averaged responses for each stimulus with optional statistics.
        
        Parameters:
        -----------
        use_zscore : bool, default=False
            Whether to use z-scored data or deltaFoverF_0
        time_window : tuple, default=None
            Time window (start, end) in seconds relative to stimulus onset
        include_stats : bool, default=True
            Whether to include SEM (standard error of mean) and trial counts
            
        Returns:
        --------
        stimulus_responses : dict
            Dictionary with stimulus names as keys and response data as values.
            Each value contains:
            - 'mean': averaged response (neurons x timepoints)
            - 'sem': standard error of mean (if include_stats=True)
            - 'n_trials': number of trials averaged (if include_stats=True)
            - 'trial_ids': list of trial identifiers used
        """
        
        # First arrange data without averaging to collect all trials
        self.arrange_data(use_zscore=use_zscore, time_window=time_window, 
                         average_trials=False)
        
        all_neurons = self.neuron_names
        all_stimuli = sorted(set(self.stimulus_labels))
        n_timepoints = len(self.time_points)
        n_neurons = len(all_neurons)
        
        stimulus_responses = {}
        
        # Group samples by stimulus
        for stimulus in all_stimuli:
            # Get indices for this stimulus
            stimulus_indices = [i for i, label in enumerate(self.stimulus_labels) 
                              if label == stimulus]
            
            # Extract data for this stimulus
            stimulus_data = self.arranged_data[stimulus_indices]  # (n_trials, n_features)
            
            # Reshape to (n_trials, n_neurons, n_timepoints)
            stimulus_data_reshaped = stimulus_data.reshape(
                len(stimulus_indices), n_neurons, n_timepoints
            )
            
            # Calculate mean across trials
            mean_response = np.mean(stimulus_data_reshaped, axis=0)  # (n_neurons, n_timepoints)
            
            stimulus_responses[stimulus] = {
                'mean': mean_response,
                'n_trials': len(stimulus_indices)
            }
            
            if include_stats:
                # Calculate standard error of mean
                sem_response = np.std(stimulus_data_reshaped, axis=0, ddof=1) / np.sqrt(len(stimulus_indices))
                stimulus_responses[stimulus]['sem'] = sem_response
                
                # Store trial information
                trial_ids = [f"trial_{i}" for i in stimulus_indices]
                stimulus_responses[stimulus]['trial_ids'] = trial_ids
        
        print(f"Created averaged responses for {len(stimulus_responses)} stimuli:")
        for stimulus, data in stimulus_responses.items():
            print(f"  {stimulus}: {data['n_trials']} trials averaged")
        
        return stimulus_responses
    
    def plot_stimulus_averaged_heatmap(self, stimulus_responses=None, 
                                     use_zscore=False, time_window=None,
                                     figsize=(15, 10), show_sem=True):
        """
        Plot heatmap of averaged stimulus responses.
        
        Parameters:
        -----------
        stimulus_responses : dict, default=None
            Pre-computed stimulus responses. If None, will compute them.
        use_zscore : bool, default=False
            Whether to use z-scored data (only used if stimulus_responses is None)
        time_window : tuple, default=None
            Time window for analysis (only used if stimulus_responses is None)
        figsize : tuple, default=(15, 10)
            Figure size
        show_sem : bool, default=True
            Whether to show error bars representing SEM
        """
        
        if stimulus_responses is None:
            stimulus_responses = self.get_stimulus_averaged_responses(
                use_zscore=use_zscore, time_window=time_window, include_stats=True
            )
        
        stimuli = sorted(stimulus_responses.keys())
        n_stimuli = len(stimuli)
        n_neurons = len(self.neuron_names)
        
        fig, axes = plt.subplots(n_stimuli, 1, figsize=figsize, 
                                sharex=True, sharey=False)
        if n_stimuli == 1:
            axes = [axes]
        
        # Find global min/max for consistent color scale
        all_means = [stimulus_responses[stim]['mean'] for stim in stimuli]
        vmin = np.min([np.min(mean) for mean in all_means])
        vmax = np.max([np.max(mean) for mean in all_means])
        
        for i, stimulus in enumerate(stimuli):
            mean_response = stimulus_responses[stimulus]['mean']
            n_trials = stimulus_responses[stimulus]['n_trials']
            
            # Plot heatmap
            im = axes[i].imshow(mean_response, aspect='auto', cmap='RdBu_r',
                               vmin=vmin, vmax=vmax,
                               extent=[self.time_points[0], self.time_points[-1], 
                                      n_neurons-0.5, -0.5])
            
            axes[i].set_ylabel('Neuron')
            axes[i].set_yticks(range(n_neurons))
            axes[i].set_yticklabels(self.neuron_names, fontsize=8)
            
            title = f'{stimulus} (n={n_trials})'
            axes[i].set_title(title)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], label='Response')
        
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('Stimulus-Averaged Neural Responses', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage function
def example_usage():
    """
    Example of how to use the EpochPCA class with compound information and color schemes.
    """
    # Assuming you have your neuron_segments_dict ready
    # neuron_segments_dict = {...}  # Your data dictionary
    
    # Initialize PCA analyzer with compound information
    # pca_analyzer = EpochPCA(neuron_segments_dict)  # Will auto-load compound info
    # Or specify custom config path:
    # pca_analyzer = EpochPCA(neuron_segments_dict, config_path='/path/to/config')
    
    # View compound information
    # summary = pca_analyzer.get_compound_summary()
    # print(summary)
    
    # Plot compound color reference
    # pca_analyzer.plot_compound_color_reference()
    
    # APPROACH 1: Individual trials as samples (default)
    # This keeps each trial as a separate sample - good for understanding trial-to-trial variability
    # arranged_data, stimulus_labels = pca_analyzer.arrange_data(
    #     use_zscore=True, 
    #     time_window=(-1, 5),
    #     average_trials=False  # Keep individual trials
    # )
    
    # APPROACH 2: Averaged stimulus responses as samples (RECOMMENDED for tea compounds)
    # This averages all trials for each stimulus - good for comparing stimulus types
    # arranged_data, stimulus_labels = pca_analyzer.arrange_data(
    #     use_zscore=True, 
    #     time_window=(-1, 5),
    #     average_trials=True  # Average trials per stimulus
    # )
    
    # APPROACH 3: Get detailed stimulus-averaged responses with statistics
    # stimulus_responses = pca_analyzer.get_stimulus_averaged_responses(
    #     use_zscore=True, 
    #     time_window=(-1, 5),
    #     include_stats=True
    # )
    # pca_analyzer.plot_stimulus_averaged_heatmap(stimulus_responses)
    
    # Perform PCA (same regardless of approach)
    # pca_results = pca_analyzer.perform_pca(n_components=20, standardize=True)
    
    # Plot results with compound names and colors
    # pca_analyzer.plot_explained_variance()
    # 
    # # PCA scatter plot with compound information
    # pca_analyzer.plot_pca_scatter(
    #     pc_x=1, pc_y=2, 
    #     use_compound_names=True,  # Use compound names instead of codes
    #     show_legend=True
    # )
    # 
    # # Try different PC combinations
    # pca_analyzer.plot_pca_scatter(pc_x=1, pc_y=3, use_compound_names=True)
    # pca_analyzer.plot_pca_scatter(pc_x=2, pc_y=3, use_compound_names=True)
    
    # pca_analyzer.plot_component_heatmap(components_to_plot=[0, 1, 2])
    
    # Get top contributions for first component
    # contributions = pca_analyzer.get_component_contributions(component_idx=0, top_n=10)
    # print("Top contributions for PC1:")
    # for neuron, time, weight in contributions:
    #     print(f"  {neuron} at {time:.2f}s: {weight:.3f}")
    
    # Save results
    # pca_analyzer.save_results('pca_results.npz')
    
    pass

