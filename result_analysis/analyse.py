import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class CelegansResponseAnalyzer:
    """
    Comprehensive analysis framework for C. elegans calcium imaging responses
    to tea compounds with outlier detection and response characterization.
    """
    
    def __init__(self, neuron_segments_dict, sampling_rate=1.0):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        neuron_segments_dict : dict
            Dictionary structure: {neuron_name: {stimulus_name: [trial_data, ...]}}
        sampling_rate : float
            Sampling rate of the calcium imaging data (Hz)
        """
        self.data = neuron_segments_dict
        self.sampling_rate = sampling_rate
        self.response_features = {}
        self.filtered_responses = {}
        
    def extract_response_features(self, trace, start_time=5, end_time=14, 
                                post_stimulus_window=5):
        """
        Extract comprehensive features from a single response trace.
        
        Parameters:
        -----------
        trace : array-like
            deltaF/F0 trace data
        start_time : float
            Stimulus onset time
        end_time : float
            Stimulus offset time
        post_stimulus_window : float
            Time window after stimulus offset to consider for analysis
        
        Returns:
        --------
        dict : Dictionary containing all extracted features
        """
        trace = np.array(trace)
        
        # Define time windows
        baseline_end_idx = int(start_time * self.sampling_rate)
        stimulus_start_idx = baseline_end_idx
        stimulus_end_idx = int(end_time * self.sampling_rate)
        post_stimulus_end_idx = int((end_time + post_stimulus_window) * self.sampling_rate)
        post_stimulus_end_idx = min(post_stimulus_end_idx, len(trace))
        
        # Extract segments
        baseline = trace[:baseline_end_idx] if baseline_end_idx > 0 else np.array([0])
        stimulus_period = trace[stimulus_start_idx:stimulus_end_idx]
        post_stimulus = trace[stimulus_end_idx:post_stimulus_end_idx]
        analysis_window = trace[stimulus_start_idx:post_stimulus_end_idx]
        
        features = {}
        
        # Basic statistics
        features['baseline_mean'] = np.mean(baseline)
        features['baseline_std'] = np.std(baseline)
        features['stimulus_mean'] = np.mean(stimulus_period)
        features['post_stimulus_mean'] = np.mean(post_stimulus) if len(post_stimulus) > 0 else 0
        
        # Peak response analysis
        features['peak_response'] = np.max(analysis_window)
        features['min_response'] = np.min(analysis_window)
        features['peak_amplitude'] = features['peak_response'] - features['baseline_mean']
        features['min_amplitude'] = features['min_response'] - features['baseline_mean']
        
        # Determine response type
        if abs(features['peak_amplitude']) > abs(features['min_amplitude']):
            primary_amp = features['peak_amplitude']
        else:
            primary_amp = features['min_amplitude']
        features['primary_response_amplitude'] = primary_amp
        features['response_type'] = 'excitatory' if primary_amp > 0 else 'inhibitory'
        
        # Area Under Curve (AUC)
        features['auc_stimulus'] = np.trapezoid(stimulus_period - features['baseline_mean'])
        features['auc_total'] = np.trapezoid(analysis_window - features['baseline_mean'])

        # Time-to-peak analysis
        peak_idx = np.argmax(analysis_window) + stimulus_start_idx
        min_idx = np.argmin(analysis_window) + stimulus_start_idx
        
        if abs(features['peak_amplitude']) > abs(features['min_amplitude']):
            primary_peak_idx = peak_idx
            features['time_to_peak'] = (primary_peak_idx - stimulus_start_idx) / self.sampling_rate
        else:
            primary_peak_idx = min_idx
            features['time_to_peak'] = (primary_peak_idx - stimulus_start_idx) / self.sampling_rate
        
        # Rise time calculation (10% to 90% of peak)
        features['rise_time'] = self._calculate_rise_time(
            trace, stimulus_start_idx, primary_peak_idx, features['baseline_mean'],
            features['primary_response_amplitude']
        )
        
        # Decay time calculation (90% to 10% from peak)
        features['decay_time'] = self._calculate_decay_time(
            trace, primary_peak_idx, features['baseline_mean'],
            features['primary_response_amplitude']
        )
        
        # Biphasic response detection
        features['is_biphasic'] = self._detect_biphasic_response(
            analysis_window, features['baseline_mean'], 
            stimulus_duration=(end_time - start_time)
        )
        
        # Response onset detection
        features['onset_time'] = self._detect_response_onset(
            trace, stimulus_start_idx, features['baseline_mean'], features['baseline_std']
        )
        
        # Signal-to-noise ratio
        signal_power = np.var(stimulus_period)
        noise_power = features['baseline_std']**2
        features['snr'] = signal_power / noise_power if noise_power > 0 else 0
        
        # Response consistency metrics
        features['response_stability'] = 1 - (np.std(stimulus_period) / 
                                            (abs(np.mean(stimulus_period)) + 1e-6))
        
        return features
     
    def _calculate_rise_time(self, trace, start_idx, peak_idx, baseline, amplitude):
        """Calculate rise time (10% to 90% of peak amplitude)"""
        if peak_idx <= start_idx or abs(amplitude) < 1e-6:
            return np.nan
        
        target_10 = baseline + 0.1 * amplitude
        target_90 = baseline + 0.9 * amplitude
        
        rise_segment = trace[start_idx:peak_idx+1]
        
        # Find 10% and 90% crossing points
        if amplitude > 0:  # Excitatory response
            idx_10 = np.where(rise_segment >= target_10)[0]
            idx_90 = np.where(rise_segment >= target_90)[0]
        else:  # Inhibitory response
            idx_10 = np.where(rise_segment <= target_10)[0]
            idx_90 = np.where(rise_segment <= target_90)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            return (idx_90[0] - idx_10[0]) / self.sampling_rate
        
        return np.nan
    
    def _calculate_decay_time(self, trace, peak_idx, baseline, amplitude):
        """Calculate decay time (90% to 10% from peak)"""
        if peak_idx >= len(trace) - 1 or abs(amplitude) < 1e-6:
            return np.nan
        
        target_90 = baseline + 0.9 * amplitude
        target_10 = baseline + 0.1 * amplitude
        
        decay_segment = trace[peak_idx:]
        
        # Find 90% and 10% crossing points
        if amplitude > 0:  # Excitatory response
            idx_90 = np.where(decay_segment <= target_90)[0]
            idx_10 = np.where(decay_segment <= target_10)[0]
        else:  # Inhibitory response
            idx_90 = np.where(decay_segment >= target_90)[0]
            idx_10 = np.where(decay_segment >= target_10)[0]
        
        if len(idx_90) > 0 and len(idx_10) > 0:
            return (idx_10[0] - idx_90[0]) / self.sampling_rate
        
        return np.nan
    
    def _detect_biphasic_response(self, analysis_window, baseline, threshold_ratio=0.3,
                                stimulus_duration=9):
        """
        Detect biphasic responses (2 excitatory peaks):
        One peak during stimulus period and another peak during post-stimulus period
        
        Parameters:
        -----------
        analysis_window : array-like
            Combined stimulus + post-stimulus response data
        baseline : float
            Baseline fluorescence level
        threshold_ratio : float
            Minimum ratio of secondary peak to primary peak amplitude
        stimulus_duration : float
            Duration of stimulus period in seconds (default 9s: from 5s to 14s)
        """
        # Split analysis window into stimulus and post-stimulus periods
        stimulus_end_idx = int(stimulus_duration * self.sampling_rate)
        stimulus_end_idx = min(stimulus_end_idx, len(analysis_window))
        
        stimulus_period = analysis_window[:stimulus_end_idx]
        post_stimulus_period = analysis_window[stimulus_end_idx:]
        
        if len(post_stimulus_period) == 0:
            return False
        
        # Find peaks in both periods (using scipy.signal.find_peaks for better detection)
        # Set minimum height above baseline to avoid noise
        min_height = baseline + 0.1 * np.std(analysis_window)
        
        # Find peaks in stimulus period
        stimulus_peaks, _ = find_peaks(stimulus_period, height=min_height, distance=int(1.0 * self.sampling_rate))
        
        # Find peaks in post-stimulus period  
        post_peaks, _ = find_peaks(post_stimulus_period, height=min_height, distance=int(1.0 * self.sampling_rate))
        
        # Get maximum peak amplitudes in each period
        if len(stimulus_peaks) > 0:
            stimulus_peak_amp = np.max(stimulus_period[stimulus_peaks]) - baseline
        else:
            stimulus_peak_amp = 0
            
        if len(post_peaks) > 0:
            post_peak_amp = np.max(post_stimulus_period[post_peaks]) - baseline
        else:
            post_peak_amp = 0
        
        # Check if both periods have significant excitatory peaks
        if stimulus_peak_amp > 0 and post_peak_amp > 0:
            # Determine primary and secondary peaks
            primary_amp = max(stimulus_peak_amp, post_peak_amp)
            secondary_amp = min(stimulus_peak_amp, post_peak_amp)
            
            # Check if secondary peak is significant relative to primary peak
            return secondary_amp > threshold_ratio * primary_amp
        
        return False
    
    def _detect_response_onset(self, trace, stimulus_start_idx, baseline, baseline_std, 
                             threshold_factor=2):
        """Detect response onset time"""
        if stimulus_start_idx >= len(trace):
            return np.nan
        
        threshold = threshold_factor * baseline_std
        post_stimulus = trace[stimulus_start_idx:]
        
        # Find first point that exceeds threshold
        significant_points = np.where(np.abs(post_stimulus - baseline) > threshold)[0]
        
        if len(significant_points) > 0:
            return significant_points[0] / self.sampling_rate
        
        return np.nan
    
    def analyze_all_responses(self):
        """Analyze all responses and extract features for each trial"""
        print("Extracting features from all trials...")
        
        for neuron_name, stimuli_data in self.data.items():
            self.response_features[neuron_name] = {}
            
            for stimulus_name, trials in stimuli_data.items():
                trial_features = []
                
                for trial in trials:
                    trace = trial['deltaFoverF_0']
                    features = self.extract_response_features(
                        trace, trial['start_time'], trial['end_time']
                    )
                    
                    # Add trial metadata and original trace
                    features.update({
                        'worm_key': trial['worm_key'],
                        'segment_index': trial['segment_index'],
                        'date': trial['date'],
                        'deltaFoverF_0': trace  # Add original deltaF/F0 trace
                    })
                    
                    trial_features.append(features)
                
                self.response_features[neuron_name][stimulus_name] = trial_features
        
        print(f"Feature extraction completed for {len(self.response_features)} neurons")
    
    def detect_outliers(self, neuron_name, stimulus_name, features_to_use=None, 
                       method='isolation_forest', contamination=0.1):
        """
        Detect outlier trials using multiple methods
        
        Parameters:
        -----------
        neuron_name : str
            Name of the neuron to analyze
        stimulus_name : str
            Name of the stimulus
        features_to_use : list
            List of feature names to use for outlier detection
        method : str
            Outlier detection method ('isolation_forest', 'zscore', 'iqr', 'dbscan')
        contamination : float
            Expected proportion of outliers
        """
        if features_to_use is None:
            features_to_use = [
                'primary_response_amplitude', 'auc_stimulus', 'time_to_peak',
                'rise_time', 'decay_time', 'snr', 'response_stability'
            ]
        
        trials = self.response_features[neuron_name][stimulus_name]
        
        # Create feature matrix
        feature_matrix = []
        for trial in trials:
            row = [trial.get(feat, np.nan) for feat in features_to_use]
            feature_matrix.append(row)
        
        feature_matrix = np.array(feature_matrix)
        
        # Handle NaN values
        valid_mask = ~np.isnan(feature_matrix).any(axis=1)
        valid_features = feature_matrix[valid_mask]
        
        if len(valid_features) < 3:  # Need minimum trials for outlier detection
            return np.zeros(len(trials), dtype=bool)  # No outliers detected
        
        outlier_mask = np.zeros(len(trials), dtype=bool)
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(valid_features, axis=0))
            outliers_valid = np.any(z_scores > 2.5, axis=1)
            
        elif method == 'iqr':
            Q1 = np.percentile(valid_features, 25, axis=0)
            Q3 = np.percentile(valid_features, 75, axis=0)
            IQR = Q3 - Q1
            outliers_valid = np.any(
                (valid_features < (Q1 - 1.5 * IQR)) | 
                (valid_features > (Q3 + 1.5 * IQR)), axis=1
            )
            
        elif method == 'dbscan':
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(valid_features)
            
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(scaled_features)
            outliers_valid = clustering.labels_ == -1
            
        else:  # isolation_forest (requires sklearn)
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outliers_valid = iso_forest.fit_predict(valid_features) == -1
            except ImportError:
                # Fallback to z-score method
                z_scores = np.abs(stats.zscore(valid_features, axis=0))
                outliers_valid = np.any(z_scores > 2.5, axis=1)
        
        # Map back to original indices
        outlier_mask[valid_mask] = outliers_valid
        
        return outlier_mask
    
    def filter_consistent_responses(self, min_trials=3, max_outlier_ratio=0.3):
        """
        Filter responses to keep only consistent, reliable responses
        
        Parameters:
        -----------
        min_trials : int
            Minimum number of trials required after outlier removal
        max_outlier_ratio : float
            Maximum allowed ratio of outliers
        """
        print("Filtering for consistent responses...")
        
        self.filtered_responses = {}
        filtering_summary = []
        
        for neuron_name, stimuli_data in self.response_features.items():
            self.filtered_responses[neuron_name] = {}
            
            for stimulus_name, trials in stimuli_data.items():
                original_count = len(trials)
                
                if original_count < min_trials:
                    continue
                
                # Detect outliers
                outlier_mask = self.detect_outliers(neuron_name, stimulus_name)
                outlier_count = np.sum(outlier_mask)
                outlier_ratio = outlier_count / original_count
                
                # Filter trials
                if outlier_ratio <= max_outlier_ratio:
                    filtered_trials = [trial for i, trial in enumerate(trials) 
                                     if not outlier_mask[i]]
                    
                    if len(filtered_trials) >= min_trials:
                        self.filtered_responses[neuron_name][stimulus_name] = {
                            'trials': filtered_trials,
                            'outliers_removed': outlier_count,
                            'outlier_ratio': outlier_ratio,
                            'n_trials': len(filtered_trials)
                        }
                
                filtering_summary.append({
                    'neuron': neuron_name,
                    'stimulus': stimulus_name,
                    'original_trials': original_count,
                    'outliers_detected': outlier_count,
                    'outlier_ratio': outlier_ratio,
                    'trials_retained': len(filtered_trials) if outlier_ratio <= max_outlier_ratio 
                                     and len(filtered_trials) >= min_trials else 0,
                    'passed_filter': stimulus_name in self.filtered_responses.get(neuron_name, {})
                })
        
        summary_df = pd.DataFrame(filtering_summary)
        print(f"Filtering completed:")
        print(f"- Total neuron-stimulus pairs: {len(summary_df)}")
        print(f"- Passed quality filter: {summary_df['passed_filter'].sum()}")
        print(f"- Average outlier ratio: {summary_df['outlier_ratio'].mean():.3f}")
        
        return summary_df
    
    def get_response_summary(self, neuron_name, stimulus_name):
        """
        Get summary statistics for a filtered neuron-stimulus pair
        """
        if (neuron_name not in self.filtered_responses or 
            stimulus_name not in self.filtered_responses[neuron_name]):
            return None
        
        data = self.filtered_responses[neuron_name][stimulus_name]
        trials = data['trials']
        
        # Calculate summary statistics
        features = [
            'primary_response_amplitude', 'auc_stimulus', 'time_to_peak',
            'rise_time', 'decay_time', 'snr', 'response_stability'
        ]
        
        summary = {
            'n_trials': len(trials),
            'outliers_removed': data['outliers_removed'],
            'response_type': trials[0]['response_type'] if trials else 'unknown'
        }
        
        for feature in features:
            values = [trial.get(feature, np.nan) for trial in trials]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                summary[f'{feature}_mean'] = np.mean(values)
                summary[f'{feature}_std'] = np.std(values)
                summary[f'{feature}_cv'] = np.std(values) / (abs(np.mean(values)) + 1e-6)
            else:
                summary[f'{feature}_mean'] = np.nan
                summary[f'{feature}_std'] = np.nan
                summary[f'{feature}_cv'] = np.nan
        
        return summary
    
    def plot_response_comparison(self, neuron_name, stimulus_name, figsize=(15, 10)):
        """
        Plot comparison between original and filtered responses
        """
        if neuron_name not in self.response_features:
            print(f"Neuron {neuron_name} not found")
            return
        
        if stimulus_name not in self.response_features[neuron_name]:
            print(f"Stimulus {stimulus_name} not found for neuron {neuron_name}")
            return
        
        # Get original trials
        original_trials = self.response_features[neuron_name][stimulus_name]
        
        # Get filtered trials
        filtered_data = self.filtered_responses.get(neuron_name, {}).get(stimulus_name)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'{neuron_name} Response to {stimulus_name}', fontsize=16)
        
        # Plot 1: All original traces
        ax1 = axes[0, 0]
        for i, trial in enumerate(original_trials):
            trace = trial['deltaFoverF_0']
            time_axis = np.arange(len(trace)) / self.sampling_rate
            ax1.plot(time_axis, trace, alpha=0.6, linewidth=1)
        ax1.set_title('All Original Traces')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('ΔF/F₀')
        ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Stimulus On')
        ax1.axvline(x=14, color='blue', linestyle='--', alpha=0.7, label='Stimulus Off')
        ax1.legend()
        
        # Plot 2: Filtered traces (if available)
        ax2 = axes[0, 1]
        if filtered_data:
            for trial in filtered_data['trials']:
                trace = trial['deltaFoverF_0']
                time_axis = np.arange(len(trace)) / self.sampling_rate
                ax2.plot(time_axis, trace, alpha=0.8, linewidth=1.5)
            ax2.set_title(f'Filtered Traces (n={len(filtered_data["trials"])})')
        else:
            ax2.text(0.5, 0.5, 'No trials passed filter', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Filtered Traces (n=0)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('ΔF/F₀')
        ax2.axvline(x=5, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(x=14, color='blue', linestyle='--', alpha=0.7)
        
        # Plot 3: Average response comparison
        ax3 = axes[0, 2]
        # Original average
        original_traces = np.array([trial['deltaFoverF_0'] for trial in original_trials])
        original_mean = np.mean(original_traces, axis=0)
        original_sem = stats.sem(original_traces, axis=0)
        time_axis = np.arange(len(original_mean)) / self.sampling_rate
        
        ax3.plot(time_axis, original_mean, 'gray', linewidth=2, label='Original', alpha=0.8)
        ax3.fill_between(time_axis, original_mean - original_sem, 
                        original_mean + original_sem, alpha=0.3, color='gray')
        
        # Filtered average
        if filtered_data:
            filtered_traces = np.array([trial['deltaFoverF_0'] for trial in filtered_data['trials']])
            filtered_mean = np.mean(filtered_traces, axis=0)
            filtered_sem = stats.sem(filtered_traces, axis=0)
            
            ax3.plot(time_axis, filtered_mean, 'blue', linewidth=2, label='Filtered')
            ax3.fill_between(time_axis, filtered_mean - filtered_sem, 
                            filtered_mean + filtered_sem, alpha=0.3, color='blue')
        
        ax3.set_title('Average Response ± SEM')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('ΔF/F₀')
        ax3.axvline(x=5, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(x=14, color='blue', linestyle='--', alpha=0.7)
        ax3.legend()
        
        # Plot 4: Feature distributions
        ax4 = axes[1, 0]
        feature = 'primary_response_amplitude'
        original_values = [trial.get(feature, np.nan) for trial in original_trials]
        original_values = [v for v in original_values if not np.isnan(v)]
        
        ax4.hist(original_values, bins=10, alpha=0.6, label='Original', color='gray')
        
        if filtered_data:
            filtered_values = [trial.get(feature, np.nan) for trial in filtered_data['trials']]
            filtered_values = [v for v in filtered_values if not np.isnan(v)]
            ax4.hist(filtered_values, bins=10, alpha=0.8, label='Filtered', color='blue')
        
        ax4.set_title('Response Amplitude Distribution')
        ax4.set_xlabel('ΔF/F₀')
        ax4.set_ylabel('Count')
        ax4.legend()
        
        # Plot 5: SNR comparison
        ax5 = axes[1, 1]
        original_snr = [trial.get('snr', np.nan) for trial in original_trials]
        original_snr = [v for v in original_snr if not np.isnan(v)]
        
        if original_snr:
            ax5.boxplot([original_snr], positions=[1], labels=['Original'])
        
        if filtered_data:
            filtered_snr = [trial.get('snr', np.nan) for trial in filtered_data['trials']]
            filtered_snr = [v for v in filtered_snr if not np.isnan(v)]
            if filtered_snr:
                ax5.boxplot([filtered_snr], positions=[2], labels=['Filtered'])
        
        ax5.set_title('Signal-to-Noise Ratio')
        ax5.set_ylabel('SNR')
        
        # Plot 6: Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"Original trials: {len(original_trials)}\n"
        if filtered_data:
            summary_text += f"Filtered trials: {len(filtered_data['trials'])}\n"
            summary_text += f"Outliers removed: {filtered_data['outliers_removed']}\n"
            summary_text += f"Outlier ratio: {filtered_data['outlier_ratio']:.3f}\n"
            
            summary = self.get_response_summary(neuron_name, stimulus_name)
            if summary:
                summary_text += f"\nResponse type: {summary['response_type']}\n"
                summary_text += f"Amplitude CV: {summary['primary_response_amplitude_cv']:.3f}\n"
                summary_text += f"Time-to-peak: {summary['time_to_peak_mean']:.2f} ± {summary['time_to_peak_std']:.2f} s"
        else:
            summary_text += "No trials passed filter"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()


#  usage and analysis pipeline
def run_analysis_pipeline(neuron_segments_dict, sampling_rate=1.0):
    """
    Complete analysis pipeline for C. elegans calcium imaging data
    """
    print("Starting C. elegans response analysis pipeline...")
    
    # Initialize analyzer
    analyzer = CelegansResponseAnalyzer(neuron_segments_dict, sampling_rate)
    
    # Step 1: Extract features from all trials
    analyzer.analyze_all_responses()
    
    # Step 2: Filter for consistent responses
    filtering_summary = analyzer.filter_consistent_responses(
        min_trials=3, max_outlier_ratio=0.3
    )
    
    # Step 3: Generate summary report
    print("\n=== ANALYSIS SUMMARY ===")
    
    # Overall statistics
    total_pairs = len(filtering_summary)
    passed_pairs = filtering_summary['passed_filter'].sum()
    
    print(f"Total neuron-stimulus pairs analyzed: {total_pairs}")
    print(f"Pairs passing quality filter: {passed_pairs} ({passed_pairs/total_pairs*100:.1f}%)")
    
    # Per-neuron summary
    print("\nPer-neuron summary:")
    neuron_summary = filtering_summary.groupby('neuron').agg({
        'passed_filter': 'sum',
        'stimulus': 'count',
        'outlier_ratio': 'mean'
    }).round(3)
    neuron_summary.columns = ['Responsive_Stimuli', 'Total_Stimuli', 'Avg_Outlier_Ratio']
    print(neuron_summary)
    
    # Stimulus responsiveness
    print("\nStimulus responsiveness:")
    stimulus_summary = filtering_summary.groupby('stimulus').agg({
        'passed_filter': 'sum',
        'neuron': 'count',
        'outlier_ratio': 'mean'
    }).round(3)
    stimulus_summary.columns = ['Responsive_Neurons', 'Total_Neurons', 'Avg_Outlier_Ratio']
    print(stimulus_summary)
    
    return analyzer, filtering_summary


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.HDF5_load import load_h5file
    neuron_segments_dict = load_h5file(
            path = r"I:\WJH\flavor\neuron_segments_dict.h5",
            root_name= 'neuron_segments_dict')
    
    # print how many trials in neuron-stimulus pairs
    for neuron, stimuli in neuron_segments_dict.items():
        for stimulus, trials in stimuli.items():
            print(f"{neuron} - {stimulus}: {len(trials)} trials")

    # Run the analysis pipeline
    analyzer, summary = run_analysis_pipeline(neuron_segments_dict, sampling_rate=1.0)

    filtered_awcl_c1_11 = analyzer.filtered_responses['AWCL']['c1_11']
    # Get all reliable neuron-stimulus pairs for sensory coding analysis
    reliable_pairs = []
    for neuron in analyzer.filtered_responses:
        for stimulus in analyzer.filtered_responses[neuron]:
            reliable_pairs.append((neuron, stimulus))

    analyzer.plot_response_comparison('ASHL', 'c1_11')