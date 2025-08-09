import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import linregress
from scipy.signal import find_peaks

class ResponseFeatureExtraction:
    def __init__(self, neuron_segments_dict):
        self.data = neuron_segments_dict
        self.features = {}

    def extract_features(self, trace, start_time=5, end_time=14, post_stimulus_window=5):
        """
        Extract features from a single trial trace.
        """
        trace = np.array(trace)

        # define time windows
        baseline_end_idx = int(start_time)
        stimulus_start_idx = baseline_end_idx
        stimulus_end_idx = int(end_time)
        post_stimulus_start_idx = stimulus_end_idx
        post_stimulus_end_idx = min(len(trace), int(stimulus_end_idx + post_stimulus_window))

        # Extract Segments
        baseline_segment = trace[:baseline_end_idx]
        stimulus_segment = trace[stimulus_start_idx:stimulus_end_idx]
        post_stimulus_segment = trace[post_stimulus_start_idx:post_stimulus_end_idx]
        analysis_window = trace[stimulus_start_idx:post_stimulus_end_idx]

        features = {}

        # Basic statistics
        features['basic_stats'] = {
            'baseline_mean': np.mean(baseline_segment),
            'baseline_std': np.std(baseline_segment),
            'stimulus_mean': np.mean(stimulus_segment),
            'stimulus_std': np.std(stimulus_segment),
            'post_stimulus_mean': np.mean(post_stimulus_segment),
            'post_stimulus_std': np.std(post_stimulus_segment),
        }

        # Response Type (Response Timing and Polarity)
        response_type = self._determine_response_type(baseline_segment, stimulus_segment, post_stimulus_segment)
        features['response_type'] = response_type

        # Biphasic Response Detection
        if response_type['response_timing'] == 'ON':
            features['response_type']['biphasic_response'] = self._detect_biphasic_response(stimulus_segment, post_stimulus_segment, 
                                                                          features['basic_stats']['baseline_mean'], 
                                                                          features['basic_stats']['baseline_std'])
        else:
            features['response_type']['biphasic_response'] = False

        # Peak Response
        if response_type['response_timing'] == 'OFF':
            features['peak_response'] = {
                'max_response': np.max(post_stimulus_segment),
                'min_response': np.min(post_stimulus_segment),
                'peak_response': np.max(post_stimulus_segment) if features['response_type']['response_polarity'] == 'excitatory' else np.min(post_stimulus_segment),
            }
        else:
            features['peak_response'] = {
                'max_response': np.max(stimulus_segment),
                'min_response': np.min(stimulus_segment),
                'peak_response': np.max(stimulus_segment) if features['response_type']['response_polarity'] == 'excitatory' else np.min(stimulus_segment),
            }

        # Area Under Curve (AUC)
        if features['response_type']['response_timing'] == 'OFF':
            features['auc'] = np.trapezoid(post_stimulus_segment - features['basic_stats']['baseline_mean'])
        else:
            features['auc'] = np.trapezoid(stimulus_segment - features['basic_stats']['baseline_mean'])
        
        # Time-to_Peak
        if features['response_type']['response_timing'] == 'OFF':
            if features['response_type']['response_polarity'] == 'inhibitory':
                features['time_to_peak'] = np.argmin(post_stimulus_segment)
            else:
                features['time_to_peak'] =  np.argmax(post_stimulus_segment)
        elif features['response_type']['response_timing'] == 'ON':
            if features['response_type']['response_polarity'] == 'inhibitory':
                features['time_to_peak'] = np.argmin(stimulus_segment)
            else:
                features['time_to_peak'] = np.argmax(stimulus_segment)
        else:
            features['time_to_peak'] = None

        # respond time
        if features['response_type']['response_timing'] == 'ON':
            if features['response_type']['biphasic_response']:
                amplitude = features['peak_response']['peak_response'] - features['basic_stats']['baseline_mean']
                features['rise_time'] = self._calculate_rise_time(stimulus_segment, features['basic_stats']['baseline_mean'], amplitude)
                features['decay_time'] = self._calculate_decay_time(stimulus_segment, features['basic_stats']['baseline_mean'], amplitude)
            else:
                amplitude = features['peak_response']['peak_response'] - features['basic_stats']['baseline_mean']
                features['rise_time'] = self._calculate_rise_time(stimulus_segment, features['basic_stats']['baseline_mean'], amplitude)
                features['decay_time'] = self._calculate_decay_time(post_stimulus_segment, features['basic_stats']['baseline_mean'], amplitude)
        elif features['response_type']['response_timing'] == 'OFF':
            amplitude = features['peak_response']['peak_response'] - features['basic_stats']['baseline_mean']
            features['rise_time'] = self._calculate_rise_time(post_stimulus_segment, features['basic_stats']['baseline_mean'], amplitude)
            features['decay_time'] = self._calculate_decay_time(post_stimulus_segment[features['time_to_peak']:], features['basic_stats']['baseline_mean'], amplitude)
        else:
            features['rise_time'] = None
            features['decay_time'] = None

        # Signal-to-Noise Ratio
        if features['response_type']['response_timing'] == 'OFF':
            features['snr'] = self._calculate_snr(post_stimulus_segment, features['basic_stats']['baseline_mean'], features['basic_stats']['baseline_std'])
        elif features['response_type']['response_timing'] == 'ON':
            features['snr'] = self._calculate_snr(stimulus_segment, features['basic_stats']['baseline_mean'], features['basic_stats']['baseline_std'])
        else:
            features['snr'] = None
    
        return features


    def _determine_response_type(self, baseline_segment, stimulus_segment, post_stimulus_segment):
        """
        Determine the type of response based on the stimulus segment and analysis window.
        """
        response_type = {}
        slope_threshold = np.std(baseline_segment) * 0.05

        # get slope of a segment
        def segment_slope(segment):
            if len(segment) < 2:
                return 0
            x = np.arange(len(segment))
            slope, _, _, _, _ = linregress(x, segment)
            return slope
        
        # Significance test (stimulus fist and post-stimulus second)
        if len(stimulus_segment) > 1 and len(baseline_segment) > 1 :
            _, p_stim = ttest_ind(baseline_segment, stimulus_segment, equal_var=False)
        
        else:
            p_stim = 1.0
        
        if p_stim >= 0.05 and len(post_stimulus_segment) > 1 and len(baseline_segment) > 1:
            _, p_post = ttest_ind(baseline_segment, post_stimulus_segment, equal_var=False)
        else:
            p_post = 1.0

        response_type['p_stim'] = p_stim
        response_type['p_post'] = p_post

        if p_stim < 0.05:
            slope = segment_slope(stimulus_segment)
            response_type['response_timing'] = 'ON'
            if abs(slope) > slope_threshold:
                response_type['response_polarity'] = 'excitatory' if slope > 0 else 'inhibitory'
            else:
                response_type['response_polarity'] = 'no_clear_trend'
        
        elif p_post < 0.05:
            slope = segment_slope(post_stimulus_segment)
            response_type['response_timing'] = 'OFF'
            if abs(slope) > slope_threshold:
                response_type['response_polarity'] = 'excitatory' if slope > 0 else 'inhibitory'
            else:
                response_type['response_polarity'] = 'no_clear_trend'
        else:
            response_type['response_polarity'] = 'no_response'
            response_type['response_timing'] = 'no_response'
        return response_type
    
    def _detect_biphasic_response(self, stimulus_segment, post_stimulus_segment, baseline_mean, baseline_std, height_factor=0.1):
        """
        Detect if the response is biphasic based on the stimulus and post-stimulus segments.
        """
        height_th = height_factor * baseline_std + baseline_mean

        stim_peaks, _ = find_peaks(stimulus_segment, height=height_th)
        post_peaks, _ = find_peaks(post_stimulus_segment, height=height_th)

        return len(stim_peaks) > 0 and len(post_peaks) > 0
    
    def _calculate_rise_time(self, segment, baseline, amplitude):
        """
        Calculate the rise time of a segment.
        """
        if len(segment) < 2:
            return None
        
        if amplitude > 0:
            idx_10 = np.where(segment >= baseline + 0.1 * amplitude)[0]
            idx_90 = np.where(segment >= baseline + 0.9 * amplitude)[0]
        else:
            idx_10 = np.where(segment <= baseline + 0.1 * amplitude)[0]
            idx_90 = np.where(segment <= baseline + 0.9 * amplitude)[0]

        if len(idx_10) > 0 and len(idx_90) > 0:
            return (idx_90[0] - idx_10[0])
        return None

    def _calculate_decay_time(self, segment, baseline, amplitude):
        """
        Calculate the decay time of a segment.
        """
        if len(segment) < 2:
            return None

        if amplitude > 0:
            idx_90 = np.where(segment <= baseline + 0.9 * amplitude)[0]
            idx_10 = np.where(segment <= baseline + 0.1 * amplitude)[0]
        else:
            idx_90 = np.where(segment >= baseline + 0.9 * amplitude)[0]
            idx_10 = np.where(segment >= baseline + 0.1 * amplitude)[0]

        if len(idx_90) > 0 and len(idx_10) > 0:
            return (idx_10[0] - idx_90[0])
        return None

    def _calculate_snr(self, segment, baseline_mean, baseline_std):
        """
        Calculate the Signal-to-Noise Ratio (SNR) of a segment.
        """
        signal = np.mean(segment) - baseline_mean
        noise = baseline_std if baseline_std != 0 else 1e-6
        return signal / noise
    
    def extract_all_features(self):
        """
        Extract features for all trials in the dataset.
        """
        for neuron_name, stimulus_data in self.data.items():
            self.features[neuron_name] = {}
            for stimulus_name, trials in stimulus_data.items():
                trial_features = []
                for trial in trials:
                    trace = trial['deltaFoverF_0']
                    features = self.extract_features(
                        trace, trial['start_time'], trial['end_time']
                    )
                    
                    features.update({
                        'neuron_name': neuron_name,
                        'stimulus_name': stimulus_name,
                        'trial_id': trial['trial_id']
                    })
                    trial_features.append(features)
                self.features[neuron_name][stimulus_name] = trial_features
