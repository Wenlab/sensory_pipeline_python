import numpy as np

#%%
# need improved
def calculate_time_to_segment_half_peak(
    neuron_segments_dict, 
    stim_onset_in_segment=30, 
    stim_duration_for_peak_detection=50
):
    """
    Calculates the time it takes for each neuron's response in each segment (trial)
    to reach half of that segment's peak value, measured from stimulus onset.

    Args:
        neuron_segments_dict (dict): Dictionary with structure 
                                     {neuron_id: {stimulus_type: [segment_data, ...]}}.
                                     Each segment_data is a dict with 'deltaFoverF_0' (np.array).
        stim_onset_in_segment (int): The index within each segment's 'deltaFoverF_0'
                                     trace where the stimulus begins.
        stim_duration_for_peak_detection (int): The duration (in frames) from stimulus
                                                onset used to determine the segment's peak response.

    Returns:
        dict: A dictionary with the same structure as neuron_segments_dict, but
              containing the time (in frames from stim_onset_in_segment) to reach
              half-peak for each segment. Values will be None if half-peak is not
              reached or if the peak is zero.
    """
    time_to_half_peak_results = {}

    for neuron_id, stimulus_data in neuron_segments_dict.items():
        time_to_half_peak_results[neuron_id] = {}
        for stimulus_type, segments in stimulus_data.items():
            time_to_half_peak_results[neuron_id][stimulus_type] = []
            
            for segment in segments:
                trace = segment.get('deltaFoverF_0')
                time_to_reach = None

                if trace is None or len(trace) == 0:
                    time_to_half_peak_results[neuron_id][stimulus_type].append(None)
                    continue

                # Ensure trace is long enough for peak detection period
                stim_peak_end_idx = stim_onset_in_segment + stim_duration_for_peak_detection
                if len(trace) < stim_peak_end_idx:
                    time_to_half_peak_results[neuron_id][stimulus_type].append(None)
                    # print(f"Warning: Trace too short for neuron {neuron_id}, stim {stimulus_type}")
                    continue

                # Determine the peak response within the specified stimulus window for this segment
                stim_period_trace_for_peak = trace[stim_onset_in_segment : stim_peak_end_idx]
                
                if len(stim_period_trace_for_peak) == 0:
                    time_to_half_peak_results[neuron_id][stimulus_type].append(None)
                    continue

                max_pos_in_stim = np.max(stim_period_trace_for_peak)
                min_neg_in_stim = np.min(stim_period_trace_for_peak)

                segment_peak_response = 0
                if abs(max_pos_in_stim) >= abs(min_neg_in_stim):
                    segment_peak_response = max_pos_in_stim
                else:
                    segment_peak_response = min_neg_in_stim
                
                if segment_peak_response == 0:
                    time_to_half_peak_results[neuron_id][stimulus_type].append(None)
                    continue

                half_peak_target = segment_peak_response / 2.0
                
                # Search for half-peak from stimulus onset onwards in the segment trace
                trace_to_search = trace[stim_onset_in_segment:]
                indices_reaching_target = []

                if segment_peak_response > 0:
                    # For positive peaks, find first time >= half_peak_target
                    potential_indices = np.where(trace_to_search >= half_peak_target)[0]
                    if len(potential_indices) > 0:
                        indices_reaching_target = potential_indices
                else: # segment_peak_response < 0
                    # For negative peaks, find first time <= half_peak_target
                    potential_indices = np.where(trace_to_search <= half_peak_target)[0]
                    if len(potential_indices) > 0:
                        indices_reaching_target = potential_indices
                
                if len(indices_reaching_target) > 0:
                    time_to_reach = indices_reaching_target[0] 
                    # This index is relative to the start of trace_to_search, 
                    # which is stim_onset_in_segment. So, it's the time from stimulus onset.
                
                time_to_half_peak_results[neuron_id][stimulus_type].append(time_to_reach)
                
    return time_to_half_peak_results