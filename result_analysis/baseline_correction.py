import numpy as np
from scipy import stats

def baseline_correction(trace, start_time=5, correction_window=5):
    """
    Param:
    - trace: (T,) shape numpy array of single trial delta_F/F_0 trace
    - start_time: stimulus onset
    - correction_window: time window for baseline correction before stimulus onset
    """
    trace = np.array(trace)
    correct_end_idx = start_time
    correct_start_idx = max(0, int(start_time - correction_window))

    # Extract baseline segment
    baseline_segment = trace[correct_start_idx:correct_end_idx]

    correct_factor = np.mean(baseline_segment)
    corrected_trace = trace - correct_factor
    return corrected_trace, correct_factor

def apply_baseline_correction(neuron_segments_dict, correction_window=5):
    """
    Apply baseline correction to trials in neuron_segments_dict.
    """
    corrected_data = {}
    for neuron_name, stimuli_data in neuron_segments_dict.items():
        corrected_data[neuron_name] = {}

        for stimulus, trials in stimuli_data.items():
            corrected_trials = []
            for trial in trials:
                original_trace = trial.get('deltaFoverF_0', [])
                corrected_trace, correct_factor = baseline_correction(
                    original_trace,
                    start_time=trial.get('start_time', 5),
                    correction_window=correction_window
                )

                corrected_trial = trial.copy()
                corrected_trial['deltaFoverF_0'] = corrected_trace
                corrected_trial['original_deltaFoverF_0'] = original_trace
                corrected_trial['baseline_correction_factor'] = correct_factor
                corrected_trials.append(corrected_trial)

            corrected_data[neuron_name][stimulus] = corrected_trials
    return corrected_data
