# %%
import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d
from data_load.get_stimulus_info import get_stimulus_info, extract_intervals_from_excel
from data_load.load_worm_data import load_worm_ID, load_worm_data
from utils.interpolate import interpolate_over_nans
from result_analysis.baseline_correction import apply_baseline_correction
# %%
def extract_neuron_groups(worm_data, vps_setting=1, boundary_method='preserve', pre_segment=5, post_segment=30):
    """
    cut off worm data into neuron groups based on biological_ID and segments.
    Parameters:
    - worm_data: dict
        A dictionary where each key is a worm name and each value is a dictionary containing:
            - 'intensity': DataFrame of intensity values (rows correspond to neurons, columns to time points)
            - 'biological_ID': DataFrame with neuron IDs (single column, indexed by integer)
            - 'stimulus_intervals': list of tuples (start, end) indicating stimulus periods
            - 'buffer_intervals': list of tuples (not used in this function)
            - 'delta_F_over_F': DataFrame of delta_F_over_F values
    - vps_setting: vps when taking photos
    - boundary_method: str, method for boundary preservation in downsampling
        - 'preserve': preserve boundary points exactly
        - 'weighted': weighted averaging with emphasis on boundaries
        - 'adaptive': adaptive method based on signal characteristics
        - 'original': original averaging method
    Returns:
    - neuron_segments_dict: dict
        Dictionary containing segments data for each neuron group and worm
    - neuron_groups: dict
        Dictionary of neuron groups categorized by group keys
    """
    # 1. Parse data to build neuron_groups
    neuron_groups = {}
    for worm_key, data in worm_data.items():
        biological_ID = data["biological_ID"][0].tolist()
        for full_id in biological_ID:
            group_key = full_id
            # if group_key.startswith("A"):
            if not group_key.isdigit():
                if group_key not in neuron_groups:
                    neuron_groups[group_key] = set()
                neuron_groups[group_key].add(full_id)

    # 2. Prepare data structures
    worm_keys = sorted(worm_data.keys())
    num_worms = len(worm_keys)

    neuron_segments_dict = {}
    neuron_group_keys = sorted(neuron_groups.keys())
    # sorted_worm_list = sorted(worm_list, key=lambda x: int(x[1:]))

    # 3. Process each neuron group
    for group_key in neuron_group_keys:
        worm_segments = {}
        group_neurons = neuron_groups[group_key]

        for idx, worm_key in enumerate(worm_keys):
            data = worm_data[worm_key]
            biological_ID = data["biological_ID"][0].tolist()
            delta_F_over_F = data["delta_F_over_F"]
            delta_F_over_F, t_inter = interpolate_over_nans(delta_F_over_F)
            stimulus_intervals = data["stimulus_intervals"]
            worm_stimuli = data["stimulus_list"]
            # worm_stimuli = stimulus_lists.get(worm_key, [])

            group_neurons_in_worm = group_neurons.intersection(biological_ID)
            if not group_neurons_in_worm:
                continue

            full_id = list(group_neurons_in_worm)[0]
            neuron_index = biological_ID.index(full_id)
            delta_F_trace = delta_F_over_F.iloc[neuron_index].values

            # cut stimulus intervals into fixed length segments
            segments_data = []
            for idx, (start_time, end_time) in enumerate(stimulus_intervals):
                start_idx = max(0, start_time - pre_segment*vps_setting)# included
                end_idx = min(len(delta_F_trace), end_time + post_segment*vps_setting)# not included

                # Define fixed stimulus duration
                fixed_stimulus_duration = 10 * vps_setting

                # Split the segment into three parts
                pre_stimulus = delta_F_trace[start_idx:start_time]
                stimulus = delta_F_trace[
                    start_time : start_time + fixed_stimulus_duration
                ]
                post_stimulus = delta_F_trace[end_time:end_idx]

                # downsample the segment using separated sampling method
                if vps_setting > 1:
                    seg_data, relative_start_time, relative_end_time = downsampling_separated_enhanced(
                        pre_stimulus, stimulus, post_stimulus, vps_setting, boundary_method
                    )
                else:
                    # If no downsampling needed, concatenate the parts
                    seg_data = np.concatenate((pre_stimulus, stimulus, post_stimulus))
                    relative_start_time = len(pre_stimulus)
                    relative_end_time = len(pre_stimulus) + len(stimulus)
                
                # Smooth the concatenated segment
                # seg_data_smooth = gaussian_filter1d(seg_data, sigma=1)

                stimulus_type = worm_stimuli[idx]

                segments_data.append(
                    {
                        "stimulus_type": stimulus_type,
                        "deltaFoverF_0": seg_data,
                        "start_time": relative_start_time,  # relative start time after separated downsampling
                        "end_time": relative_end_time,      # relative end time after separated downsampling
                    }
                )
            if not segments_data:
                continue

            worm_segments[worm_key] = segments_data

        if not worm_segments:
            continue

        # use a dict to store detailed segments
        detailed_segments = {}
        for worm_key, segments in worm_segments.items():
            detailed_segments[worm_key] = []
            for seg_idx, segment in enumerate(segments):
                detailed_segments[worm_key].append(
                    {
                        "segment_index": seg_idx,
                        "stimulus_type": segment["stimulus_type"],
                        "deltaFoverF_0": segment["deltaFoverF_0"],
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                    }
                )

        neuron_segments_dict[group_key] = detailed_segments

    return neuron_segments_dict, neuron_groups


def downsampling_with_boundary_preservation(segment_data, vps_setting=5, boundary_method='preserve'):
    """
    Enhanced downsampling function with multiple boundary preservation strategies.
    
    Parameters:
    - segment_data: array-like, input signal data
    - vps_setting: int, downsampling factor
    - boundary_method: str, method for handling boundaries
        - 'preserve': preserve first and last points exactly
        - 'weighted': give more weight to boundary points in averaging
        - 'adaptive': use different strategies based on signal characteristics
        - 'original': standard averaging (original method)
    
    Returns:
    - downsampled_data: numpy array of downsampled signal
    """
    if not isinstance(segment_data, np.ndarray):
        segment_data = np.array(segment_data)
    
    original_length = len(segment_data)
    if original_length <= vps_setting:
        return segment_data.copy()
    
    if boundary_method == 'original':
        # Original method: simple averaging
        downsampled_length = original_length // vps_setting
        trimmed_length = downsampled_length * vps_setting
        trimmed_data = segment_data[:trimmed_length]
        reshaped_data = trimmed_data.reshape(downsampled_length, vps_setting)
        return np.mean(reshaped_data, axis=1)
    
    elif boundary_method == 'preserve':
        # Preserve exact boundary points while maintaining correct output length
        downsampled_length = original_length // vps_setting
        if downsampled_length == 0:
            return np.array([segment_data[0]]) if original_length > 0 else np.array([])
        
        result = np.zeros(downsampled_length)
        
        # Always preserve first point
        result[0] = segment_data[0]
        
        # Always preserve last point (if we have more than one output point)
        if downsampled_length > 1:
            result[-1] = segment_data[-1]
        
        # Fill middle points with averaged values, avoiding boundary regions
        if downsampled_length > 2:
            # Calculate indices for middle points
            middle_indices = np.linspace(1, downsampled_length - 2, downsampled_length - 2, dtype=int)
            
            for i, out_idx in enumerate(middle_indices):
                # Calculate corresponding input range for this middle point
                # Skip boundary regions to avoid double-counting boundary points
                start_input = max(vps_setting, (out_idx * original_length) // downsampled_length)
                end_input = min(original_length - vps_setting, ((out_idx + 1) * original_length) // downsampled_length)
                
                if end_input > start_input:
                    result[out_idx] = np.mean(segment_data[start_input:end_input])
                else:
                    # Fallback: use interpolation between boundaries
                    alpha = (out_idx) / (downsampled_length - 1)
                    result[out_idx] = (1 - alpha) * segment_data[0] + alpha * segment_data[-1]
        
        return result
    
    elif boundary_method == 'weighted':
        # Give more weight to boundary points while maintaining correct length
        downsampled_length = original_length // vps_setting
        result = np.zeros(downsampled_length)
        
        for i in range(downsampled_length):
            start_idx = i * vps_setting
            end_idx = min((i + 1) * vps_setting, original_length)
            window = segment_data[start_idx:end_idx]
            
            if len(window) == vps_setting:
                # Apply weighted average: more weight to edges within each window
                weights = np.ones(vps_setting)
                weights[0] *= 1.5  # First point gets more weight
                weights[-1] *= 1.5  # Last point gets more weight
                weights /= np.sum(weights)  # Normalize
                result[i] = np.average(window, weights=weights)
            else:
                result[i] = np.mean(window)
        
        # Additionally preserve exact boundary points
        if downsampled_length > 0:
            result[0] = segment_data[0]
        if downsampled_length > 1:
            result[-1] = segment_data[-1]
        
        return result
    
    elif boundary_method == 'adaptive':
        # Adaptive method based on signal characteristics with fixed output length
        # Detect rapid changes and preserve them
        gradient = np.abs(np.gradient(segment_data))
        threshold = np.percentile(gradient, 75)  # Top 25% of changes
        
        downsampled_length = original_length // vps_setting
        result = np.zeros(downsampled_length)
        
        # Standard bin-based processing but with adaptive weighting
        for i in range(downsampled_length):
            start_idx = i * vps_setting
            end_idx = min((i + 1) * vps_setting, original_length)
            window = segment_data[start_idx:end_idx]
            window_grad = gradient[start_idx:end_idx]
            
            # If window contains significant changes, use different strategy
            if np.any(window_grad > threshold):
                # For regions with rapid change, preserve extreme values
                if len(window) >= 3:
                    # Weight extremes more heavily
                    min_val = np.min(window)
                    max_val = np.max(window)
                    mean_val = np.mean(window)
                    # Blend between mean and extremes based on gradient magnitude
                    max_grad = np.max(window_grad)
                    extreme_weight = min(max_grad / (threshold * 2), 1.0)
                    if abs(max_val - mean_val) > abs(min_val - mean_val):
                        result[i] = mean_val * (1 - extreme_weight) + max_val * extreme_weight
                    else:
                        result[i] = mean_val * (1 - extreme_weight) + min_val * extreme_weight
                else:
                    result[i] = np.mean(window)
            else:
                # Standard averaging for smooth regions
                result[i] = np.mean(window)
        
        # Ensure boundary preservation
        if downsampled_length > 0:
            result[0] = segment_data[0]
        if downsampled_length > 1:
            result[-1] = segment_data[-1]
        
        return result
    
    else:
        raise ValueError(f"Unknown boundary_method: {boundary_method}")


def downsampling_separated_enhanced(pre_stimulus, stimulus, post_stimulus, vps_setting=5, boundary_method='preserve'):
    """
    Enhanced separate downsampling with configurable boundary preservation.
    
    Parameters:
    - pre_stimulus, stimulus, post_stimulus: array-like, signal segments
    - vps_setting: int, downsampling factor
    - boundary_method: str, boundary preservation method
    
    Returns:
    - seg_data_downsampled: concatenated downsampled signal
    - relative_start_time: start time of stimulus in downsampled signal
    - relative_end_time: end time of stimulus in downsampled signal
    """
    # Apply enhanced downsampling to each segment
    pre_downsampled = downsampling_with_boundary_preservation(
        pre_stimulus, vps_setting, boundary_method
    ) if len(pre_stimulus) > 0 else np.array([])
    
    stimulus_downsampled = downsampling_with_boundary_preservation(
        stimulus, vps_setting, boundary_method
    ) if len(stimulus) > 0 else np.array([])
    
    post_downsampled = downsampling_with_boundary_preservation(
        post_stimulus, vps_setting, boundary_method
    ) if len(post_stimulus) > 0 else np.array([])
    
    # Calculate relative times
    pre_length = len(pre_downsampled)
    stimulus_length = len(stimulus_downsampled)
    
    # Concatenate segments
    seg_data_downsampled = np.concatenate([
        pre_downsampled, stimulus_downsampled, post_downsampled
    ])
    
    return seg_data_downsampled, pre_length, pre_length + stimulus_length - 1

def per_worm_zscore(neuron_segments_dict, group_size=5, if_group=False):
    """
    对每个worm_key对应的所有deltaFoverF_0按group_size进行分组后拼接进行z-score标准化，并记录每个组的均值和标准差。
    Parameters:
    - neuron_segments_dict: dict
        Dictionary containing segments data for each neuron group and worm
        - group_key: neuron group key
            - worm_key: worm key
                - segment_index: segment index
                - stimulus_type: stimulus type
                - deltaFoverF_0: deltaFoverF_0
                - start_time: start time
                - end_time: end time
    - group_size: int, a group contains different stimulus segments
    """
    # 用于存储每个worm_key的均值和标准差
    worm_stats = {}

    for group_key, worm_data in neuron_segments_dict.items():
        for worm_key, segments in worm_data.items():
            # 将所有deltaFoverF_0收集到一个列表中
            all_data = [np.array(seg["deltaFoverF_0"]) for seg in segments]

            # 按照group_size分组并拼接
            if if_group:
                grouped_data = []
                for i in range(0, len(all_data), group_size):
                    # 取出当前组的deltaFoverF_0并进行拼接
                    group = np.concatenate(all_data[i : i + group_size])
                    grouped_data.append(group)

                # 计算拼接后的分组数据的均值和标准差
                group_means = []
                group_stds = []
                for group in grouped_data:
                    mean_val = np.mean(group)
                    std_val = np.std(group, ddof=1) if len(group) > 1 else 1.0
                    group_means.append(mean_val)
                    group_stds.append(std_val if std_val != 0 else 1.0)
            else:
                # 如果不分组，直接计算均值和标准差
                all_data = np.concatenate(all_data)
                mean_val = np.mean(all_data)
                std_val = np.std(all_data, ddof=1) if len(all_data) > 1 else 1.0
                group_means = [mean_val]
                group_stds = [std_val if std_val != 0 else 1.0]

            # 对每个deltaFoverF_0进行z-score标准化
            for i, seg in enumerate(segments):
                # 计算当前segment的索引
                group_idx = i // group_size if if_group else 0
                seg_data = np.array(seg["deltaFoverF_0"])

                # 获取对应组的均值和标准差
                mean_val = group_means[group_idx]
                std_val = group_stds[group_idx]

                seg["original_mean"] = mean_val
                seg["original_std"] = std_val
                seg["z_scored"] = (seg_data - mean_val) / std_val

            # 记录worm_key的均值和标准差
            worm_stats[worm_key] = {"mean": group_means, "std": group_stds}

    return neuron_segments_dict, worm_stats


def reorganize_neuron_segments(neuron_segments_dict, date):
    """
    exchange the places of the neuron group key and stimulus type
    """
    reorganized_dict = {}

    for group_key, worm_segments in neuron_segments_dict.items():
        if group_key not in reorganized_dict:
            reorganized_dict[group_key] = {}

        # 遍历每个worm的segments
        for worm_key, segments in worm_segments.items():
            for segment in segments:
                stimulus_type = segment["stimulus_type"]

                # 确保stimulus_type在目标字典中存在
                if stimulus_type not in reorganized_dict[group_key]:
                    reorganized_dict[group_key][stimulus_type] = []

                # 将worm_key和date加入到每个segment中
                reorganized_dict[group_key][stimulus_type].append(
                    {
                        "worm_key": worm_key,  # 保存worm_key
                        "segment_index": segment["segment_index"],
                        "deltaFoverF_0": segment["deltaFoverF_0"],
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "z_scored": segment.get(
                            "z_scored", None
                        ),  # z-score标准化后的数据
                        "scaled_data": segment.get(
                            "scaled_data", None
                        ),  # max_abs_scale_neuron_segments_group_worm
                        "date": date,  # 添加日期字段
                    }
                )

    return reorganized_dict


def drop_wrong_trials(neuron_segments_dict_reorganized, neuron, stimulus, wrong_trials_list):
    """
    Drop trials specified in wrong_trials_list from neuron_segments_dict_reorganized for a specific neuron and stimulus.
    
    Parameters:
    - neuron_segments_dict_reorganized: dict
        The dictionary returned by reorganize_neuron_segments.
    - neuron: str
        The neuron group key (e.g., 'AWC_ON').
    - stimulus: str
        The stimulus type (e.g., 'benzaldehyde_high').
    - wrong_trials_list: list of str
        List of unique identifiers for trials to drop. 
        Format: "{worm_key}_{segment_index}_{date}"
    
    Returns:
    - neuron_segments_dict_reorganized: dict
        The dictionary with specified trials removed.
    """
    if neuron not in neuron_segments_dict_reorganized:
        print(f"Warning: Neuron group '{neuron}' not found.")
        return neuron_segments_dict_reorganized
        
    if stimulus not in neuron_segments_dict_reorganized[neuron]:
        print(f"Warning: Stimulus '{stimulus}' not found in neuron group '{neuron}'.")
        return neuron_segments_dict_reorganized

    wrong_trials_set = set(wrong_trials_list)
    
    # Filter the specific list of segments
    original_segments = neuron_segments_dict_reorganized[neuron][stimulus]
    cleaned_segments = []
    
    for segment in original_segments:
        # Construct unique ID
        trial_id = f"{segment['worm_key']}_{segment['segment_index']}_{segment['date']}"
        
        if trial_id not in wrong_trials_set:
            cleaned_segments.append(segment)
            
    neuron_segments_dict_reorganized[neuron][stimulus] = cleaned_segments
                
    return neuron_segments_dict_reorganized


# merge different date's neuron segments
def merge_multiple_dicts(*dicts):
    """
    merge multiple dictionaries into one
    """
    merged_dict = {}

    for d in dicts:
        for group_key, stimulus_data in d.items():
            if group_key not in merged_dict:
                merged_dict[group_key] = stimulus_data
            else:
                for stimulus_type, segments in stimulus_data.items():
                    if stimulus_type not in merged_dict[group_key]:
                        merged_dict[group_key][stimulus_type] = segments
                    else:
                        merged_dict[group_key][stimulus_type].extend(segments)

    return merged_dict


def process_neuron_segments(neuron_segments_dict, group_size=5, if_group=False):
    """
    z-score
    """
    # z-score标准化并记录每种stimulus_type的统计量
    neuron_segments_dict, stimulus_stats = per_worm_zscore(
        neuron_segments_dict, group_size, if_group
    )

    return neuron_segments_dict


def extract_and_normalize_worm_data(worm_data, date, group_size=5, if_group=False, vps_setting=1, boundary_method='preserve', **kwargs):
    """
    Process worm data to extract neuron segments and perform z-score normalization.
    Parameters:
    - worm_data: dict
        A dictionary where each key is a worm name and each value is a dictionary containing:
            - 'intensity': DataFrame of intensity values (rows correspond to neurons, columns to time points)
            - 'biological_ID': DataFrame with neuron IDs (single column, indexed by integer)
            - 'stimulus_intervals': list of tuples (start, end) indicating stimulus periods
            - 'buffer_intervals': list of tuples (not used in this function)
            - 'delta_F_over_F': DataFrame of delta_F_over_F values
    - group_size: int, a group contains different stimulus segments
    - boundary_method: str, method for boundary preservation in downsampling
    """
    # 1. Extract neuron groups and segments
    neuron_segments_dict, neuron_groups = extract_neuron_groups(worm_data, vps_setting, boundary_method, pre_segment=kwargs.get('pre_segment',5), post_segment=kwargs.get('post_segment',30))

    # 2. Process neuron segments
    neuron_segments_dict = process_neuron_segments(
        neuron_segments_dict, group_size, if_group
    )

    # 3. Reorganize neuron segments
    neuron_segments_dict_reorganized = reorganize_neuron_segments(
        neuron_segments_dict, date=date
    )

    return neuron_segments_dict, neuron_groups, neuron_segments_dict_reorganized

def transfer_dict2dataframe(neuron_segments_dict):
    """
    convert a nested dictionary into a pandas DataFrame for easier group and statistics calculate.
    """ 
    data_list = []
    for neuron_name, stimuli_data in neuron_segments_dict.items():
        for stimulus_type, segments in stimuli_data.items():
                for segment in segments:
                    delta_F_over_F0 = np.array(segment['deltaFoverF_0'])
                    time_points = len(delta_F_over_F0)

                    for t, dff in zip(range(time_points), delta_F_over_F0):
                        data_list.append({
                            'neuron': neuron_name,
                            'stimulus': stimulus_type,
                            'time_point': t,
                            'delta_F_over_F0': dff,
                            'worm_key': segment.get('worm_key', 'unknown'),
                            'segment_index': segment.get('segment_index', 'unknown'),
                            'date': segment.get('date', 'unknown')
                        })

    df = pd.DataFrame(data_list)
    return df

def load_and_process_worm_data(
    h5_file_path,
    channel_info_path,
    ID_info_path,
    date,
    stimulus_lists=None,
    sorting_config=None,
    exclude_key=None,
    vps_setting=1,
    baseline_pre=6,
    baseline_post=1,
    background_noise=102,
    group_size=5,
    if_group=False,
    boundary_method='preserve',
    **kwargs
):
    """
    Load and process worm data with enhanced boundary-preserving downsampling.
    
    Parameters:
    - h5_file_path: str, path to HDF5 data file
    - channel_info_path: str, path to channel information file
    - ID_info_path: str, path to ID information file
    - date: str, experiment date
    - stimulus_lists: optional, predefined stimulus lists
    - sorting_config: optional, configuration for sorting
    - exclude_key: optional, keys to exclude from processing
    - vps_setting: int, frames per second setting for downsampling
    - baseline_pre: int, baseline period before stimulus
    - baseline_post: int, baseline period after stimulus
    - background_noise: float, background noise level
    - group_size: int, size of stimulus groups for z-score normalization
    - if_group: bool, whether to perform group-wise normalization
    - boundary_method: str, method for boundary preservation during downsampling
        - 'preserve': preserve exact boundary points
        - 'weighted': weighted averaging with boundary emphasis
        - 'adaptive': adaptive method based on signal characteristics
        - 'original': standard averaging method
    
    Returns:
    - return_dict: dict containing processed data including segments with preserved boundaries
    """
    # get stimulus info and ID info
    if stimulus_lists is None:
        experiment_df, stimulus_lists = get_stimulus_info(
            channel_info_path, if_generate_stimulus_lists=True
        )
    else:
        experiment_df = extract_intervals_from_excel(channel_info_path)

    ID_info = load_worm_ID(ID_info_path)
    # load intensity data
    worm_data = load_worm_data(h5_file_path=h5_file_path,
                               experiment_info=experiment_df,
                               worm_id=ID_info,
                               stimulus_lists=stimulus_lists,
                               sorting_config=sorting_config,
                               exclude_key=exclude_key,
                               vps_setting=vps_setting,
                               baseline_pre=baseline_pre,
                               baseline_post=baseline_post,
                               background_noise=background_noise)

    # process and segment
    neuron_segments_dict, neuron_groups, neuron_segments_dict_reorganized = extract_and_normalize_worm_data(worm_data=worm_data,
                                                                                                            date=date,
                                                                                                            group_size=group_size,
                                                                                                            if_group=if_group,
                                                                                                            vps_setting=vps_setting,
                                                                                                            boundary_method=boundary_method,
                                                                                                            **kwargs
                                                                                                            )

    neuron_segments_dict_corrected = apply_baseline_correction(neuron_segments_dict_reorganized, correction_window=kwargs.get('correction_window', 5))

    neuron_segments_df = transfer_dict2dataframe(neuron_segments_dict_corrected)
    # return a dict
    return_dict = {
        "experiment_df": experiment_df,
        "stimulus_lists": stimulus_lists,
        "ID_info": ID_info,
        "worm_data": worm_data,
        "neuron_segments_dict": neuron_segments_dict,
        "neuron_segments_dict_reorganized": neuron_segments_dict_reorganized,
        "neuron_segments_dict_corrected": neuron_segments_dict_corrected,
        "neuron_groups": neuron_groups,
        "neuron_segments_df": neuron_segments_df
    }
    return return_dict

if __name__ == "__main__":
    result_dict_20251111 = load_and_process_worm_data(
        h5_file_path=r"H:\Process_temporary\WJH\olfactory\infer_result\20251111\20251111.h5",
        channel_info_path=r"H:\Process_temporary\WJH\olfactory\labjack_result\20251111\output_volumes_merged.xlsx",
        ID_info_path=r"H:\Process_temporary\WJH\olfactory\ID\result\20251111\20251111.xlsx",
        date="20251111",
        vps_setting=1,
        exclude_key=None)