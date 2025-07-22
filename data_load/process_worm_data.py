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


# %%
def extract_neuron_groups(worm_data, vps_setting=1):
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
            group_key = full_id[:4]
            if group_key.startswith("A"):
                if group_key not in neuron_groups:
                    neuron_groups[group_key] = set()
                neuron_groups[group_key].add(full_id)

    # 2. Prepare data structures
    worm_keys = sorted(worm_data.keys())
    num_worms = len(worm_keys)

    neuron_segments_dict = {}
    max_num_segments = 0
    neuron_group_keys = sorted(neuron_groups.keys())
    # sorted_worm_list = sorted(worm_list, key=lambda x: int(x[1:]))

    # 3. Process each neuron group
    for group_key in neuron_group_keys:
        worm_segments = {}
        min_num_segments = None
        group_neurons = neuron_groups[group_key]

        for idx, worm_key in enumerate(worm_keys):
            data = worm_data[worm_key]
            biological_ID = data["biological_ID"][0].tolist()
            delta_F_over_F = data["delta_F_over_F"]
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
                start_idx = max(0, start_time - 5*vps_setting)# included
                end_idx = min(len(delta_F_trace), end_time + 25*vps_setting)# not included

                # Define fixed stimulus duration
                fixed_stimulus_duration = 10 * vps_setting

                # Split the segment into three parts
                pre_stimulus = delta_F_trace[start_idx:start_time]
                stimulus = delta_F_trace[
                    start_time : start_time + fixed_stimulus_duration
                ]
                post_stimulus = delta_F_trace[end_time:end_idx]

                # Concatenate the parts
                seg_data = np.concatenate((pre_stimulus, stimulus, post_stimulus))

                # downsample the segment
                if vps_setting > 1:
                    seg_data = downsampling(seg_data, vps_setting)
                # Smooth the concatenated segment
                seg_data_smooth = gaussian_filter1d(seg_data, sigma=1)

                stimulus_type = worm_stimuli[idx]

                segments_data.append(
                    {
                        "stimulus_type": stimulus_type,
                        "deltaFoverF_0": seg_data_smooth,
                        "start_time": (start_time-start_idx)//vps_setting,# relative start time
                        "end_time": (start_time+fixed_stimulus_duration-start_idx-1)//vps_setting,# relative end time
                    }
                )
            if not segments_data:
                continue

            seg_count = len(segments_data)
            if min_num_segments is None:
                min_num_segments = seg_count
            else:
                min_num_segments = min(min_num_segments, seg_count)

            worm_segments[worm_key] = segments_data

        if not worm_segments:
            continue

        # Ensure all worms have the same length of segments
        for worm_key in worm_segments:
            worm_segments[worm_key] = worm_segments[worm_key][:min_num_segments]

        if min_num_segments > max_num_segments:
            max_num_segments = min_num_segments

        # 构建每个段的信息
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

def downsampling(segment_data, vps_setting=5):
    # Convert to numpy array if it's not already
    if not isinstance(segment_data, np.ndarray):
        segment_data = np.array(segment_data)
    
    # Calculate the length of downsampled data
    original_length = len(segment_data)
    downsampled_length = original_length // vps_setting
    
    # Trim the data to make it divisible by vps_setting
    trimmed_length = downsampled_length * vps_setting
    trimmed_data = segment_data[:trimmed_length]
    
    # Reshape and calculate mean for each bin
    reshaped_data = trimmed_data.reshape(downsampled_length, vps_setting)
    downsampled_data = np.mean(reshaped_data, axis=1)
    
    return downsampled_data

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


def extract_and_normalize_worm_data(worm_data, date, group_size=5, if_group=False, vps_setting=1):
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
    """
    # 1. Extract neuron groups and segments
    neuron_segments_dict, neuron_groups = extract_neuron_groups(worm_data, vps_setting)

    # 2. Process neuron segments
    neuron_segments_dict = process_neuron_segments(
        neuron_segments_dict, group_size, if_group
    )

    # 3. Reorganize neuron segments
    neuron_segments_dict_reorganized = reorganize_neuron_segments(
        neuron_segments_dict, date=date
    )

    return neuron_segments_dict, neuron_groups, neuron_segments_dict_reorganized


def load_and_process_worm_data(
    h5_file_path,
    channel_info_path,
    ID_info_path,
    date,
    stimulus_lists=None,
    sorting_config=None,
    exclude_key=None,
    vps_setting=1,
    group_size=5,
    if_group=False,
):
    """
    h5_file_path
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
                               exclude_key=exclude_key)
    
    # process and segment
    neuron_segments_dict, neuron_groups, neuron_segments_dict_reorganized = extract_and_normalize_worm_data(worm_data=worm_data,
                                                                                                            date=date,
                                                                                                            group_size=group_size,
                                                                                                            if_group=if_group,
                                                                                                            vps_setting=vps_setting
                                                                                                            )

    # return a dict
    return experiment_df, stimulus_lists, ID_info, worm_data, neuron_segments_dict, neuron_segments_dict_reorganized, neuron_groups

if __name__ == "__main__":
    # sorting_config_0604 = {
    #     'w3':{
    #             'stimulus_sort': [6,7,4,5,2,3,0,1],
    #             'buffer_sort': [0,7,8,5,6,3,4,1,2]
    #     },
    #     'w4':{
    #             'stimulus_sort': [6,7,4,5,2,3,0,1],
    #             'buffer_sort': [0,7,8,5,6,3,4,1,2]
    #     },
    #     'w6':{
    #             'stimulus_sort': [6,7,4,5,2,3,0,1],
    #             'buffer_sort': [0,7,8,5,6,3,4,1,2]
    #     },
    #     'w7':{
    #             'stimulus_sort': [6,7,4,5,2,3,0,1],
    #             'buffer_sort': [0,7,8,5,6,3,4,1,2]
    #     },
    #     'w10':{
    #             'stimulus_sort': [6,7,4,5,2,3,0,1],
    #             'buffer_sort': [0,7,8,5,6,3,4,1,2]
    #     }
    # }

    experiment_df_0604_odor, stimulus_lists_0604_odor, ID_info_0604_odor, worm_data_0604_odor, neuron_segments_dict_0604_odor, neuron_segments_dict_reorganized_0604_odor, neuron_groups_0604_odor = load_and_process_worm_data(
        h5_file_path=  r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\20250604.h5",
        channel_info_path= r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\output_volumes.xlsx",
        ID_info_path= r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\ID0604_odor.xlsx",
        date='20250604',
        vps_setting=5
    )
    from utils.HDF5Toolkit import save_h5file
    save_h5file(
        r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\neuron_segments_dict_0604_odor.h5",
        root_name='neuron_segments_dict',
        mode='w',
        **neuron_segments_dict_reorganized_0604_odor
    )