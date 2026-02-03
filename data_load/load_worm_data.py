#%%
import sys
import os
import h5py
import pandas as pd
import numpy as np
if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_load.get_stimulus_info import extract_intervals_from_excel, extract_intervals
from data_load.curve_fit import calculate_delta_F_over_F0
from data_load.preprocessing import detect_and_mask_step_drops

#%%
## load worm_ID
def load_worm_ID(file_path):
    '''
    Load the worm ID file.
    '''
    worm_ID_excel = file_path
    xls = pd.ExcelFile(worm_ID_excel)
    sheets = xls.sheet_names
    worm = {}
    for sheet in sheets:
        worm[sheet] = xls.parse(sheet)
    return worm

def load_worm_data_dict(
    key, f, experiment_df,
    worm, stimulus_lists, 
    vps_setting=1,
    baseline_pre=6, baseline_post=1,
    background_noise=102,
):
    """处理单个worm的数据并存储到worm_data字典"""
    # check if there's key in worm ID, if not use digital ID as biological ID
    if key not in worm:
        worm[key] = pd.DataFrame({
            'biological': np.arange(f[key]['intensity'].shape[1]),
            'digital': np.arange(f[key]['intensity'].shape[1])
        })

    biologicalID = worm[key]['biological'].to_numpy()
    digitalID = worm[key]['digital'].to_numpy()
    id_map = {}
    for d_id, b_id in zip(digitalID, biologicalID):
        if isinstance(b_id, str):
            id_map[d_id] = b_id
        else:
            id_map[d_id] = str(d_id)

    modified_ID = [
        id_map.get(id_value, str(id_value))
        for index, id_value in enumerate(range(max(digitalID) + 1))
    ]
    modified_ID_df = pd.DataFrame(modified_ID)
    
    stimulus_intervals, buffer_intervals = extract_intervals(experiment_df, key)
    intensity = pd.DataFrame(f[key]['intensity'][:])# this conversion is useless but Fit function is based on dataframe(which is our old data's form)
    
    # Detect and mask step artifacts
    intensity = detect_and_mask_step_drops(intensity)
    try:
        raw_n_seq = np.asarray(f[key]['n_seg'][:]).ravel()
        n_seq = []
        for value in raw_n_seq:
            try:
                length = int(value)
            except (TypeError, ValueError):
                continue
            if length > 0:
                n_seq.append(length)
        if not n_seq:
            n_seq = None
    except KeyError:
        n_seq = None
    # print(f"Processing {key}...")
    # print(f"intensity_dtype: {intensity.dtypes}")
    # fit curve
    delta_F_over_F0, fitted_F0, quality_info = calculate_delta_F_over_F0(
        intensity, stimulus_intervals, vps_setting=vps_setting,
        baseline_pre=baseline_pre, baseline_post=baseline_post, background_noise=background_noise,
        n_seq=n_seq,
    )
    # print(f"delta_F_over_F0_dtype: {delta_F_over_F0.dtypes}")

    stimulus_list = stimulus_lists.get(key, [])

    # 存储原始数据
    return {
        'intensity': intensity,
        'biological_ID': modified_ID_df,
        'fitted_F0': fitted_F0,
        'quality_info': quality_info,
        'delta_F_over_F': delta_F_over_F0,
        'stimulus_intervals': stimulus_intervals,
        'buffer_intervals': buffer_intervals,
        'stimulus_list': stimulus_list
    }

def load_worm_data(
    h5_file_path, experiment_info, worm_id, 
    stimulus_lists, exclude_key=None, vps_setting=1,
    baseline_pre=6, baseline_post=1,
    background_noise=102,
):
    """
    Load worm data from HDF5 file and process it.
    
    Parameters:
        folder_path (str): Path to the folder containing the HDF5 file.
        experiment_info (pd.DataFrame): DataFrame containing experiment information.
        worm_id (dict): Dictionary containing worm IDs (dataframe).
        stimulus_lists: a dictionary containing each worm's stimulus list.
        sorting_config: dict, optional
            {
                'worm_key': {
                    'stimulus_sort': [0, 1, 2, ...],
                    'buffer_sort': [0, 1, 2, ...]
                }
            }
        exclude_key (list, optional): List of keys to exclude from processing.
    Returns:
        dict: Processed worm data dictionary.
    """
    worm_data = {}
    h5_file_path = os.path.join(h5_file_path)
    with h5py.File(h5_file_path, 'r') as f:
        for key in f.keys():
            if exclude_key and key in exclude_key:
                continue
            worm_data[key] = load_worm_data_dict(
                key, f, experiment_info, worm_id, stimulus_lists,
                vps_setting=vps_setting,
                baseline_pre=baseline_pre,
                baseline_post=baseline_post,
                background_noise=background_noise
            )
    return worm_data

#%%
if __name__ == '__main__':
    folder_path = r"H:\Process_temporary\WJH\olfactory\ID\result\20250421_EGCG"
    info_excel = os.path.join(folder_path, 'output_volumes.xlsx')
    experiment_info = extract_intervals_from_excel(info_excel)
    ID_0421_EGCG = load_worm_ID(folder_path + r'\ID0421EGCG.xlsx')
    worm_data_0421_EGCG = {}
    with h5py.File(folder_path + r'\20250421_EGCG.h5', 'r') as f:
        for key in f.keys():
            worm_data_0421_EGCG[key] = load_worm_data_dict(
                key, f, experiment_info, ID_0421_EGCG, 
            )

    print('done')