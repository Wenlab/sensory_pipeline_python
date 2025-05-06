if __name__ == '__main__':
    folder_path = r"G:\LAB\DATA\result\20250421_EGCG"
#%%
import sys
import os
import h5py
import pandas as pd
import numpy as np
if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_load.get_stimulus_info import extract_intervals_from_excel, extract_intervals
from data_load.df_sort import combine_alternately, split_alternately, sort_by_intervals, sort_intervals_combined
from data_load.curve_fit import calculate_delta_F_over_F0

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

def process_worm_data(
    key, f, experiment_df,
    worm, stimulus_lists, 
    stimulus_sort=None, buffer_sort=None, 
    need_sorting=False
):
    """处理单个worm的数据并存储到worm_data字典"""
    # 公共处理部分：读取ID、拟合F0等
    ID = worm[key]['biological'].to_numpy()
    
    modified_ID = [
        id_value if isinstance(id_value, str) and id_value.startswith('A') else f'{index}'
        for index, id_value in enumerate(ID)
    ]
    modified_ID_df = pd.DataFrame(modified_ID)
    
    stimulus_intervals, buffer_intervals = extract_intervals(experiment_df, key)
    intensity = pd.DataFrame(f[key]['intensity'][:])
    # print(f"Processing {key}...")
    # print(f"intensity_dtype: {intensity.dtypes}")
    # fit curve
    delta_F_over_F0, fitted_F0, quality_info = calculate_delta_F_over_F0(
        intensity, stimulus_intervals)
    # print(f"delta_F_over_F0_dtype: {delta_F_over_F0.dtypes}")

    stimulus_list = stimulus_lists.get(key, [])

    # 需要排序的特殊处理
    if need_sorting:
        combined_intervals = sort_intervals_combined(
            stimulus_intervals, stimulus_sort, 
            buffer_intervals, buffer_sort
        )
        
        # 对各个DataFrame进行排序
        delta_F_over_F_sorted = sort_by_intervals(
            delta_F_over_F0, stimulus_intervals, stimulus_sort, 
            buffer_intervals, buffer_sort
        )
        intensity_sorted = sort_by_intervals(
            intensity, stimulus_intervals, stimulus_sort, 
            buffer_intervals, buffer_sort
        )
        fitted_F0_sorted = sort_by_intervals(
            fitted_F0, stimulus_intervals, stimulus_sort, 
            buffer_intervals, buffer_sort
        )


        # 调整intervals
        interval_list = [end - start for start, end in combined_intervals]
        
        start = 0
        combined_intervals_recovered = []
        for index in interval_list:
            end = start + index
            combined_intervals_recovered.append((start, end))
            start = end
        
        buffer_intervals_re, stimulus_intervals_re = split_alternately(
            combined_intervals_recovered
        )

        stimulus_list = [stimulus_list[i] for i in stimulus_sort]

        # 存储排序后的数据
        return {
            'intensity': intensity_sorted,
            'biological_ID': modified_ID_df,
            'fitted_F0': fitted_F0_sorted,
            'quality_info': quality_info,
            'delta_F_over_F': delta_F_over_F_sorted,
            'stimulus_intervals': stimulus_intervals_re,
            'buffer_intervals': buffer_intervals_re,
            'stimulus_list': stimulus_list
        }
    else:
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

#%%
if __name__ == '__main__':
    info_excel = os.path.join(folder_path, 'output_volumes.xlsx')
    experiment_info = extract_intervals_from_excel(info_excel)

    stimulus_lists_0421_EGCG = {
    'w1': ['c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5'] * 3,
    'w2': ['c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5'] * 3,
    'w3': ['c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5'] * 3,
    'w4': ['c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5'] * 3,
    'w5': ['c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5'] * 3,
    'w6': ['c1_5', 'c1_4', 'c1_3', 'c1_2', 'c1_1'] * 3,  # Reversed and reorder the delta_F_over_F0
    }

    ID_0421_EGCG = load_worm_ID(folder_path + r'\ID0421EGCG.xlsx')
    worm_data_0421_EGCG = {}
    with h5py.File(folder_path + r'\20250421_EGCG.h5', 'r') as f:
        for key in f.keys():
            if key == 'w2' or key == 'w3':
                worm_data_0421_EGCG[key] = process_worm_data(
                    key, f, experiment_info, ID_0421_EGCG, 
                    stimulus_lists=stimulus_lists_0421_EGCG
                )

    print('done')