import pandas as pd
import numpy as np


def combine_alternately(buffer_list, stimulus_list):
    '''
    Combines two lists alternately.
    '''
    combined_list = []
    # 按照索引依次取出元素
    min_length = min(len(buffer_list), len(stimulus_list))
    for i in range(min_length):
        combined_list.append(buffer_list[i])
        combined_list.append(stimulus_list[i])
    # 如果一个列表剩余未处理的部分，直接追加
    combined_list.extend(buffer_list[min_length:])
    combined_list.extend(stimulus_list[min_length:])
    return combined_list

def split_alternately(combined_list):
    '''
    Splits a combined list into two lists alternately.
    '''
    # 计算 buffer_list 和 stimulus_list 的长度
    # 因为两个列表交替，所以 combined_list 的长度应该是 buffer_list 和 stimulus_list 长度的两倍（或接近）
    buffer_list = combined_list[::2]  # 取所有偶数索引的元素，作为 buffer_list
    stimulus_list = combined_list[1::2]  # 取所有奇数索引的元素，作为 stimulus_list
    
    return buffer_list, stimulus_list

def sort_by_intervals(input_df: pd.DataFrame, stimulus_intervals: list,stimulus_sort: list[int], buffer_intervals, buffer_sort: list[int]):
    '''
    Sorts the input DataFrame by the stimulus and buffer
    '''
    values = input_df.values
    stimulus_intervals_re = np.take(np.array(stimulus_intervals), stimulus_sort, axis=0)
    # stimulus_intervals_re_list = stimulus_intervals_re.tolist()
    # stimulus_intervals_re_list_with_tuples = [tuple(item) for item in stimulus_intervals_re_list]
    buffer_intervals_re = np.take(np.array(buffer_intervals), buffer_sort, axis=0)
    # buffer_intervals_re_list = buffer_intervals_re.tolist()
    # buffer_intervals_re_list_with_tuples = [tuple(item) for item in buffer_intervals_re_list]
    # combined_list = combine_alternately(buffer_intervals_re_list_with_tuples, stimulus_intervals_re_list_with_tuples)
    index = [i for i in range(buffer_intervals_re[0][0], buffer_intervals_re[0][1])]
    for i in range(len(stimulus_intervals)):
        index += [j for j in range(stimulus_intervals_re[i][0], stimulus_intervals_re[i][1])]
        index += [j for j in range(buffer_intervals_re[i+1][0], buffer_intervals_re[i+1][1])]
    return pd.DataFrame(np.take(values, index, axis=1))

def sort_intervals_combined(stimulus_intervals: list,stimulus_sort: list[int], buffer_intervals, buffer_sort: list[int]):
    '''
    Sorts the combined list of stimulus and buffer intervals.
    '''
    stimulus_intervals_re = np.take(np.array(stimulus_intervals), stimulus_sort, axis=0)
    stimulus_intervals_re_list = stimulus_intervals_re.tolist()
    stimulus_intervals_re_list_with_tuples = [tuple(item) for item in stimulus_intervals_re_list]
    buffer_intervals_re = np.take(np.array(buffer_intervals), buffer_sort, axis=0)
    buffer_intervals_re_list = buffer_intervals_re.tolist()
    buffer_intervals_re_list_with_tuples = [tuple(item) for item in buffer_intervals_re_list]
    combined_list = combine_alternately(buffer_intervals_re_list_with_tuples, stimulus_intervals_re_list_with_tuples)
    return combined_list