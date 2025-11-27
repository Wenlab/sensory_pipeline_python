if __name__ == "__main__":
    folder_path = r'H:\Process_temporary\WJH\olfactory\analysis\olfactory\20240619_wen0065\neuronal_activity'

#%%
import os
import re
import numpy as np
import pandas as pd
import h5py 

#%%
def extract_intervals_from_excel(excel_path):
    '''
    Extracts the stimulus and buffer intervals from the excel file.
    '''
    # read the excel file
    experiment_excel = pd.ExcelFile(excel_path)
    ex_sheet_names = experiment_excel.sheet_names
    # store the experiment information in a dictionary
    experiment_df = {}
    for sheet_name in ex_sheet_names:
        df = experiment_excel.parse(sheet_name)
        df['start'] = df['start']
        df['end'] = df['end']
        experiment_df[sheet_name] = df
    
    return experiment_df

def stimulus_lists_generate(experiment_df):
    '''
    Generate stimulus lists for each worm.
    '''
    stimulus_lists = {}
    excluded_states = ["Buffer", "All Off", "Unknown"]
    for key in experiment_df.keys():
        stimulus_lists[key] = []
        for index, row in experiment_df[key].iterrows():
            if row['end'] - row['start'] != 0:
                if row['state'] not in excluded_states:
                    stimulus_lists[key].append(row['state'])
    return stimulus_lists

def get_stimulus_info(excel_path, if_generate_stimulus_lists=False):
    '''
    Main function to extract stimulus information from the excel file.
    '''
    # Load the excel file
    experiment_df = extract_intervals_from_excel(excel_path)

    if if_generate_stimulus_lists:
        # Generate stimulus lists for each worm
        stimulus_lists = stimulus_lists_generate(experiment_df)
        return experiment_df, stimulus_lists
    else:
        return experiment_df
    
def extract_intervals(experiment_df,key):
    '''
    Extracts the stimulus and buffer intervals for a given worm.
    '''
    stimulus_intervals = []
    buffer_intervals = []
    excluded_states = ["Buffer", "All Off", "Unknown"]
    for index, row in experiment_df[key].iterrows():
        start = row['start']
        end = row['end']
        if end - start != 0:
            if not any(state in row['state'] for state in excluded_states):
                stimulus_intervals.append((start, end))
            elif "Buffer" in row['state']:
                buffer_intervals.append((start, end))
    return stimulus_intervals, buffer_intervals

#%%
if __name__ == "__main__":
    # Load the excel file
    excel_path = os.path.join(folder_path, 'experiment.xlsx')
    experiment_df, stimulus_lists = get_stimulus_info(excel_path, if_generate_stimulus_lists=True)
    for key in experiment_df.keys():
        stimulus_intervals, buffer_intervals = extract_intervals(experiment_df, key)
        print(f"Worm: {key}")
        print(f"Stimulus Intervals: {stimulus_intervals}")
        print(f"Buffer Intervals: {buffer_intervals}")

