import sys
import os
import re
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glob import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from channel_info_get.extract_channel_info import ExtractChannelInfo
from channel_info_get.stimulus_config_builder import generate_config_from_channel_meanings

def _get_trial_folders4multiple(root_folder):
    """
    Get trial folders for multiple experiments under root_folder
    Args:
        root_folder: root folder containing multiple experiment folders
    Returns:
        trial_folders: list of trial folder paths
    """
    experiment_folders = glob(os.path.join(root_folder, 'w*'))
    return experiment_folders

def _extract_prefix_and_suffix(sheet_name):
    """
    Extract prefix and suffix from sheet name
    Args:
        sheet_name: sheet name like 'w1', 'w1_1', 'w1_2'
    Returns:
        prefix: 'w1'
        suffix: 0 for 'w1', 1 for 'w1_1', 2 for 'w1_2'
    """
    match = re.match(r'^(w\d+)(?:_(\d+))?$', sheet_name)
    if match:
        prefix = match.group(1)
        suffix = int(match.group(2)) if match.group(2) else 0
        return prefix, suffix
    return None, None

def _merge_sheets_by_prefix(excel_path, output_path):
    """
    Merge sheets with same prefix, adjusting start/end counters
    Args:
        excel_path: path to input Excel file
        output_path: path to output merged Excel file
    """
    # Read all sheets
    all_sheets = pd.read_excel(excel_path, sheet_name=None)
    
    # Group sheets by prefix
    sheet_groups = defaultdict(list)
    for sheet_name in all_sheets.keys():
        prefix, suffix = _extract_prefix_and_suffix(sheet_name)
        if prefix:
            sheet_groups[prefix].append((suffix, sheet_name))
    
    # Sort each group by suffix
    for prefix in sheet_groups:
        sheet_groups[prefix].sort(key=lambda x: x[0])
    
    # Merge sheets
    merged_sheets = {}
    for prefix, sheets in sheet_groups.items():
        merged_df = pd.DataFrame()
        cumulative_offset = 0
        
        for idx, (suffix, sheet_name) in enumerate(sheets):
            df = all_sheets[sheet_name].copy()
            
            if idx > 0:
                # Adjust start and end by adding cumulative offset
                if 'start' in df.columns:
                    df['start'] = df['start'] + cumulative_offset
                if 'end' in df.columns:
                    df['end'] = df['end'] + cumulative_offset
            
            # Update cumulative offset with the last 'end' value of current sheet
            if 'end' in df.columns and len(df) > 0:
                cumulative_offset = df['end'].iloc[-1]
            
            # Append to merged dataframe
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        
        merged_sheets[prefix] = merged_df
    
    # Write merged sheets to new Excel file
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in merged_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Merged sheets saved to {output_path}")
    return merged_sheets


def process_labjack_data(root_folder, output_folder, config_file = None, FileMode='multiple', mode='merge'):
    """
    Process labjack data for multiple trial folders
    Args:
        root_folder: root folder containing trial folders
        output_folder: folder to save processed data
        config_file: configuration file path
        FileMode: 'multiple' or 'single'
        mode: 'merge' or 'split' for single worm different trials
    """
    if FileMode == 'multiple':
        trial_folders = _get_trial_folders4multiple(root_folder)
    else:
        trial_folders = [root_folder]

    extractor = ExtractChannelInfo(
        input_folder_list=trial_folders,
        output_folder=output_folder,
        config_file=config_file,
        FileMode=FileMode,
    )
    extractor.extract_channel_info()

    if mode == 'merge':
        # Merge sheets with same prefix
        states_excel_path = os.path.join(output_folder, "output_states.xlsx")
        volumes_excel_path = os.path.join(output_folder, "output_volumes.xlsx")
        
        merged_states_path = os.path.join(output_folder, "output_states_merged.xlsx")
        merged_volumes_path = os.path.join(output_folder, "output_volumes_merged.xlsx")
        
        if os.path.exists(states_excel_path):
            _merge_sheets_by_prefix(states_excel_path, merged_states_path)
        
        if os.path.exists(volumes_excel_path):
            _merge_sheets_by_prefix(volumes_excel_path, merged_volumes_path)
        
        print(f"Merge completed. Output files: {merged_states_path}, {merged_volumes_path}")
        


if __name__ == "__main__":
    channel_meanings = {
        "1": "Control1",
        "2": "Control2",
        "3": "Buffer",
        "00": "60 supernatant",
        "01": "61 supernatant",
        "02": "62 supernatant",
        "03": "68 supernatant",
        "04": "78 supernatant",
        "05": "c12 supernatant",
        "06": "swb supernatant",
        "09": "swb 1night",
        "10": "c12 1night",
        "11": "78 1night",
        "12": "68 1night",
        "13": "62 1night",
        "14": "61 1night",
        "15": "60 1night"
    }
    generate_config_from_channel_meanings(
        channel_meanings=channel_meanings,
        odor_json_path="data_load/config/bacteria.json",
        color_scheme_path="data_load/config/bacteria_color_scheme.json",
        output_directory=r"H:\Process_temporary\WJH\olfactory\labjack_result\20251105_bac",
        bit_mode="16-bit",
        slice_number=20,
        state_length=8
    )

    root_folder = r'\\192.168.1.192\Odor\Jinghao-Wang\20251105_bac\labjack'
    process_labjack_data(root_folder, output_folder=r'H:\Process_temporary\WJH\olfactory\labjack_result\20251105_bac', config_file= r"H:\Process_temporary\WJH\olfactory\labjack_result\20251105_bac\config.json" ,FileMode='multiple', mode='merge')