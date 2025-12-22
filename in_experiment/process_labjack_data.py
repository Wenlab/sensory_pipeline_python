import sys
import os
import re
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from channel_info_get.extract_channel_info import ExtractChannelInfo
from channel_info_get.stimulus_config_builder import generate_config_from_channel_meanings
from in_experiment.transfer_tiff2npy import _group_folders_by_worm, _get_camera_subfolder
from utils.read_vols_using_dask import extract_tiff_max_number

def _get_experiment_tiff_num(ex_folder, **kwargs):
    """
    Get the maximum tiff number for each worm in the experiment folder
    Args:
        ex_folder: experiment folder path
        camera_type: 'green' or 'red'
    Returns:
        tiff_num_dict: dict of worm_name to list of max tiff numbers from each ex folder
    """
    worm_groups = _group_folders_by_worm(Path(ex_folder))
    for worm_name, folders in worm_groups.items():
        print(f"  - {worm_name}: {len(folders['ex'])} ex folders")
    
    camera_type = kwargs.get('camera_type', 'green')
    camera_subfolder = _get_camera_subfolder(camera_type)
    tiff_num_dict = {}
    for worm_name, folders in worm_groups.items():
        tiff_nums = []
        for worm_folder in folders['ex']:
            tiff_folder = worm_folder / camera_subfolder
            tiff_num = extract_tiff_max_number(tiff_folder)
            tiff_nums.append(tiff_num)
        tiff_num_dict[worm_name] = tiff_nums
    return tiff_num_dict

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


def process_labjack_data(root_folder, output_folder, ex_folder=None, config_file = None, FileMode='multiple', mode='merge'):
    """
    Process labjack data for multiple trial folders
    Args:
        root_folder: root folder containing trial folders
        output_folder: folder to save processed data
        ex_folder: experiment folder containing image data (optional, for correcting end frames)
        config_file: configuration file path
        FileMode: 'multiple' or 'single'
        mode: 'merge' or 'split' for single worm different trials
    """
    if FileMode == 'multiple':
        trial_folders = _get_trial_folders4multiple(root_folder)
    else:
        trial_folders = [root_folder]
    
    tiff_num_dict = None
    if ex_folder:
        print(f"Extracting tiff numbers from {ex_folder}...")
        tiff_num_dict = _get_experiment_tiff_num(ex_folder)
    extractor = ExtractChannelInfo(
        input_folder_list=trial_folders,
        output_folder=output_folder,
        config_file=config_file,
        FileMode=FileMode,
        tiff_num_dict=tiff_num_dict
    )
    extractor.extract_channel_info()

    if mode == 'merge':
        # Merge sheets with same prefix
        states_excel_path = os.path.join(output_folder, "output_states.xlsx")
        volumes_excel_path = os.path.join(output_folder, "output_volumes.xlsx")
        
        merged_volumes_path = os.path.join(output_folder, "output_volumes_merged.xlsx")
        
        if os.path.exists(volumes_excel_path):
            _merge_sheets_by_prefix(volumes_excel_path, merged_volumes_path)
        
        print(f"Merge completed. Output files: {merged_volumes_path}")
        


if __name__ == "__main__":
    channel_meanings = {
        "1": "Control1",
        "2": "Control2",
        "3": "Buffer",
        "00": "Diacetyl E6 s",
        "01": "NaCl 100mM s",
        "02": "Benzaldehyde E6 s",
        "15": "2-Nonanone E6 s",
        "14": "Isoamylalcohol E6 s",
        "13": "1-Octanol E6 s",
        "03": "EGCG 1uM t",
        "12": "EGCG 10uM t",
        "04": "L-Theanine 10uM t",
        "11": "L-Theanine 1uM t",
        "05": "Ethanol E3 s",
        "10": "Ethanol E2 s",
        "06": "Methanol E3 s",
        "09": "Methanol E2 s",
        "07": "DMSO E3 s",
        "08": "DMSO E2 s",
    }

    generate_config_from_channel_meanings(
        channel_meanings=channel_meanings,
        odor_json_path=r"H:\Process_temporary\WJH\sensory_pipeline_python\data\config\stimulus.json",
        color_scheme_path=r"H:\Process_temporary\WJH\sensory_pipeline_python\data\config\stimulus_color_scheme.json",
        output_directory=r"H:\Process_temporary\WJH\olfactory\labjack_result\20251205",
        bit_mode="16-bit",
        slice_number=20,
        state_length=8
    )

    root_folder = r'\\192.168.1.192\Odor\Jinghao-Wang\20251205_tea_vehicle_test\labjack'
    process_labjack_data(root_folder, ex_folder=r"\\192.168.1.192\Odor\Jinghao-Wang\20251205_tea_vehicle_test",output_folder=r'H:\Process_temporary\WJH\olfactory\labjack_result\20251205\fix', config_file= r"H:\Process_temporary\WJH\olfactory\labjack_result\20251205\config.json" ,FileMode='multiple', mode='merge')