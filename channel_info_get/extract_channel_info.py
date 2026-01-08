#%%
import os
import h5py
import pandas as pd
import math
import json
import re
from tqdm import tqdm
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#%%
class ExtractChannelInfo:
    def __init__(self, input_folder_list, output_folder, metadata_path=None, config_file=None, FileMode='multiple', tiff_num_dict=None):
        """
        FileMode: 'single' or 'multiple'
        """
        self.input_folder_list = input_folder_list
        self.output_folder = output_folder
        self.metadata_path = metadata_path
        self.config_file = config_file
        self.file_mode = FileMode
        self.tiff_num_dict = tiff_num_dict
        self.worm_trial_counters = defaultdict(int)
        self.channel_meanings, self.state_mappings, self.slice_number = self.load_config()
        self.shift_frame = self.read_shift_frames_from_metadata() if metadata_path else 0
        
    def extract_worm_id(self, string):
        """
        extract worm id from string
        """
        match = re.search(r'w(\d+)', string, re.IGNORECASE)
        if match:
            return f"{match.group()}"
        else:
            return "NoWORM"

    def load_config(self):
        default_channel_meanings = {
            "1": "Control1",
            "2": "Control2",
            "3": "Buffer",
            "4": "Odor1",
            "5": "Odor2",
            "6": "Odor3",
            "7": "Odor4",
            "8": "Odor5"
        }   

        default_state_mappings = {
            (0,0,0,0,0,0,0,0): "All Off",  # 直接使用元组而不是字符串
            (1,1,1,0,0,0,0,0): "Buffer",
            (1,0,1,1,0,0,0,0): "Buffer",
            (1,0,1,0,1,0,0,0): "Buffer",
            (1,0,1,0,0,1,0,0): "Buffer",
            (1,0,1,0,0,0,1,0): "Buffer",
            (1,0,1,0,0,0,0,1): "Buffer",
            (0,1,1,1,0,0,0,0): "Odor1",
            (0,1,1,0,1,0,0,0): "Odor2",
            (0,1,1,0,0,1,0,0): "Odor3",
            (0,1,1,0,0,0,1,0): "Odor4",
            (0,1,1,0,0,0,0,1): "Odor5"
        }

        default_slice_number = 20

        if not self.config_file:
            return default_channel_meanings, default_state_mappings, default_slice_number
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            channel_meanings = config.get("channel_meanings", default_channel_meanings)
            state_mappings = config.get("state_mappings", default_state_mappings)
            slice_number = config.get("slice_number", default_slice_number)
            
            # Convert state_mappings keys from string lists to tuples（json doesn't support using tuple as key）
            converted_state_mappings = {}
            for key, value in state_mappings.items():
                key_tuple = tuple(int(x) for x in key.strip("[]").split(","))
                converted_state_mappings[key_tuple] = value
            
            # Convert channel_meanings keys from strings to integers
            converted_channel_meanings = {int(k): v for k, v in channel_meanings.items()}
            
            return converted_channel_meanings, converted_state_mappings, slice_number
        except Exception as e:
            print(f"Error loading config file: {e}")
            return default_channel_meanings, default_state_mappings, default_slice_number

    def read_shift_frames_from_metadata(self):
        metadata_df = pd.read_csv(self.metadata_path)
        shift_frame = metadata_df['frame_number'].values[0]
        return shift_frame

    def map_state(self, state):
        return self.state_mappings.get(state, "Unknown")

    def correct_counter(self, master, slave):
        corrected = np.array(master, copy=True)
        n = len(master)
        diff_indices = np.where(master != slave)[0]
        
        for i in diff_indices:
            prev_val = corrected[i-1] if i > 0 else None
            next_val = master[i+1] if i < n-1 else None
            
            master_ok = True
            if prev_val is not None and abs(master[i] - prev_val) > 1:
                master_ok = False
            if next_val is not None and abs(next_val - master[i]) > 1:
                master_ok = False
            
            slave_val = slave[i]
            slave_next = slave[i+1] if i < n-1 else None

            slave_ok = True
            if prev_val is not None and abs(slave_val - prev_val) > 1:
                slave_ok = False
            if slave_next is not None and abs(slave_next - slave_val) > 1:
                slave_ok = False
            
            if not master_ok and slave_ok:
                corrected[i] = slave[i]
            
            if not master_ok and not slave_ok:
                if prev_val >= 0:
                    if abs(next_val - prev_val) <= 1:
                        corrected[i] = max(prev_val, next_val)
        return corrected

    def _read_data_from_single(self, h5_filename):
        with h5py.File(h5_filename, 'r') as f:
            counter_data = f['counter'][:][:, 0] # counter for master camera
            counter_data_slave = f['counter'][:][:, 1] # counter for slave camera
            dout_states_data = f['dout_states'][:]

        counter_data = self.correct_counter(counter_data, counter_data_slave)

        df = pd.DataFrame({
            'counter': counter_data,
            'state': [tuple(row) for row in dout_states_data]
        })
        df_unique = df.drop_duplicates(subset='counter', keep='last')
        df_sorted = df_unique.sort_values(by='counter').reset_index(drop=True)
        df_sorted['state_name'] = df_sorted['state'].apply(lambda x: self.map_state(x))

        return df_sorted
    
    def _get_start_end_counters_single(self, df):
        """
        Extracts the state changes from a single HDF5 file.
        """
        states = []
        start_counters = []
        end_counters = []
        current_state = None
        current_start = None

        # Iterate through the DataFrame to extract state changes
        for _, row in df.iterrows():
            state = row['state_name']
            counter = row['counter']

            if current_start is None:
                current_state = state
                current_start = counter
            # add counter when state changes
            if state != current_state:
                states.append(current_state)
                start_counters.append(current_start)
                end_counters.append(prev_counter)
                current_state = state
                current_start = counter
            prev_counter = counter
        # add last state
        if current_state is not None:
            states.append(current_state)
            start_counters.append(current_start)
            end_counters.append(counter)
        return states, start_counters, end_counters

    def generate_volume_df(self, states, start_counters, end_counters, max_frame_number=None):
        if max_frame_number is not None and len(end_counters) > 0:
            end_counters[-1] = max_frame_number + start_counters[0]

        if states[0] == 'All Off':
            adjust_frame = end_counters[0]
            start_idx = 1
        else:
            adjust_frame = start_counters[0]
            start_idx = 0
        
        start_volumes = [math.floor((x - adjust_frame + 1) / self.slice_number) for x in start_counters]
        end_volumes = [math.floor((x - adjust_frame + 1) / self.slice_number) for x in end_counters]
        end_idx = -1 if states[-1] == 'All Off' else None

        states = states[start_idx:end_idx]
        start_counters = start_counters[start_idx:end_idx]
        end_counters = end_counters[start_idx:end_idx]
        start_volumes = start_volumes[start_idx:end_idx]
        end_volumes = end_volumes[start_idx:end_idx]

        frame_df = pd.DataFrame({
            'state': states,
            'start': [start_counter - self.shift_frame for start_counter in start_counters],
            'end': [end_counter - self.shift_frame for end_counter in end_counters]
        })
        volume_df = pd.DataFrame({
            'state': states,
            'start': start_volumes,
            'end': end_volumes
        })
        return frame_df, volume_df
        
    def process_single_file(self, h5_filename):
        """
        Process h5 file that is saved all in one.
        """
        df = self._read_data_from_single(h5_filename)
        states, start_counters, end_counters = self._get_start_end_counters_single(df)
        max_frame_number = None
        if self.tiff_num_dict and worm_id in self.tiff_num_dict:
            trial_idx = self.worm_trial_counters[worm_id]
            if trial_idx < len(self.tiff_num_dict[worm_id]):
                max_frame_number = self.tiff_num_dict[worm_id][trial_idx]
            self.worm_trial_counters[worm_id] += 1
        frame_df, volume_df = self.generate_volume_df(states, start_counters, end_counters, max_frame_number=max_frame_number)
        
        # Create output paths
        states_excel_path = os.path.join(self.output_folder, "output_states.xlsx")
        volumes_excel_path = os.path.join(self.output_folder, "output_volumes.xlsx")
        
        # Write to Excel files
        states_write_mode = 'a' if os.path.exists(states_excel_path) else 'w'
        volumes_write_mode = 'a' if os.path.exists(volumes_excel_path) else 'w'
        with pd.ExcelWriter(states_excel_path, engine='openpyxl', mode=states_write_mode) as states_writer, \
             pd.ExcelWriter(volumes_excel_path, engine='openpyxl', mode=volumes_write_mode) as volumes_writer:
            existing_sheets_states = states_writer.book.sheetnames if states_writer.book else []
            existing_sheets_volumes = volumes_writer.book.sheetnames if volumes_writer.book else []
            original_worm_id = worm_id
            suffix = 1
            while worm_id in existing_sheets_states:
                worm_id = f"{original_worm_id}_{suffix}"
                suffix += 1
            
            while worm_id in existing_sheets_volumes:
                worm_id = f"{original_worm_id}_{suffix}"
                suffix += 1

            frame_df.to_excel(states_writer, sheet_name=worm_id, index=False)
            volume_df.to_excel(volumes_writer, sheet_name=worm_id, index=False)
            print(f"Processed {h5_filename} and saved to {states_excel_path} and {volumes_excel_path}.")
            print(f"Added sheet '{worm_id}' to both Excel files.")

    def _get_h5_filename_list(self, h5_folder, show_msg=True):
        h5_file_list = glob.glob(os.path.join(h5_folder, "*.h5"))
        h5_file_list.sort(key=lambda x: int(re.search(r'(\d+)', Path(x).stem).group(1)))

        if show_msg:
            print(f"Found {len(h5_file_list)} H5 files in folder {h5_folder}:")
            if len(h5_file_list) <= 10:
                for file in h5_file_list:
                    print(f"{h5_folder} files :{file}")
            else:
                for file in h5_file_list[:1]:
                    print(f"{h5_folder} files :{file}")
                print("...")
                for file in h5_file_list[-1:]:
                    print(f"{h5_folder} files :{file}")
        return h5_file_list
    
    def _read_data_from_multi(self, h5_filename):
        with h5py.File(h5_filename, "r") as h5_file:
            # get actual scan rate of the camera
            actual_scan_rate = h5_file.attrs["actual_scan_rate"]
            
            # read "/0_raw_data" if there is
            if "/0_raw_data" in h5_file:
                root_path = "/0_raw_data"
            else:
                root_path = ""

            # read dataset that needs to be extracted
            counter_data = h5_file[root_path+"/counter"][:, 0]
            counter_data_slave = h5_file[root_path+"/counter"][:, 1]
            dout_states_data = h5_file[root_path+"/dout_states"][:]
            
        counter_data = self.correct_counter(counter_data, counter_data_slave)

        df = pd.DataFrame({
            'counter': counter_data,
            'state': [tuple(row) for row in dout_states_data]
        })
        
        df_unique = df.drop_duplicates(subset='counter', keep='last')
        df_sorted = df_unique.sort_values(by='counter').reset_index(drop=True)
        df_sorted['state_name'] = df_sorted['state'].apply(lambda x: self.map_state(x))

        return df_sorted

    def _get_start_end_counters_multi(self, h5_folder):
        h5_file_list = self._get_h5_filename_list(h5_folder)
        states = []
        start_counters = []
        end_counters = []
        for h5_file_name in tqdm(h5_file_list, desc=f"Processing h5 files from {h5_folder}:"):
            df_sorted = self._read_data_from_multi(h5_file_name)
            states_split, start_counters_split, end_counters_split = self._get_start_end_counters_single(df_sorted)
            states.extend(states_split) # There will be adjacent duplicate elements. 
            start_counters.extend(start_counters_split)
            end_counters.extend(end_counters_split)

        df = pd.DataFrame({
            'state': states,
            'start': start_counters,
            'end': end_counters
        })

        is_not_continuous = is_not_continuous = (df['start'] != df['end'].shift()) & (df['start'] != df['end'].shift() + 1)
        df['state_change'] = (df['state'] != df['state'].shift()) | is_not_continuous
        df['group'] = df['state_change'].cumsum()

        df_merge_successive = df.groupby('group').agg({
            'state': 'first',
            'start': 'min',
            'end': 'max'
        }).reset_index(drop=True)

        return df_merge_successive['state'].tolist(), df_merge_successive['start'].tolist(), df_merge_successive['end'].tolist()
    
    def process_multi_files(self, h5_folder):
        """
        Process files that are splitted by seconds.
        """
        states, start_counters, end_counters = self._get_start_end_counters_multi(h5_folder)
        # Extract worm ID from the folder name
        worm_id = self.extract_worm_id(Path(h5_folder).name)
        max_frame_number = None
        if self.tiff_num_dict and worm_id in self.tiff_num_dict:
            trial_idx = self.worm_trial_counters[worm_id]
            if trial_idx < len(self.tiff_num_dict[worm_id]):
                max_frame_number = self.tiff_num_dict[worm_id][trial_idx]
            self.worm_trial_counters[worm_id] += 1
        frame_df, volume_df = self.generate_volume_df(states, start_counters, end_counters, max_frame_number=max_frame_number)

        # Create output paths
        states_excel_path = os.path.join(self.output_folder, "output_states.xlsx")
        volumes_excel_path = os.path.join(self.output_folder, "output_volumes.xlsx")

        # Write to Excel files
        states_write_mode = 'a' if os.path.exists(states_excel_path) else 'w'
        volumes_write_mode = 'a' if os.path.exists(volumes_excel_path) else 'w'
        with pd.ExcelWriter(states_excel_path, engine='openpyxl', mode=states_write_mode) as states_writer, \
             pd.ExcelWriter(volumes_excel_path, engine='openpyxl', mode=volumes_write_mode) as volumes_writer:       
            
            existing_sheets_states = states_writer.book.sheetnames if states_writer.book else []
            existing_sheets_volumes = volumes_writer.book.sheetnames if volumes_writer.book else []
            original_worm_id = worm_id
            suffix = 1
            while worm_id in existing_sheets_states:
                worm_id = f"{original_worm_id}_{suffix}"
                suffix += 1

            while worm_id in existing_sheets_volumes:
                worm_id = f"{original_worm_id}_{suffix}"
                suffix += 1

            frame_df.to_excel(states_writer, sheet_name=worm_id, index=False)
            volume_df.to_excel(volumes_writer, sheet_name=worm_id, index=False)
            print(f"Processed {Path(h5_folder).name} and saved to {states_excel_path} and {volumes_excel_path}")
            print(f"Added sheet '{worm_id}' to both Excel files.")

    def extract_channel_info(self):
        """
        Extract channel info from multiple h5 files.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if self.file_mode == 'single':
            if len(self.input_folder_list) != 1:
                raise ValueError("For single file mode, input_folder_list should contain exactly one folder.")
            for h5_file in tqdm(self._get_h5_filename_list(self.input_folder_list[0]), desc="Processing single h5 file:"):
                self.process_single_file(h5_file)
        elif self.file_mode == 'multiple':
            for h5_folder in tqdm(self.input_folder_list, desc="Processing h5 folders:"):
                self.process_multi_files(h5_folder)
        else:
            raise ValueError("FileMode must be either 'single' or 'multiple'.")

#%%
if __name__ == "__main__":
    extractor = ExtractChannelInfo(
        input_folder_list=[r"I:\WJH\infer\Ikrma_0730\w1\W1_2025-07-29_22-22-11_labjack"],
        output_folder=r"I:\WJH\infer\Ikrma_0730\w1\labjack_try",
        config_file=r"H:\Process_temporary\WJH\sensory_pipeline_python\channel_info_get\config.json",
        FileMode='multiple',
    )
    extractor.extract_channel_info()