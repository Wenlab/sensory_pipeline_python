if __name__ == "__main__":
    h5_file_folder = r"H:\Process_temporary\WJH\olfactory\Code\back_up\Labjack_channel_info_extraction\data"
    config_file = r"H:\Process_temporary\WJH\sensory_pipeline_python\channel_info_get\config.json"
    output_folder = r"H:\Process_temporary\WJH\olfactory\Code\back_up\Labjack_channel_info_extraction\labjack"
    metadata_path = r"H:\Process_temporary\WJH\olfactory\Code\back_up\Labjack_channel_info_extraction\metadata.csv"

#%%
import os
import h5py
import pandas as pd
import math
import json
import re
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#%%
def load_config(config_path):
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
        "[0,0,0,0,0,0,0,0]": "All Off",
        "[1,1,1,0,0,0,0,0]": "Buffer",
        "[1,0,1,1,0,0,0,0]": "Buffer",
        "[1,0,1,0,1,0,0,0]": "Buffer",
        "[1,0,1,0,0,1,0,0]": "Buffer",
        "[1,0,1,0,0,0,1,0]": "Buffer",
        "[1,0,1,0,0,0,0,1]": "Buffer",
        "[0,1,1,1,0,0,0,0]": "Odor1",
        "[0,1,1,0,1,0,0,0]": "Odor2",
        "[0,1,1,0,0,1,0,0]": "Odor3",
        "[0,1,1,0,0,0,1,0]": "Odor4",
        "[0,1,1,0,0,0,0,1]": "Odor5"
    }

    default_slice_number = 20

    default_slice_number = 20

    if not config_path:
        return default_channel_meanings, default_state_mappings, default_slice_number

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        channel_meanings = config.get("channel_meanings", default_channel_meanings)
        state_mappings = config.get("state_mappings", default_state_mappings)
        slice_number = config.get("slice_number", default_slice_number)
        
        # 转换state_mappings的键从字符串列表到元组
        converted_state_mappings = {}
        for key, value in state_mappings.items():
            # 移除括号并分割数字
            key_tuple = tuple(int(x) for x in key.strip("[]").split(","))
            converted_state_mappings[key_tuple] = value
        
        # 转换channel_meanings的键从字符串到整数
        converted_channel_meanings = {int(k): v for k, v in channel_meanings.items()}
        
        return converted_channel_meanings, converted_state_mappings, slice_number
    except Exception as e:
        print(f"Error loading config file: {e}")
        return default_channel_meanings, default_state_mappings
    
def map_state(state, state_mappings):
    return state_mappings.get(state, "Unknown State")

def process_hdf5(file_path, state_mappings, slice_number = 20):
    with h5py.File(file_path, 'r') as f:
        counter_dataset = f['counter'][:]
        dout_states_dataset = f['dout_states'][:]
    counters = counter_dataset[:, 0]
    df = pd.DataFrame({
        'counter': counters,
        'state': [tuple(row) for row in dout_states_dataset]
    })
    df_unique = df.drop_duplicates(subset='counter', keep='last')
    df_sorted = df_unique.sort_values(by='counter').reset_index(drop=True)
    df_sorted['state_name'] = df_sorted['state'].apply(lambda x: map_state(x, state_mappings))
    states = []
    start_counters = []
    end_counters = []
    current_state = None
    current_start = None
    for _, row in df_sorted.iterrows():
        state = row['state_name']
        counter = row['counter']
        if current_state is None:
            current_state = state
            current_start = counter
        elif state != current_state:
            states.append(current_state)
            start_counters.append(current_start)
            end_counters.append(prev_counter)
            current_state = state
            current_start = counter
        prev_counter = counter
    if current_state is not None:
        states.append(current_state)
        start_counters.append(current_start)
        end_counters.append(prev_counter)
    adjust_frame = end_counters[0]
    start_volumes = [math.ceil((x - adjust_frame)/slice_number) for x in start_counters]
    end_volumes = [math.ceil((x - adjust_frame)/slice_number) for x in end_counters]

    if states[-1] == 'All Off':
        states = states[1:-1]
        start_counters = start_counters[1:-1]
        end_counters = end_counters[1:-1]
        start_volumes = start_volumes[1:-1]
        end_volumes = end_volumes[1:-1]
    else:
        states = states[1:]
        start_counters = start_counters[1:]
        end_counters = end_counters[1:]
        start_volumes = start_volumes[1:]
        end_volumes = end_volumes[1:]
    result_df = pd.DataFrame({
        'state': states,
        'start': start_counters,
        'end': end_counters
    })
    volume_df = pd.DataFrame({
        'state': states,
        'start': start_volumes,
        'end': end_volumes
    })
    return result_df, volume_df

def extract_worm_id(filename):
    """
    从文件名中提取 'w' 后的编号，例如 'w1', 'w2' 等。
    返回 'w' 加编号的字符串，如果找不到则返回 'Unknown'.
    """
    match = re.search(r'w(\d+)', filename, re.IGNORECASE)
    if match:
        return f"w{match.group(1)}"
    else:
        return "Unknown"

def save_channel_meanings(output_folder_path, channel_meanings):
    """
    save channel meanings to txt
    """
    channel_meanings_file = os.path.join(output_folder_path, 'channel_meanings.txt')
    with open(channel_meanings_file, 'w') as f:
        for key, value in channel_meanings.items():
            f.write(f"Channel {key}: {value}\n")
    print(f"Saved channel meanings to {channel_meanings_file}")


def process_folder(folder_path, output_folder_path, matadata_path, channel_meanings, state_mappings, slice_number = 20):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # 保存通道含义到文本文件
    save_channel_meanings(output_folder_path, channel_meanings)
    
    matadata = pd.read_csv(matadata_path)
    frame_0 = matadata['frame_number'].values[0]
    
    # 创建ExcelWriter对象，用于写入多个sheet
    states_excel_path = os.path.join(output_folder_path, "output_states.xlsx")
    volumes_excel_path = os.path.join(output_folder_path, "output_volumes.xlsx")
    
    # 使用 with 语句确保Excel文件正确保存和关闭
    with pd.ExcelWriter(states_excel_path, engine='openpyxl') as states_writer, \
         pd.ExcelWriter(volumes_excel_path, engine='openpyxl') as volumes_writer:
        
        for file in os.listdir(folder_path):
            if file.endswith('.h5'):
                h5_path = os.path.join(folder_path, file)
                states_df, volumes_df = process_hdf5(h5_path, state_mappings, slice_number)
                states_df['start'] = states_df['start'] - frame_0
                states_df['end'] = states_df['end'] - frame_0
                min_value = (states_df['start'].min()//20)*20
                max_value = ((states_df['end'].max()+19)//20)*20-1
                print(f"File: {file}, Start: {min_value}, End: {max_value}")
                
                # 提取w编号作为sheet名称
                worm_id = extract_worm_id(file)
                
                # 检查sheet名称是否已经存在，避免重复
                existing_sheets_states = states_writer.book.sheetnames if states_writer.book else []
                existing_sheets_volumes = volumes_writer.book.sheetnames if volumes_writer.book else []
                
                # 如果sheet已存在，添加一个后缀数字以区分
                original_worm_id = worm_id
                suffix = 1
                while worm_id in existing_sheets_states:
                    worm_id = f"{original_worm_id}_{suffix}"
                    suffix += 1
                
                # 写入到对应的sheet
                states_df.to_excel(states_writer, sheet_name=worm_id, index=False)
                volumes_df.to_excel(volumes_writer, sheet_name=worm_id, index=False)
                print(f"Added sheet '{worm_id}' to both Excel files.")
    
    print(f"All states have been saved to {states_excel_path}")
    print(f"All volumes have been saved to {volumes_excel_path}")


def extract_channel_info(input_folder, output_folder, metadata_path, config_file):
    # Load configurations
    channel_meanings, state_mappings, slice_number = load_config(config_file if config_file != "No config file selected" else None)

    try:
        process_folder(input_folder, output_folder, metadata_path, channel_meanings, state_mappings, slice_number=slice_number)
    except Exception as e:
        print(f"an error occurred: {str(e)}")

#%%
if __name__ == "__main__":
    extract_channel_info(h5_file_folder,output_folder,metadata_path,config_file)