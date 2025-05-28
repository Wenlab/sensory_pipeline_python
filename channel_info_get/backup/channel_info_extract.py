import os
import h5py
import pandas as pd
import math
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import re
import json

# --------------------- Data Processing ---------------------
def load_config(config_path):
    """
    从JSON配置文件加载channel_meanings和state_mappings。
    如果配置文件无效或不存在，返回默认值。
    """
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
        messagebox.showerror("配置文件错误", f"加载配置文件时出错: {str(e)}\n将使用默认配置。")
        return default_channel_meanings, default_state_mappings

def map_state(state_tuple, state_mappings):
    return state_mappings.get(state_tuple, 'ERROR')

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

def process_folder(folder_path, output_folder_path, matadata, channel_meanings, state_mappings, slice_number = 20):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # 保存通道含义到文本文件
    save_channel_meanings(output_folder_path, channel_meanings)
    
    matadata = pd.read_csv(matadata)
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

# --------------------- Save Channel Meanings to Text ---------------------
def save_channel_meanings(output_folder_path, channel_meanings):
    channel_meanings_file = os.path.join(output_folder_path, 'channel_meanings.txt')
    with open(channel_meanings_file, 'w') as f:
        for key, value in channel_meanings.items():
            f.write(f"Channel {key}: {value}\n")
    print(f"Saved channel meanings to {channel_meanings_file}")

# --------------------- GUI ---------------------
class FileSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File and Folder Selection")
        self.root.geometry("500x400")  # 调整窗口大小以适应新增内容

        # input folder selection
        self.input_label = tk.Label(root, text="Select Input Folder")
        self.input_label.pack(pady=5)
        self.input_button = tk.Button(root, text="Browse Input", command=self.select_input_folder)
        self.input_button.pack(pady=5)
        self.input_path = tk.StringVar()
        self.input_path.set("No folder selected")

        # output folder selection
        self.output_label = tk.Label(root, text="Select Output Folder")
        self.output_label.pack(pady=5)
        self.output_button = tk.Button(root, text="Browse Output", command=self.select_output_folder)
        self.output_button.pack(pady=5)
        self.output_path = tk.StringVar()
        self.output_path.set("No folder selected")

        # metadata file selection
        self.metadata_label = tk.Label(root, text="Select Metadata File")
        self.metadata_label.pack(pady=5)
        self.metadata_button = tk.Button(root, text="Browse Metadata", command=self.select_metadata_file)
        self.metadata_button.pack(pady=5)
        self.metadata_path = tk.StringVar()
        self.metadata_path.set("No file selected")

        # config file selection
        self.config_label = tk.Label(root, text="Select Config File (Optional)")
        self.config_label.pack(pady=5)
        self.config_button = tk.Button(root, text="Browse Config", command=self.select_config_file)
        self.config_button.pack(pady=5)
        self.config_path = tk.StringVar()
        self.config_path.set("No config file selected")

        # Submit button to process
        self.submit_button = tk.Button(root, text="Submit", command=self.submit)
        self.submit_button.pack(pady=20)

    def select_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_path.set(folder)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path.set(folder)

    def select_metadata_file(self):
        file = filedialog.askopenfilename(title="Select Metadata CSV File", filetypes=[("CSV Files", "*.csv")])
        if file:
            self.metadata_path.set(file)

    def select_config_file(self):
        file = filedialog.askopenfilename(title="Select Config JSON File", filetypes=[("JSON Files", "*.json")])
        if file:
            self.config_path.set(file)

    def submit(self):
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()
        metadata_file = self.metadata_path.get()
        config_file = self.config_path.get()


        # Check if required fields are filled
        if input_folder == "No folder selected" or output_folder == "No folder selected" or metadata_file == "No file selected":
            messagebox.showerror("Error", "Please select input folder, output folder, and metadata file before submitting!")
            print("Please select input folder, output folder, and metadata file before submitting!")
            return

        # Load configurations
        channel_meanings, state_mappings, slice_number = load_config(config_file if config_file != "No config file selected" else None)

        # Confirm with the user
        messagebox.showinfo("Success", f"Input: {input_folder}\nOutput: {output_folder}\nMetadata: {metadata_file}\nConfig: {config_file if config_file != 'No config file selected' else 'Default'}\nProceeding with processing...")

        try:
            process_folder(input_folder, output_folder, metadata_file, channel_meanings, state_mappings, slice_number=slice_number)
            messagebox.showinfo("Completed", "Processing completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"An error occurred: {str(e)}")
            
# --------------------- Main Execution ---------------------
def main():
    root = tk.Tk()
    app = FileSelectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
