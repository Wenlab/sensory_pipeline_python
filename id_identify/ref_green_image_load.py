if __name__ == "__main__":
    nas_folder_path = rf"//192.168.1.192/worm-tools/Jinghao-Wang/tea_experiment/20250421_EGCG_high"
    save_folder_path = rf"H:\Process_temporary\WJH\olfactory\ID\image_data\20250421_EGCG_high"

#%%
import sys
import os
from pathlib import Path
import numpy as np
if __name__ == "__main__":
    # Add the parent directory to the system path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.read_vols_using_dask import lazy_read_tiff_stack

#%%
def save_dask_array_as_npy(dask_array, output_path):
    # Convert the Dask array to a NumPy array
    numpy_array = dask_array.compute()
    # Ensure the output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the NumPy array to a .npy file
    np.save(output_path, numpy_array)

def batch_process_folder(folder_path, output_path):
    # Get all subfolders in the directory
    subfolders = [f for f in Path(folder_path).iterdir() if f.is_dir()]
    
    for subfolder in subfolders:
        # Process each subfolder
        # if subfolder startswith("w"):
        if subfolder.name.startswith("w"):
            # Process the subfolder
            print(f"Processing {subfolder}...")
            tiff_dir_name = os.path.join(folder_path, subfolder.name)
            exp_path = tiff_dir_name
            red_tiff_path_ = rf"{exp_path}\0_Camera-Red_VSC-10629"
            green_tiff_path_ = rf"{exp_path}\1_Camera-Green_VSC-09321"
            volume_read_params = dict(
                z_start_frame_number=0,
                z_end_frame_number=17,
                mod2_reverse=[False, False],
                img_width=1024,
                img_height=1024,
                frame_number_per_volume=20,
                img_dtype=np.uint16,
            )
            red = lazy_read_tiff_stack(red_tiff_path_, volume_read_params)
            green = lazy_read_tiff_stack(green_tiff_path_, volume_read_params)
            save_dask_array_as_npy(red[:10], os.path.join(output_path, subfolder.name, "red.npy"))
            save_dask_array_as_npy(green[:10], os.path.join(output_path, subfolder.name, "green.npy"))


#%%
if __name__ == "__main__":
    batch_process_folder(nas_folder_path, save_folder_path)