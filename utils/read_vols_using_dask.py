import os
import shutil
import numpy as np
from tqdm import tqdm
import dask.array as da
import napari
import tifffile
from pathlib import Path

#%%
def _read_frame(filenames_vol,**kwargs):
    return tifffile.imread( filenames_vol[0][0] )[np.newaxis,np.newaxis,:,:]

def extract_volume_numbers_from_dir(image_dir, frames_per_volume=20):
    volume_numbers = []
    frame_numbers = (
        {}
    )  # key: volume number, value: a set of frame numbers in the volume

    # Get a list of all files in the directory
    files = os.listdir(image_dir)

    # Sort the files in ascending order
    files.sort()

    # Iterate over the files
    for file in files:
        # Check if the file is a tiff image
        if file.endswith(".tif"):
            # Extract the frame number from the file name
            frame_number = int(file.split(".")[0])
            cur_volume_number = frame_number // frames_per_volume
            # add the frame number to the set of frame numbers for the current volume
            if cur_volume_number not in frame_numbers:
                frame_numbers[cur_volume_number] = set()
            frame_numbers[cur_volume_number].add(frame_number)

    # Iterate over the frame numbers
    for volume_number, frame_number_set in frame_numbers.items():
        # Check if the frame numbers are continuous
        if len(frame_number_set) == frames_per_volume:
            volume_numbers.append(volume_number)

    return volume_numbers


def get_filenames_vols(
    volume_numbers,
    tiff_root_path,
    img_height,
    img_width,
    img_dtype,
    frame_number_per_volume,
    z_start_frame_number,
    z_end_frame_number,
    mod2_reverse,
    show_progress=False,  # New parameter to control tqdm progress bar
):
    img_depth = z_end_frame_number - z_start_frame_number + 1
    # volumes_img = np.zeros(
    #     (len(volume_numbers), img_depth, img_height, img_width), dtype=img_dtype
    # )
    filenames_vols = []
    iterable = tqdm(volume_numbers) if show_progress else volume_numbers
    for index, volume_number in enumerate(iterable):
        filenames_frames = []
        frame_numbers = list(
            range(
                volume_number * frame_number_per_volume + z_start_frame_number,
                volume_number * frame_number_per_volume + z_end_frame_number + 1,
            )
        )
        if mod2_reverse[volume_number % 2]:
            frame_numbers = frame_numbers[::-1]
        for frame_number in frame_numbers:
            tiff_file_name = f"{frame_number:08d}.tif"
            # read tiff file and place them in volume_img
            tiff_file_path = os.path.join(tiff_root_path, tiff_file_name)
            filenames_frames.append(tiff_file_path)
            # volumes_img[index, frame_numbers.index(frame_number), :, :] = plt.imread(
            #     tiff_file_path
            # )
        filenames_vols.append(filenames_frames)
    return filenames_vols

def lazy_read_tiff_stack(
    tiff_path_,
    volume_read_params,
):
    # Extract volume numbers from the directory
    vols = extract_volume_numbers_from_dir(tiff_path_)
    file_names = get_filenames_vols(
        volume_numbers=vols,
        tiff_root_path=tiff_path_,
        show_progress=True,
        **volume_read_params,
    )
    valid_frames_per_volume = volume_read_params["z_end_frame_number"] - volume_read_params["z_start_frame_number"] + 1
    file_names_dask = da.from_array(file_names, chunks=(1, 1))
    images_dask = file_names_dask.map_blocks(
        _read_frame,
        chunks=da.core.normalize_chunks((1, 1, 1024, 1024), (len(vols), valid_frames_per_volume, 1024, 1024)),
        # multiple_files=True,
        new_axis=[2, 3],
        meta=np.array((), dtype=np.uint16),  # meta overwrites `dtype` argument
    )
    return images_dask

def display_channel_volume_data_in_napari(
    red_data,
    green_data,
    red_contrast_limits=(150, 400),
    green_contrast_limits=(150, 400),
    red_scale=(1, 5, 1, 1),
    green_scale=(1, 5, 1, -1),
    green_translate=(0, 0, -5, 1024 - 5),
):    
    viewer = napari.Viewer()
    if red_data is not None:
        red_layer = viewer.add_image(red_data, name="red", colormap="red")
        # red_layer.colormap = "red"
        red_layer.blending = "additive"
        red_layer.contrast_limits = red_contrast_limits
        red_layer.scale = red_scale
    if green_data is not None:
        green_layer = viewer.add_image(green_data, name="green", colormap="green")
        # green_layer.colormap = "green"
        green_layer.blending = "additive"
        green_layer.contrast_limits = green_contrast_limits
        green_layer.scale = green_scale
        green_layer.translate = green_translate
    return viewer

def show_volumes_in_napari(
    red_tiff_path,
    green_tiff_path,
    volume_read_params,
    use_visual_stack=True,

    visual_volume_count=None,
    volume_read_start=0,
    volume_read_interval=1,
    volume_read_end=None,

    red_contrast_limits=(150, 400),
    green_contrast_limits=(150, 400),
    red_scale=(1, 5, 1, 1),
    green_scale=(1, 5, 1, -1),
    green_translate=(0, 0, -5, 1024 - 5),
):  
    if not use_visual_stack:
        visual_volume_count=None
    if red_tiff_path is not None:
        red = lazy_read_tiff_stack(
            red_tiff_path, volume_read_params=volume_read_params
        )[volume_read_start:volume_read_end:volume_read_interval]
        if not use_visual_stack:
            red = red.compute()
    else:
        red = None
    if green_tiff_path is not None:
        green = lazy_read_tiff_stack(
            green_tiff_path,
            volume_read_params=volume_read_params,
    )[volume_read_start:volume_read_end:volume_read_interval]
        if not use_visual_stack:
            green = green.compute()
    else:
        green = None

    viewer = display_channel_volume_data_in_napari(
        red,
        green,
        red_contrast_limits=red_contrast_limits,
        green_contrast_limits=green_contrast_limits,
        red_scale=red_scale,
        green_scale=green_scale,
        green_translate=green_translate,
    )
    return viewer
            
def save_volumes_with_vol_range(
    tiff_path_,
    volume_read_params,
    output_folder,
    worm_name,
    date_info,
    start_volume_number=0,
    end_volume_number=None,
    show_progress=True
):
    """
    Save volumes in specified range with custom naming convention.
    
    Parameters:
    -----------
    tiff_path_ : str
        Path to the tiff directory
    volume_read_params : dict
        Parameters for volume reading
    output_folder : str
        Output directory to save .npy files
    worm_name : str
        Name of the worm for file naming
    start_volume_number : int, default=0
        Starting volume number (inclusive)
    end_volume_number : int, default=None
        Ending volume number (inclusive). If None, process all available volumes
    show_progress : bool, default=True
        Whether to show progress bar
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract all available volume numbers from the directory
    all_vols = extract_volume_numbers_from_dir(tiff_path_)
    
    # Filter volumes based on start and end range
    if end_volume_number is None:
        end_volume_number = max(all_vols) if all_vols else 0
    
    selected_vols = [vol for vol in all_vols if start_volume_number <= vol <= end_volume_number]
    
    if not selected_vols:
        print(f"No volumes found in range {start_volume_number}-{end_volume_number}")
        return
    
    print(f"Processing volumes {min(selected_vols)} to {max(selected_vols)} ({len(selected_vols)} volumes)")
    
    # Get filenames for selected volumes
    file_names = get_filenames_vols(
        volume_numbers=selected_vols,
        tiff_root_path=tiff_path_,
        show_progress=False,  # We'll show progress in the saving loop
        **volume_read_params,
    )
    
    valid_frames_per_volume = volume_read_params["z_end_frame_number"] - volume_read_params["z_start_frame_number"] + 1
    
    # Create dask array
    file_names_dask = da.from_array(file_names, chunks=(1, 1))
    images_dask = file_names_dask.map_blocks(
        _read_frame,
        chunks=da.core.normalize_chunks((1, 1, 1024, 1024), (len(selected_vols), valid_frames_per_volume, 1024, 1024)),
        new_axis=[2, 3],
        meta=np.array((), dtype=np.uint16),
    )
    
    # Save each volume individually with progress bar
    iterable = tqdm(enumerate(selected_vols), total=len(selected_vols), desc="Saving volumes") if show_progress else enumerate(selected_vols)
    
    for idx, vol_num in iterable:
        # Extract single volume from dask array
        single_volume = images_dask[idx, :, :, :]  # Shape: (z, y, x)
        
        # Compute the volume (load into memory)
        volume_data = single_volume.compute()
        
        # Transpose from (z, y, x) to (y, x, z)
        volume_transposed = volume_data.transpose(1, 2, 0)
        
        # Create filename following the specified convention
        filename = f"ImgStk001_dk001_{worm_name}_Dt{date_info}_{vol_num:06d}.npy"
        filepath = output_path / filename
        
        # Save the volume
        np.save(filepath, volume_transposed)
    
    print(f"Successfully saved {len(selected_vols)} volumes to {output_folder}")

# Example usage function
def process_experiment_with_vol_range(
    exp_path,
    output_folder,
    worm_name,
    date_info,
    start_volume=0,
    end_volume=None,
    camera_type="red"  # "red" or "green"
):
    """
    Process a single experiment with volume range specification.
    
    Parameters:
    -----------
    exp_path : str
        Path to experiment directory
    output_folder : str
        Output directory
    worm_name : str
        Worm name for file naming
    start_volume : int
        Starting volume number
    end_volume : int or None
        Ending volume number (None for all available)
    camera_type : str
        "red" or "green"
    """
    if camera_type == "red":
        tiff_path_ = os.path.join(exp_path, "0_Camera-Red_VSC-10629")
    elif camera_type == "green":
        tiff_path_ = os.path.join(exp_path, "1_Camera-Green_VSC-09321")
    else:
        raise ValueError("camera_type must be 'red' or 'green'")
    
    volume_read_params = dict(
        z_start_frame_number=0,
        z_end_frame_number=17,
        mod2_reverse=[False, False],
        img_width=1024,
        img_height=1024,
        frame_number_per_volume=20,
        img_dtype=np.uint16,
    )
    
    # Create camera-specific output folder
    camera_output_folder = os.path.join(output_folder, camera_type)
    
    save_volumes_with_vol_range(
        tiff_path_=tiff_path_,
        volume_read_params=volume_read_params,
        output_folder=camera_output_folder,
        worm_name=worm_name,
        date_info=date_info,
        start_volume_number=start_volume,
        end_volume_number=end_volume,
        show_progress=True
    )

def save_volumes_with_tiff_range(
    tiff_path_,
    volume_read_params,
    output_folder,
    worm_name,
    date_info,
    start_tiff_number=0,
    end_tiff_number=None,
    show_progress=True,
    filename_pattern=None,
    sequence_start=0,
    extra_format_kwargs=None,
):
    """
    Save volumes in specified tiff file range with custom naming convention.
    
    Parameters:
    -----------
    tiff_path_ : str
        Path to the tiff directory
    volume_read_params : dict
        Parameters for volume reading
    output_folder : str
        Output directory to save .npy files
    worm_name : str
        Name of the worm for file naming
    date_info : str
        Date information for file naming
    start_tiff_number : int, default=0
        Starting tiff file number (inclusive)
    end_tiff_number : int, default=None
        Ending tiff file number (inclusive). If None, process all available tiffs
    show_progress : bool, default=True
        Whether to show progress bar
    filename_pattern : str, optional
        Python format string for output filenames. Available keys: 'worm', 'date',
        'vol', 'volume', 'seq', 'sequence', plus any provided via extra_format_kwargs.
        Defaults to "ImgStk001_dk001_{worm}_Dt{date}_{vol:06d}.npy".
    sequence_start : int, default=0
        Starting index for the sequential 'seq' placeholder when multiple volumes are saved.
    extra_format_kwargs : dict, optional
        Additional key/value pairs made available to filename_pattern formatting.

    Returns
    -------
    list[Path]
        List of file paths that were written.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename_pattern is None:
        filename_pattern = "ImgStk001_dk001_{worm}_Dt{date}_{vol:06d}.npy"

    extra_format_kwargs = extra_format_kwargs or {}
    
    # Get frame_number_per_volume from volume_read_params
    frame_number_per_volume = volume_read_params["frame_number_per_volume"]
    
    # Calculate which volumes contain the specified tiff range
    start_volume = start_tiff_number // frame_number_per_volume
    if end_tiff_number is None:
        # Get all available volumes and find the maximum tiff number
        all_vols = extract_volume_numbers_from_dir(tiff_path_)
        if not all_vols:
            print("No volumes found in directory")
            return []
        max_volume = max(all_vols)
        end_tiff_number = (max_volume + 1) * frame_number_per_volume - 1
    
    end_volume = end_tiff_number // frame_number_per_volume
    
    # Get all volumes that might contain tiffs in our range
    all_vols = extract_volume_numbers_from_dir(tiff_path_)
    selected_vols = [vol for vol in all_vols if start_volume <= vol <= end_volume]
    
    if not selected_vols:
        print(f"No volumes found for tiff range {start_tiff_number}-{end_tiff_number}")
        return []
    
    print(f"Processing tiff files {start_tiff_number} to {end_tiff_number}")
    print(f"This spans volumes {min(selected_vols)} to {max(selected_vols)} ({len(selected_vols)} volumes)")
    
    # Get filenames for selected volumes
    file_names = get_filenames_vols(
        volume_numbers=selected_vols,
        tiff_root_path=tiff_path_,
        show_progress=False,
        **volume_read_params,
    )
    
    valid_frames_per_volume = volume_read_params["z_end_frame_number"] - volume_read_params["z_start_frame_number"] + 1
    
    # Create dask array
    file_names_dask = da.from_array(file_names, chunks=(1, 1))
    images_dask = file_names_dask.map_blocks(
        _read_frame,
        chunks=da.core.normalize_chunks((1, 1, 1024, 1024), (len(selected_vols), valid_frames_per_volume, 1024, 1024)),
        new_axis=[2, 3],
        meta=np.array((), dtype=np.uint16),
    )
    
    # Filter volumes based on actual tiff numbers they contain
    volumes_to_save = []
    for idx, vol_num in enumerate(selected_vols):
        # Calculate the tiff range for this volume
        vol_start_tiff = vol_num * frame_number_per_volume + volume_read_params["z_start_frame_number"]
        vol_end_tiff = vol_num * frame_number_per_volume + volume_read_params["z_end_frame_number"]
        
        # Check if this volume's tiff range overlaps with our desired range
        if not (vol_end_tiff < start_tiff_number or vol_start_tiff > end_tiff_number):
            volumes_to_save.append((idx, vol_num))
    
    # Save volumes with progress bar
    iterable = tqdm(volumes_to_save, desc="Saving volumes") if show_progress else volumes_to_save
    
    saved_files = []
    
    for offset, (idx, vol_num) in enumerate(iterable):
        # Extract single volume from dask array
        single_volume = images_dask[idx, :, :, :]  # Shape: (z, y, x)
        
        # Compute the volume (load into memory)
        volume_data = single_volume.compute()
        
        # Transpose from (z, y, x) to (y, x, z)
        volume_transposed = volume_data.transpose(1, 2, 0)
        
        # Prepare formatting context for filename
        seq_index = sequence_start + offset
        format_context = {
            "worm": worm_name,
            "date": date_info,
            "vol": vol_num,
            "volume": vol_num,
            "seq": seq_index,
            "sequence": seq_index,
            "sequence_index": seq_index,
            "volume_index": vol_num,
        }
        format_context.update(extra_format_kwargs)

        try:
            filename = filename_pattern.format(**format_context)
        except KeyError as exc:
            missing_key = exc.args[0]
            raise KeyError(
                f"Missing placeholder '{missing_key}' for filename_pattern '{filename_pattern}'"
            ) from exc

        filepath = output_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the volume
        np.save(filepath, volume_transposed)
        saved_files.append(filepath)
    
    print(f"Successfully saved {len(saved_files)} volumes to {output_folder}")

    return saved_files

# Updated wrapper function
def process_experiment_with_tiff_range(
    exp_path,
    output_folder,
    worm_name,
    date_info,
    start_tiff_number=0,
    end_tiff_number=None,
    camera_type="red"  # "red" or "green"
):
    """
    Process a single experiment with tiff file range specification.
    
    Parameters:
    -----------
    exp_path : str
        Path to experiment directory
    output_folder : str
        Output directory
    worm_name : str
        Worm name for file naming
    date_info : str
        Date information for file naming
    start_tiff_number : int
        Starting tiff file number
    end_tiff_number : int or None
        Ending tiff file number (None for all available)
    camera_type : str
        "red" or "green"
    """
    if camera_type == "red":
        tiff_path_ = os.path.join(exp_path, "0_Camera-Red_VSC-10629")
    elif camera_type == "green":
        tiff_path_ = os.path.join(exp_path, "1_Camera-Green_VSC-09321")
    else:
        raise ValueError("camera_type must be 'red' or 'green'")
    
    volume_read_params = dict(
        z_start_frame_number=0,
        z_end_frame_number=17,
        mod2_reverse=[False, False],
        img_width=1024,
        img_height=1024,
        frame_number_per_volume=20,
        img_dtype=np.uint16,
    )
    
    # Create camera-specific output folder
    camera_output_folder = os.path.join(output_folder, 'volume', f"ImgStk001_dk001_{worm_name}_Dt{date_info}")
    
    save_volumes_with_tiff_range(
        tiff_path_=tiff_path_,
        volume_read_params=volume_read_params,
        output_folder=camera_output_folder,
        worm_name=worm_name,
        date_info=date_info,
        start_tiff_number=start_tiff_number,
        end_tiff_number=end_tiff_number,
        show_progress=True
    )

#%%
if __name__ == "__main__":
    # exp_path = tiff_dir_name
    # red_tiff_path_ = rf"{exp_path}\0_Camera-Red_VSC-10629"
    # green_tiff_path_ = rf"{exp_path}\1_Camera-Green_VSC-09321"
    # volume_read_params = dict(
    #     z_start_frame_number=0,
    #     z_end_frame_number=17,
    #     mod2_reverse=[False, False],
    #     img_width=1024,
    #     img_height=1024,
    #     frame_number_per_volume=20,
    #     img_dtype=np.uint16,
    # )
    # red = lazy_read_tiff_stack(red_tiff_path_, volume_read_params)
    # green = lazy_read_tiff_stack(green_tiff_path_, volume_read_params)
    # worm_name = "worm1"
    # date_info = "202504"
    # # Process red camera, volumes 10-50
    # process_experiment_with_vol_range(
    #     exp_path=exp_path,
    #     output_folder=output_folder,
    #     worm_name=worm_name,
    #     date_info=date_info,
    #     start_volume=10,
    #     end_volume=50,
    #     camera_type="red"
    # )
    
    # # Process green camera, volumes 10-50
    # process_experiment_with_vol_range(
    #     exp_path=exp_path,
    #     output_folder=output_folder,
    #     worm_name=worm_name,
    #     date_info=date_info,
    #     start_volume=10,
    #     end_volume=50,
    #     camera_type="green"
    # )

    # process_experiment_with_tiff_range(
    #     exp_path=exp_path,
    #     output_folder=output_folder,
    #     worm_name=worm_name,
    #     date_info=date_info,
    #     start_tiff_number=200,
    #     end_tiff_number=1000,
    #     camera_type="green"
    # )
    red_tiff_path_ = None
    green_tiff_path_ = r"I:\WJH\flavor\signal_check\20250705\w1"

    use_visual_stack_ = True
    visual_volume_count_ = None
    volume_read_start_ = 0
    volume_read_interval_ = 1
    volume_read_end_ = None

    volume_read_params_ = dict(
        z_start_frame_number=0,
        z_end_frame_number=17,
        mod2_reverse=[False, False],
        img_width=1024,
        img_height=1024,
        frame_number_per_volume=20,
        img_dtype=np.uint16,
    )

    viewer = show_volumes_in_napari(
        red_tiff_path=red_tiff_path_,
        green_tiff_path=green_tiff_path_,
        volume_read_params=volume_read_params_,
        use_visual_stack=use_visual_stack_,
        visual_volume_count=visual_volume_count_,
        volume_read_start=volume_read_start_,
        volume_read_interval=volume_read_interval_,
        volume_read_end=volume_read_end_,
        red_contrast_limits=(150, 400),
        green_contrast_limits=(150, 300),
        red_scale=(1, 1, 1, 1),
        green_scale=(1, 1, 1, -1),
        # green_translate=(0, 0, -5, 1024 - 5),
    )
    napari.run()