if __name__ == "__main":
    tiff_dir_name = r"\\192.168.1.192/worm-tools/Jinghao-Wang/tea_experiment/20250421_EGCG/w1_2025-04-21_11-39-40/"

#%%
import os
import numpy as np
from tqdm import tqdm
import dask.array as da
import napari
import tifffile

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

if __name__ == "__main__":
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