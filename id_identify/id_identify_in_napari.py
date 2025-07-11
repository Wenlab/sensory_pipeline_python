if __name__ == "__main__":
    green_file_path = r"H:\Process_temporary\WJH\olfactory\ID\image_data\20250522_LThe\w5_2025-05-22_16-37-58\green.npy"
    red_file_path = r"H:\Process_temporary\WJH\olfactory\ID\image_data\20250522_LThe\w5_2025-05-22_16-37-58\red.npy"
    aligned_volume_path = r"H:\Process_temporary\WJH\olfactory\ID\image_data\20250522_LThe\w5_2025-05-22_16-37-58\aligned_volumes_mip.npy"
    neuron_pt_tuple_path = r"H:\Process_temporary\WJH\olfactory\ID\image_data\20250522_LThe\w5_2025-05-22_16-37-58\all_neuron_pt_tuple.npy"
# %%
import numpy as np
import napari
import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from id_identify.box_region_npy import box_region_npy

# %%
def image_process(image_np, background_value = 110):
    """
    Denoise
    """
    image_np = image_np.astype(np.float32)  # Convert to float
    image_np[image_np < background_value] = 0  # Set background pixels to 0
    return image_np

def image_and_box_load(green_file_path, red_file_path,  neuron_pt_tuple_path , aligned_volume_path = None):
    """_summary_

    Args:
        green_file_path (_type_): green channel images (2vols)
        red_file_path (_type_): red channel image (2vols)
        neuron_pt_tuple_path (_type_, optional): neuron_pt_tuple(x,y,z*5,w,h,d)
        aligned_volume_path (_type_, optional): aligned MIP volume 

    Returns:
        processed green array and red array 
    """
    green = np.load(green_file_path)
    red = np.load(red_file_path)
    if aligned_volume_path is not None:
        aligned_volume_npy = np.load(aligned_volume_path)
        green[1] = aligned_volume_npy.transpose(2,0,1)

    green = image_process(green)
    red = image_process(red)
    neuron_pt_tuple = np.load(neuron_pt_tuple_path)
    output_shape = (2,18,1024,1024)
    mask = box_region_npy(neuron_pt_tuple[:2],output_shape=output_shape, if_save=False)

    return green, red, mask

def view_in_napari(green, red, mask, mask_name = "mask",green_scale=[1,5,1,-1], red_scale = [1,5,1,1], red_translate = [0,0,0,0], green_translate = [0,0,0,1024], mask_scale=[1,5,1,-1], mask_translate=[0,0,0,1024]):
    viewer = napari.Viewer()
    viewer.add_image(green, name='green', colormap='green', blending= 'additive', scale=green_scale, translate=green_translate, contrast_limits=[120,250])
    viewer.add_image(red, name='red', colormap='red', blending='additive', scale=red_scale, translate=red_translate, contrast_limits=[120,250])
    viewer.add_labels(mask, name=mask_name, blending='additive', scale=mask_scale, translate=mask_translate, opacity=1.0)
    return viewer

def id_identify_in_napari(green_file_path, red_file_path,  neuron_pt_tuple_path , aligned_volume_path = None, mask_name = "mask",green_scale=[1,5,1,-1], red_scale = [1,5,1,1], red_translate = [0,0,0,0], green_translate = [0,0,0,1024], mask_scale=[1,5,1,-1], mask_translate=[0,0,0,1024]):
    
    green, red, mask = image_and_box_load(green_file_path=green_file_path,
                                          red_file_path=red_file_path,
                                          neuron_pt_tuple_path=neuron_pt_tuple_path,
                                          aligned_volume_path=aligned_volume_path)
    
    viewer = view_in_napari(green=green,
                            red=red,
                            mask=mask,
                            mask_name=mask_name,
                            green_scale=green_scale,
                            red_scale=red_scale,
                            green_translate=green_translate,
                            red_translate=red_translate,
                            mask_scale=mask_scale,
                            mask_translate=mask_translate)

    
    return viewer

# %%
if __name__ == "__main__":
    viewer = id_identify_in_napari(
        green_file_path=green_file_path,
        red_file_path=red_file_path,
        neuron_pt_tuple_path=neuron_pt_tuple_path,
        aligned_volume_path=aligned_volume_path,
        mask_name="w5",
        green_scale=[1, 5, 1, -1],
        red_scale=[1, 5, 1, 1],
        red_translate=[0, 0, 0, 0],
        green_translate=[0, 0, 0, 1024],
        mask_scale=[1, 5, 1, -1],
        mask_translate=[0, 0, 0, 1024],
    )
    napari.run()