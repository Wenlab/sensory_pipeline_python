# %%
import numpy as np
import napari
import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from id_identify.draw_neuron_box import draw_neuron_box
from id_identify.trace_check_in_napari import _lazy_read_image_npy

# %%
def image_process(image_np, background_value = 110):
    """
    Denoise
    """
    image_np = image_np.astype(np.float32)  # Convert to float
    image_np[image_np < background_value] = 0  # Set background pixels to 0
    return image_np

def image_and_box_load(green_file_path, red_file_path,  neuron_pt_tuple_path , **kwargs):
    """
    Args:
        green_file_path (_type_): green channel volume npy
        red_file_path (_type_): red channel volume npy
        neuron_pt_tuple_path (_type_, optional): neuron pt tuple npy
        aligned_volume_path (_type_, optional): aligned MIP volume 
        aligned_neuron_pt_tuple (_type_, optional): neuron pt tuple for aligned MIP volume
        ref_green_folder_path (_type_, optional): reference green channel folder path
        ref_red_folder_path (_type_, optional): reference red channel folder path
    Returns:
        processed green array and red array 
    """
    aligned_volume_path = kwargs.get("aligned_volume_path", None)
    aligned_neuron_pt_tuple = kwargs.get("aligned_neuron_pt_tuple", None)
    ref_green_folder_path = kwargs.get("ref_green_folder_path", None)
    ref_red_folder_path = kwargs.get("ref_red_folder_path", None)
    ref_neuron_pt_tuple_path = kwargs.get("ref_neuron_pt_tuple", None)
    ex_green = np.load(green_file_path)
    ex_red = np.load(red_file_path)
    neuron_pt_tuple = np.load(neuron_pt_tuple_path)
    if ref_green_folder_path is not None:
        ref_green = _lazy_read_image_npy(ref_green_folder_path)
        ref_green = ref_green[0].compute()
        ex_green[0] = ref_green.transpose(2,0,1)
        if ref_neuron_pt_tuple_path is not None:
            ref_neuron_pt_tuple = np.load(ref_neuron_pt_tuple_path)
            neuron_pt_tuple[0] = ref_neuron_pt_tuple[0]
    if ref_red_folder_path is not None:
        ref_red = _lazy_read_image_npy(ref_red_folder_path)
        ref_red[0] = ref_red[0].compute()
        ex_red[0] = ref_red[0].transpose(2,0,1)
    if aligned_volume_path is not None:
        align_volume = np.load(aligned_volume_path)
        ex_green[1] = align_volume.transpose(2,0,1)
        if aligned_neuron_pt_tuple is not None:
            neuron_pt_tuple[1] = np.load(aligned_neuron_pt_tuple)
    ex_green = image_process(ex_green)
    ex_red = image_process(ex_red)
    output_shape = kwargs.get("output_shape", ex_green.shape)
    mask = draw_neuron_box(neuron_pt_tuple[:2],output_shape=output_shape, save_dir=None)
    mask = mask.compute()
    return ex_green, ex_red, mask

def view_in_napari(green, red, mask, mask_name = "mask",green_scale=[1,5,1,-1], red_scale = [1,5,1,1], red_translate = [0,0,0,0], green_translate = [0,0,0,1024], mask_scale=[1,5,1,-1], mask_translate=[0,0,0,1024]):
    viewer = napari.Viewer()
    viewer.add_image(green, name='green', colormap='green', blending= 'additive', scale=green_scale, translate=green_translate, contrast_limits=[120,250])
    viewer.add_image(red, name='red', colormap='red', blending='additive', scale=red_scale, translate=red_translate, contrast_limits=[120,250])
    viewer.add_labels(mask, name=mask_name, blending='translucent', scale=mask_scale, translate=mask_translate, opacity=1.0)
    return viewer

def id_identify_in_napari(green_file_path, red_file_path,  neuron_pt_tuple_path , **kwargs):
    """
    Args:
        green_file_path (_type_): green channel volume npy
        red_file_path (_type_): red channel volume npy
        neuron_pt_tuple_path (_type_, optional): neuron pt tuple npy
        aligned_volume_path (_type_, optional): aligned MIP volume 
        aligned_nueron_pt_tuple (_type_, optional): neuron pt tuple for aligned MIP volume
        mask_name (_type_, optional): mask name in napari viewer
        green_scale (_type_, optional): green channel scale in napari viewer
        red_scale (_type_, optional): red channel scale in napari viewer
        red_translate (_type_, optional): red channel translate in napari viewer
        green_translate (_type_, optional): green channel translate in napari viewer
        mask_scale (_type_, optional): mask scale in napari viewer
        mask_translate (_type_, optional): mask translate in napari viewer
    """
    mask_name = kwargs.get("mask_name", "mask")
    green_scale = kwargs.get("green_scale", [1,5,1,-1])
    red_scale = kwargs.get("red_scale", [1,5,1,1])
    red_translate = kwargs.get("red_translate", [0,0,0,0])
    green_translate = kwargs.get("green_translate", [0,0,0,1024])
    mask_scale = kwargs.get("mask_scale", [1,5,1,-1])
    mask_translate = kwargs.get("mask_translate", [0,0,0,1024])
    
    green, red, mask = image_and_box_load(green_file_path=green_file_path,
                                          red_file_path=red_file_path,
                                          neuron_pt_tuple_path=neuron_pt_tuple_path,
                                          **kwargs)
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
    worm_number = "w1"
    green_file_path =rf"H:\Process_temporary\WJH\olfactory\ID\image_data\20251104\{worm_number}\green.npy"
    red_file_path = rf"H:\Process_temporary\WJH\olfactory\ID\image_data\20251104\{worm_number}\red.npy"
    neuron_pt_tuple_path = rf"H:\Process_temporary\WJH\olfactory\ID\image_data\20251104\{worm_number}\output\ex_neuron_pt_tuple.npy"
    raw_args = {
        "mask_name": worm_number,
        "green_scale": [1, 5, 1, -1],
        "green_translate": [0, 0, 0, 1024],
        "red_scale": [1, 5, 1, 1],
        "mask_scale": [1, 5, 1, -1],
        "mask_translate": [0, 0, 0, 1024],
        "aligned_volume_path": None,
        "aligned_neuron_pt_tuple": None,
        "ref_green_folder_path": rf"H:\Process_temporary\WJH\olfactory\ID\image_data\20251104\{worm_number}\ref",
        "ref_red_folder_path":  rf"H:\Process_temporary\WJH\olfactory\ID\image_data\20251104\{worm_number}\ref_red",
        "ref_neuron_pt_tuple": rf"H:\Process_temporary\WJH\olfactory\ID\image_data\20251104\{worm_number}\output\reference_inference_results\ref_neuron_pt_tuple_filled.npy",
    }
    viewer = id_identify_in_napari(green_file_path=green_file_path,
                            red_file_path=red_file_path,
                            neuron_pt_tuple_path=neuron_pt_tuple_path,
                            **raw_args)
    napari.run()