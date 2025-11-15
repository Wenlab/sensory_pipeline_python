import numpy as np
import napari
import sys
import os
from dask import delayed
import dask.array as da
from glob import glob
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.read_vols_using_dask import show_volumes_in_napari
from id_identify.draw_neuron_box import draw_neuron_box

#%%
def _lazy_read_neuro_boxes(neuron_boxes_path):
    '''
    Lazy load neuron boxes from npy file
    Args:
        neuron_boxes_path: path to neuron boxes npy file'
    Returns:
        dask array of neuron boxes
    '''
    file_list = sorted(glob(os.path.join(neuron_boxes_path, '*.npy')))
    sample_volume = np.load(file_list[0])
    volume_shape = sample_volume.shape
    volume_dtype = sample_volume.dtype
    lazy_loader = delayed(np.load, pure=True)
    lazy_volumes = []
    for file_path in file_list:
        dask_volume = da.from_delayed(
            lazy_loader(file_path),
            shape=volume_shape,
            dtype=volume_dtype,
        )
        lazy_volumes.append(dask_volume)

    dask_volumes = da.stack(lazy_volumes, axis=0)

    return dask_volumes

def _lazy_read_image_npy(image_npy_path):
    '''
    Lazy load image volume from npy file
    Args:
        image_npy_path: path to image npy file
    Returns:
        dask array of image volume
    '''
    # glob all npy files in the directory(including subdirectories)
    file_list = sorted(glob(os.path.join(image_npy_path, '**', '*.npy'), recursive=True))
    sample_volume = np.load(file_list[0])
    volume_shape = sample_volume.shape
    volume_dtype = sample_volume.dtype
    lazy_loader = delayed(np.load, pure=True)
    lazy_volumes = []
    for file_path in file_list:
        dask_volume = da.from_delayed(
            lazy_loader(file_path),
            shape=volume_shape,
            dtype=volume_dtype,
        )
        lazy_volumes.append(dask_volume)
    dask_volumes = da.stack(lazy_volumes, axis=0)
    return dask_volumes

def trace_check_in_napari_tiff(
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
    red_scale=(1, 1, 1, 1),
    green_scale=(1, 1, 1, -1),
    green_translate=(0, 0, 0, 1024), 

    neuron_pt_tuple_path=None,
    neuron_boxes_path=None,
    label_scale=(1, 1, 1, -1),
    label_translate=(0, 0, 0, 1024),
):
    '''
    Args:
        red_tiff_path: path to red channel tiff path
        green_tiff_path: path to green channel tiff path
        volume_read_params: dict, parameters for reading volumes
        use_visual_stack: bool, whether to use dask to lazy load volumes
        visual_volume_count: visual volume number if use_visual_stack is True
        volume_read_start: starting volume index to read
        volume_read_interval: interval between volumes to read
        volume_read_end: ending volume index to read, None for all volumes
    '''
    viewer = show_volumes_in_napari(
        red_tiff_path=red_tiff_path,
        green_tiff_path=green_tiff_path,
        volume_read_params=volume_read_params,
        use_visual_stack=use_visual_stack,
        visual_volume_count=visual_volume_count,
        volume_read_start=volume_read_start,
        volume_read_interval=volume_read_interval,
        volume_read_end=volume_read_end,
        red_contrast_limits=red_contrast_limits,
        green_contrast_limits=green_contrast_limits,
        red_scale=red_scale,
        green_scale=green_scale,
        green_translate=green_translate,
    )

    if neuron_boxes_path is not None:
        neuron_boxes = _lazy_read_neuro_boxes(neuron_boxes_path)
        viewer.add_labels(
            neuron_boxes,
            name="neuron_boxes",
            blending='additive',
            scale=label_scale,
            translate=label_translate,
            opacity=1.0,
        )
    elif neuron_pt_tuple_path is not None:
        neuron_pt_tuple = np.load(neuron_pt_tuple_path)
        output_shape = (neuron_pt_tuple.shape[0], volume_read_params['z_end_frame_number'] - volume_read_params['z_start_frame_number'] + 1, 1024, 1024)
        neuron_boxes = draw_neuron_box(neuron_pt_tuple, output_shape=output_shape, save_dir=None)
        viewer.add_labels(
            neuron_boxes,
            name="neuron_boxes",
            blending='additive',
            scale=label_scale,
            translate=label_translate,
            opacity=1.0,
        )
    else:
        print("No neuron boxes or neuron point tuple provided. Skipping label addition.")

    return viewer

def trace_check_in_napari_npy(
    image_npy_path,
    use_visual_stack=True,

    visual_volume_count=None,
    volume_read_start=0,
    volume_read_interval=1,
    volume_read_end=None,

    image_contrast_limit=(150, 400),
    image_scale=(1, 5, 1, 1),
    neuron_pt_tuple_path=None,
    neuron_boxes_path=None,
    label_scale=(1, 5, 1, 1),
):
    '''
    Args:
        red_npy_path: path to red channel npy file
        green_npy_path: path to green channel npy file
        use_visual_stack: bool, whether to use dask to lazy load volumes
        visual_volume_count: visual volume number if use_visual_stack is True
        volume_read_start: starting volume index to read
        volume_read_interval: interval between volumes to read
        volume_read_end: ending volume index to read, None for all volumes
    '''
    volume_dask = _lazy_read_image_npy(image_npy_path)
    volume_dask = volume_dask.transpose(0,3,1,2) # t,y,x,z to t,z,y,x
    viewer = napari.Viewer()
    if use_visual_stack:
        viewer.add_image(
            volume_dask[
                volume_read_start:volume_read_end:volume_read_interval
            ][:visual_volume_count],
            contrast_limits=image_contrast_limit,
            name='image_channel',
            scale=image_scale,
        )
    else:
        volume = volume_dask[
            volume_read_start:volume_read_end:volume_read_interval].compute()
        viewer.add_image(
            volume,
            contrast_limits=image_contrast_limit,
            name='image_channel',
            scale=image_scale,
        )

    if neuron_boxes_path is not None:
        neuron_boxes = _lazy_read_neuro_boxes(neuron_boxes_path)
        viewer.add_labels(
            neuron_boxes[volume_read_start:volume_read_end:volume_read_interval].compute() if not use_visual_stack else neuron_boxes[
                volume_read_start:volume_read_end:volume_read_interval],
            name="neuron_boxes",
            blending='additive',
            scale=label_scale,
            opacity=1.0,
        )
    elif neuron_pt_tuple_path is not None:
        neuron_pt_tuple = np.load(neuron_pt_tuple_path)
        output_shape = (neuron_pt_tuple.shape[0], neuron_pt_tuple.shape[1], 1024, 1024)       
        neuron_boxes = draw_neuron_box(neuron_pt_tuple, output_shape=output_shape, save_dir=None)
        viewer.add_labels(
            neuron_boxes[volume_read_start:volume_read_end:volume_read_interval].compute() if not use_visual_stack else neuron_boxes[
                volume_read_start:volume_read_end:volume_read_interval],
            name="neuron_boxes",
            blending='additive',
            scale=label_scale,
            opacity=1.0,
        )
    else:
        print("No neuron boxes or neuron point tuple provided. Skipping label addition.")

    return viewer

if __name__ == "__main__":
    red_tiff_path_ = None
    green_tiff_path_ = r"I:\WJH\flavor\signal_check\20250705\w1"

    use_visual_stack_ = True
    visual_volume_count_ = None
    volume_read_start_ = 0
    volume_read_interval_ = 1
    volume_read_end_ = None

    napari_settings = dict(
        red_contrast_limits=(150, 300),
        green_contrast_limits=(150, 300),
        red_scale=(1, 5, 1, 1),
        green_scale=(1, 5, 1, -1),
        green_translate=(0, 0, 0, 1024),
        label_scale=(1, 5, 1, -1),
        label_translate=(0, 0, 0, 1024),
        )


    volume_read_params_ = dict(
        z_start_frame_number=0,
        z_end_frame_number=17,
        mod2_reverse=[False, False],
        img_width=1024,
        img_height=1024,
        frame_number_per_volume=20,
        img_dtype=np.uint16,
    )

    label_read_params_ = dict(
        neuron_pt_tuple_path = r"H:\Process_temporary\WJH\olfactory\ID\image_data\20250705\w1\synthetic_volume\all_neuron_pt_tuple.npy",
        neuron_boxes_path = None
    )
        

    viewer = trace_check_in_napari_tiff(
        red_tiff_path=red_tiff_path_,
        green_tiff_path=green_tiff_path_,
        use_visual_stack=use_visual_stack_,
        visual_volume_count=visual_volume_count_,
        volume_read_start=volume_read_start_,
        volume_read_interval=volume_read_interval_,
        volume_read_end=volume_read_end_,
        volume_read_params=volume_read_params_,
        **napari_settings,
        **label_read_params_,)
    
    napari.run()

