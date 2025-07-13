import numpy as np
import napari
import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.read_vols_using_dask import show_volumes_in_napari
from id_identify.box_region_npy import box_region_npy

#%%
def trace_check_in_napari(
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
        neuron_boxes = np.load(neuron_boxes_path)
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
        mask = box_region_npy(neuron_pt_tuple, output_shape=output_shape, if_save=True, save_path= neuron_pt_tuple_path.replace('.npy', '_mask.npy'))
        viewer.add_labels(
            mask,
            name="mask",
            blending='additive',
            scale=label_scale,
            translate=label_translate,
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
        red_contrast_limits=(150, 400),
        green_contrast_limits=(150, 400),
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
        

    viewer = trace_check_in_napari(
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

