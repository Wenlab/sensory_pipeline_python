import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm 
import dask
import dask.array as da
from pathlib import Path

def transfer_neuron_pt_tuple_to_dict(neuron_pt_tuple):
    '''
    neuron_pt_tuple: numpyarray(volume_number,num_neurons,8)
    pt_dict: {neuron_ID: {volume_number: [x,y,z,width,height,depth]}}
    '''
    if neuron_pt_tuple.ndim == 2:
        neuron_pt_tuple = neuron_pt_tuple[np.newaxis, ...]
    neuron_points = neuron_pt_tuple.transpose(1, 0, 2)
    pt_dict = {}
    for neuron_ID in range(neuron_points.shape[0]):
        pt_dict[neuron_ID] = {}
        for volume_number in range(neuron_points.shape[1]):
            pt_dict[neuron_ID][volume_number] = neuron_points[neuron_ID][volume_number][:6]
    return pt_dict

def create_mask_for_neuron_box(pt_tuple_dict, output_shape, volume_start_number=0,  depth_correction=0, save_dir=None, chunk_shape=(1, 8, 256, 256))->da.Array:
    """
    Create a mask for neuron boxes
    Args:
        pt_tuple_dict: {neuron_ID: {volume_number: [x,y,z*5,width,height,depth*5]}}
        output_shape: (time_steps, layers, height, width)
        volume_start_number: starting index for volume to draw
        depth_correction: correction to apply to depth dimension
        save_dir: directory to save individual volume masks, if None, return a dask array
        chunk_shape: chunk shape for dask array
    Returns:
        save_dir is not None: saves individual volume masks to save_dir
        save_dir is None: returns a dask array of shape output_shape
    """
    # time_steps, num_boxes, _ = neuron_pt_tuple.shape
    time_steps, layers, height, width = output_shape

    def build_mask_np(vol_idx: int) -> np.ndarray:
        vol_key = vol_idx + volume_start_number
        mask_np = np.zeros((layers, height, width), dtype=np.int16)
        for neuron_id, volumes in pt_tuple_dict.items():
            v = volumes.get(vol_key)
            if v is None:
                continue
            x, y, z_scaled, w, h, d_scaled = v
            # handle nan values
            if np.isnan([x, y, z_scaled, w, h, d_scaled]).any():
                continue
            z_centre = z_scaled / 5.0
            depth_layers = d_scaled / 5.0
            half_depth = depth_layers / 2.0
            z_min = max(0, int(np.ceil(z_centre - half_depth)))
            z_max = min(layers, int(np.ceil(z_centre + half_depth))) + depth_correction
            x_min = max(0, int((x - w / 2.0)))
            x_max = min(width, int((x + w / 2.0)))
            y_min = max(0, int((y - h / 2.0)))
            y_max = min(height, int((y + h / 2.0)))
            if z_min <= z_max and y_min < y_max and x_min < x_max:
                mask_np[z_min:z_max, y_min:y_max, x_min:x_max] = neuron_id + 1
        return mask_np
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for vol_idx in tqdm(range(time_steps), desc="Saving neuron boxes"):
            mask_np = build_mask_np(vol_idx)
            path = Path(save_dir) / f"volume{vol_idx + volume_start_number:06d}.npy"
            np.save(path, mask_np)
        return None
    
    
    volumes = []
    for t in range(time_steps):
        delayed_vol = dask.delayed(build_mask_np)(t)
        arr = da.from_delayed(delayed_vol, shape=(layers, height, width), dtype=np.int16)
        volumes.append(arr)
        
    return da.stack(volumes).rechunk(chunk_shape)

def draw_neuron_box(neuron_pt_tuple, output_shape, volume_start_number=0, save_dir=None):
    """
    Draw neuron boxes into a mask array.
    Args:
        neuron_pt_tuple (np.ndarray): Array of shape (time_steps, num_neurons, 8) containing neuron box parameters.
        output_shape (tuple): Shape of the output mask array (time_steps, layers, height, width).
        volume_start_number (int): Starting index for volume numbering.
        save_path (str, optional): Path to save the mask array if if_save is True.
        if_save (bool): Whether to save the generated mask array.
    """
    pt_tuple_dict = transfer_neuron_pt_tuple_to_dict(neuron_pt_tuple) # first transfer neuron_pt_tuple to dict
    da_mask = create_mask_for_neuron_box(pt_tuple_dict, output_shape, volume_start_number=volume_start_number, save_dir= save_dir)

    return da_mask

if __name__ == "__main__":
    neuron_pt_tuple = np.load(r"H:\Process_temporary\WJH\olfactory\ID\image_data\20250712\w1\synthetic_volume\all_neuron_pt_tuple.npy")  # shape (time_steps, num_neurons, 8)
    output_shape = (neuron_pt_tuple.shape[0], 18, 1024, 1024)  # Example output shape
    mask = draw_neuron_box(neuron_pt_tuple, output_shape, volume_start_number=0, save_dir=r"H:\Process_temporary\WJH\olfactory\ID\image_data\20250712\w1\synthetic_volume\neuron_masks")