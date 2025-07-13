import h5py
import numpy as np
import os
import sys
if __name__ == "__main__":
    sys.path.append("../")
    sys.path.append("./python/")
    sys.path.append("./")  
from utils.HDF5Toolkit import save_h5file
import dask.array as da
from tqdm import tqdm 
import cv2
import random

def transfer_neuron_pt_tuple_to_dict(neuron_pt_tuple):
    '''
    neuron_pt_tuple: numpyarray(volume_number,num_neurons,8)
    pt_dict: {neuron_ID: {volume_number: [x,y,z,width,height,depth]}}
    '''
    neuron_points = neuron_pt_tuple.transpose(1, 0, 2)
    pt_dict = {}
    for neuron_ID in range(neuron_points.shape[0]):
        pt_dict[neuron_ID] = {}
        for volume_number in range(neuron_points.shape[1]):
            pt_dict[neuron_ID][volume_number] = neuron_points[neuron_ID][volume_number][:6]
    return pt_dict

def create_mask_for_bounding_boxes_gray(pt_tuple_dict, output_shape, volume_start_number=0):
    """
    在指定形状的稀疏矩阵中绘制目标框的灰度蒙版层，框外像素为NaN
    
    参数:
    pt_tuple_dict: {neuron_ID: {volume_number: [x,y,z*5,width,height,depth*5]}}
    output_shape: 输出蒙版的形状 (time_steps, layers, height, width)
    
    返回:
    masks: 形状为output_shape的灰度蒙版数组，每个框有不同的灰度值，框外为NaN
    """
    # time_steps, num_boxes, _ = neuron_pt_tuple.shape
    output_time_steps, output_layers, height, width = output_shape
    
    # 创建输出掩码Dask数组，初始值为0
    # masks = np.full(output_shape, 0, dtype=np.int16)
    masks = np.zeros(output_shape, dtype=np.int16)

    # num_boxes = len(pt_tuple_dict.keys())
    
    # 为每个神经元框生成与其ID对应的灰度值(ID+1)
    
    # gray_values = np.array([i + 1 for i in range(num_boxes)])
    
    # 遍历每个时间步长
    for id, id_values in tqdm(pt_tuple_dict.items(), desc="Processing neuron boxes"):
        # 遍历每个框
        for volume_num, pt_tuple in id_values.items():
            
            # 提取框参数
            x, y, z_reshaped, width_box, height_box, depth_reshaped = pt_tuple
            
            z = z_reshaped // 5  # 除以5以匹配输出层的深度
            depth = depth_reshaped // 5
            # 计算框的边界
            x1 = int(max(0, x - width_box / 2))
            y1 = int(max(0, y - height_box / 2))
            x2 = int(min(width - 1, x + width_box / 2))
            y2 = int(min(height - 1, y + height_box / 2))
            
            # 确定框所在的层范围
            z_min = int(max(0, z - depth / 2))
            z_max = int(min(output_layers - 1, z + depth / 2 + 1))
            
            # 在对应的层上绘制矩形
            for layer in range(z_min, z_max):
                # 创建临时掩码
                layer_mask = np.full((height, width), -1, dtype=np.int16)
                
                # 绘制填充矩形
                cv2.rectangle(layer_mask, (x1, y1), (x2, y2), int(id+1), -1)
                
                # 将此层的掩码应用到结果上
                # 只在非零区域更新值（即矩形内部）
                mask_indices = np.where(layer_mask > 0)
                masks[volume_num-volume_start_number, layer][mask_indices] = layer_mask[mask_indices]
    return masks

def box_region_npy(neuron_pt_tuple, output_shape, volume_start_number=0, save_path=None, if_save=False):
    """
    生成目标框的灰度蒙版层，框外像素为NaN
    
    参数:
    neuron_pt_tuple: numpyarray(volume_number,num_neurons,8)
    output_shape: 输出蒙版的形状 (time_steps, layers, height, width)
    
    返回:
    masks: 形状为output_shape的灰度蒙版数组，每个框有不同的灰度值，框外为NaN
    """
    pt_tuple_dict = transfer_neuron_pt_tuple_to_dict(neuron_pt_tuple)
    masks = create_mask_for_bounding_boxes_gray(pt_tuple_dict, output_shape, volume_start_number=volume_start_number)

    if if_save:
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "box_region.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, masks)
        
    return masks