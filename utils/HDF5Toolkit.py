import h5py
import torch
import numpy as np

def save_to_hdf5(group, name, data):
    """
    根据数据类型将数据保存到HDF5文件中，并记录其原始数据类型。
    :param group: h5py的group对象。
    :param name: 数据的名称。
    :param data: 要保存的数据。
    """
    if type(data) == type(None) :
        data_type = "None"
    else:
        data_type = str(type(data)) 

    if isinstance(data, torch.Tensor):
        if data.dtype == torch.float16:
            data = data.to(torch.float32)
        dset = group.create_dataset(name, data=data.cpu().numpy())
    elif isinstance(data, np.ndarray):
        dset = group.create_dataset(name, data=data)
    elif isinstance(data, (str, int, float, complex)):
        dset = group.create_dataset(name, data=data)
    elif data is None:
        dset = group.create_dataset(name, shape=(0,))
    elif isinstance(data, dict):
        dset = group.create_group(name)
        for key, value in data.items():
            save_to_hdf5(dset, str(key), value)
    elif isinstance(data, (list, tuple)):
        try:
            np_array_data = np.array(data)
            dset = group.create_dataset(name, data=np_array_data)
        except (ValueError, TypeError):
            dset = group.create_group(name)
            for idx, item in enumerate(data):
                save_to_hdf5(dset, str(idx), item)
    else:
        raise TypeError("Unsupported data type")

    # 为数据集添加原始数据类型的属性
    dset.attrs['data_type'] = data_type

def save_h5file(path, root_name, mode = "a",**kwargs):
    with h5py.File(path, mode) as file:
        for key, value in kwargs.items():
            save_to_hdf5(file, f"{root_name}/{key}", value)