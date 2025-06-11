if __name__ == "__main__":
    h5_file_path = r"H:\Process_temporary\WJH\olfactory\ID\result\20250604\20250604.h5"

#%% 
import h5py
import numpy as np

def load_h5_file(file_path, key):
    with h5py.File(file_path, "r") as f:
        intensity = f[key]['intensity'][:]
    return intensity

def interpolation_and_replacement(intensity, target_region, replace_region):
    """
    Intensity: np.ndarray(Neurons, Time)
    target_region: tuple (start, end)
    replace_region: tuple (start, end)
    这里用来处理在target_region(设定为异常区域)内的所有值的替换
    """
    start, end = target_region
    # 利用replace_region中的值的特征进行替换
    rep_start, rep_end = replace_region
    # 取replace_region的所有值
    replace_values = intensity[:, rep_start:rep_end]
    # 计算每个神经元在replace_region的均值和标准差
    mean_values = np.mean(replace_values, axis=1, keepdims=True)
    std_values = np.std(replace_values, axis=1, keepdims=True)
    # 生成与target_region长度相同的随机噪声，保持均值和方差一致
    region_length = end - start
    random_noise = np.random.randn(intensity.shape[0], region_length)
    replacement = mean_values + std_values * random_noise
    # 替换target_region内的值
    intensity[:, start:end] = replacement
    return intensity

def replace_abnormal_values(h5_file_path, key, target_region, replace_region):
    intensity = load_h5_file(h5_file_path, key)
    intensity = interpolation_and_replacement(intensity, target_region, replace_region)
    # 替换h5中的异常区域
    with h5py.File(h5_file_path, "a") as f:
        f[key]['intensity'][:] = intensity
    
    return None

if __name__ == "__main__":
    target_region = (825, 872)
    replace_region = (800, 825)
    replace_abnormal_values(h5_file_path, "w4", target_region, replace_region)