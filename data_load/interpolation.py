#%% 
import h5py
import numpy as np
from scipy import interpolate
from scipy.ndimage import uniform_filter1d

def load_h5_file(file_path, key):
    with h5py.File(file_path, "r") as f:
        intensity = f[key]['intensity'][:]
    return intensity

def interpolation_and_replacement(intensity, target_region, replace_region=None, 
                                interpolation_method='cubic_spline', interpolation_window=20):
    """
    Intensity: np.ndarray(Neurons, Time)
    target_region: tuple (start, end) - 异常区域
    replace_region: tuple (start, end) or None - 参考区域
    interpolation_method: str - 插值方法 ('cubic_spline', 'polynomial', 'moving_average', 'linear')
    interpolation_window: int - 插值窗口大小
    
    如果提供replace_region，则用该区域的统计特征进行替换
    如果replace_region为None，则使用指定的插值方法
    """
    start, end = target_region
    
    if replace_region is not None:
        # 使用replace_region的统计特征进行替换
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
    else:
        # 使用指定的插值方法
        if interpolation_method == 'cubic_spline':
            intensity = _cubic_spline_interpolation(intensity, target_region, interpolation_window)
        elif interpolation_method == 'polynomial':
            intensity = _polynomial_interpolation(intensity, target_region, interpolation_window)
        elif interpolation_method == 'moving_average':
            intensity = _moving_average_interpolation(intensity, target_region, interpolation_window)
        elif interpolation_method == 'linear':
            intensity = _linear_interpolation(intensity, target_region, interpolation_window)
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")
    
    return intensity

def _cubic_spline_interpolation(intensity, target_region, window):
    """三次样条插值 - 最平滑的方法"""
    start, end = target_region
    
    for neuron_idx in range(intensity.shape[0]):
        # 获取插值区域外的数据点
        left_start = max(0, start - window)
        right_end = min(intensity.shape[1], end + window)
        
        # 构建插值用的数据点（排除target_region）
        x_points = []
        y_points = []
        
        # 左侧数据点
        if left_start < start:
            x_left = np.arange(left_start, start)
            y_left = intensity[neuron_idx, left_start:start]
            # 过滤掉NaN值
            valid_mask = ~np.isnan(y_left)
            x_points.extend(x_left[valid_mask])
            y_points.extend(y_left[valid_mask])
        
        # 右侧数据点
        if end < right_end:
            x_right = np.arange(end, right_end)
            y_right = intensity[neuron_idx, end:right_end]
            # 过滤掉NaN值
            valid_mask = ~np.isnan(y_right)
            x_points.extend(x_right[valid_mask])
            y_points.extend(y_right[valid_mask])
        
        if len(x_points) >= 4:  # 三次样条至少需要4个点
            # 使用三次样条插值
            try:
                cs = interpolate.CubicSpline(x_points, y_points, bc_type='natural')
                x_interp = np.arange(start, end)
                intensity[neuron_idx, start:end] = cs(x_interp)
            except:
                # 如果样条插值失败，回退到线性插值
                intensity = _linear_interpolation(intensity, target_region, window)
        else:
            # 数据点太少，使用线性插值
            intensity = _linear_interpolation(intensity, target_region, window)
    
    return intensity

def _polynomial_interpolation(intensity, target_region, window):
    """多项式插值 - 基于更多数据点的趋势"""
    start, end = target_region
    
    for neuron_idx in range(intensity.shape[0]):
        # 获取更大的窗口用于多项式拟合
        left_start = max(0, start - window)
        right_end = min(intensity.shape[1], end + window)
        
        # 构建拟合用的数据点（排除target_region）
        x_points = []
        y_points = []
        
        # 左侧数据点
        if left_start < start:
            x_left = np.arange(left_start, start)
            y_left = intensity[neuron_idx, left_start:start]
            valid_mask = ~np.isnan(y_left)
            x_points.extend(x_left[valid_mask])
            y_points.extend(y_left[valid_mask])
        
        # 右侧数据点
        if end < right_end:
            x_right = np.arange(end, right_end)
            y_right = intensity[neuron_idx, end:right_end]
            valid_mask = ~np.isnan(y_right)
            x_points.extend(x_right[valid_mask])
            y_points.extend(y_right[valid_mask])
        
        if len(x_points) >= 3:
            try:
                # 使用2次多项式拟合（避免过拟合）
                degree = min(2, len(x_points) - 1)
                coeffs = np.polyfit(x_points, y_points, degree)
                x_interp = np.arange(start, end)
                intensity[neuron_idx, start:end] = np.polyval(coeffs, x_interp)
            except:
                # 如果多项式拟合失败，回退到线性插值
                intensity = _linear_interpolation(intensity, target_region, window)
        else:
            # 数据点太少，使用线性插值
            intensity = _linear_interpolation(intensity, target_region, window)
    
    return intensity

def _moving_average_interpolation(intensity, target_region, window):
    """移动平均插值 - 基于局部平均值的平滑方法"""
    start, end = target_region
    
    for neuron_idx in range(intensity.shape[0]):
        # 获取左右两侧的数据
        left_start = max(0, start - window)
        right_end = min(intensity.shape[1], end + window)
        
        # 计算左侧平均值
        if left_start < start:
            left_data = intensity[neuron_idx, left_start:start]
            left_data = left_data[~np.isnan(left_data)]
            left_avg = np.mean(left_data) if len(left_data) > 0 else 0
        else:
            left_avg = intensity[neuron_idx, start] if start > 0 else 0
        
        # 计算右侧平均值
        if end < right_end:
            right_data = intensity[neuron_idx, end:right_end]
            right_data = right_data[~np.isnan(right_data)]
            right_avg = np.mean(right_data) if len(right_data) > 0 else 0
        else:
            right_avg = intensity[neuron_idx, end-1] if end < intensity.shape[1] else 0
        
        # 使用加权平均进行插值
        region_length = end - start
        for i, pos in enumerate(range(start, end)):
            # 根据位置计算权重
            weight_right = i / (region_length - 1) if region_length > 1 else 0.5
            weight_left = 1 - weight_right
            
            # 加权平均
            interpolated_value = weight_left * left_avg + weight_right * right_avg
            
            # 添加一些平滑的随机噪声以保持信号的自然变化
            noise_scale = 0.1 * abs(right_avg - left_avg)
            noise = np.random.normal(0, noise_scale)
            
            intensity[neuron_idx, pos] = interpolated_value + noise
    
    return intensity

def _linear_interpolation(intensity, target_region, window):
    """线性插值 - 简单但有效的方法"""
    start, end = target_region
    
    for neuron_idx in range(intensity.shape[0]):
        # 确定插值的边界点
        left_start = max(0, start - window)
        left_end = start
        right_start = end
        right_end = min(intensity.shape[1], end + window)
        
        # 获取左边和右边区域的平均值作为插值端点
        if left_end > left_start:
            left_data = intensity[neuron_idx, left_start:left_end]
            left_data = left_data[~np.isnan(left_data)]
            left_value = np.mean(left_data) if len(left_data) > 0 else 0
        else:
            left_value = intensity[neuron_idx, start] if start > 0 else intensity[neuron_idx, 0]
            
        if right_end > right_start:
            right_data = intensity[neuron_idx, right_start:right_end]
            right_data = right_data[~np.isnan(right_data)]
            right_value = np.mean(right_data) if len(right_data) > 0 else 0
        else:
            right_value = intensity[neuron_idx, end-1] if end < intensity.shape[1] else intensity[neuron_idx, -1]
        
        # 线性插值
        region_length = end - start
        interpolated_values = np.linspace(left_value, right_value, region_length)
        intensity[neuron_idx, start:end] = interpolated_values
    
    return intensity

def replace_abnormal_values(h5_file_path, key, target_region, replace_region=None, 
                          interpolation_method='cubic_spline', interpolation_window=20):
    """
    替换HDF5文件中的异常区域
    
    Args:
        h5_file_path: HDF5文件路径
        key: 数据键名
        target_region: tuple (start, end) - 要替换的异常区域
        replace_region: tuple (start, end) or None - 参考区域，如果为None则使用插值方法
        interpolation_method: str - 插值方法 ('cubic_spline', 'polynomial', 'moving_average', 'linear')
        interpolation_window: int - 插值窗口大小
    """
    intensity = load_h5_file(h5_file_path, key)
    intensity = interpolation_and_replacement(intensity, target_region, replace_region, 
                                            interpolation_method, interpolation_window)
    # 替换h5中的异常区域
    with h5py.File(h5_file_path, "a") as f:
        f[key]['intensity'][:] = intensity
    
    return None

if __name__ == "__main__":
    h5_file_path = r"H:\Process_temporary\WJH\olfactory\ID\result\20250706\20250706.h5"
    target_region = (133, 238)
    
    # 方式1: 使用replace_region的统计特征进行替换
    # replace_region = (800, 825)
    # replace_abnormal_values(h5_file_path, "w4", target_region, replace_region)
    
    # 方式2: 使用三次样条插值（推荐）
    # replace_abnormal_values(h5_file_path, "w5", target_region, 
    #                       interpolation_method='cubic_spline', interpolation_window=30)
    
    # 方式3: 使用多项式插值
    # replace_abnormal_values(h5_file_path, "w5", target_region, 
    #                       interpolation_method='polynomial', interpolation_window=25)
    
    # 方式4: 使用移动平均插值
    # replace_abnormal_values(h5_file_path, "w5", target_region, 
    #                       interpolation_method='moving_average', interpolation_window=20)
    
    # 方式5: 使用线性插值
    replace_abnormal_values(h5_file_path, "w5", target_region, 
                          interpolation_method='linear', interpolation_window=15)