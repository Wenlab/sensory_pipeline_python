import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def calculate_delta_F_over_F0(intensity, intervals, baseline_pre=10, baseline_post=0, background_noise=102):
    """
    使用改进的拟合方法计算deltaF/F0
    
    Parameters:
    - intensity: DataFrame, 强度值数据
    - intervals: list, 刺激区间列表
    - baseline_pre, baseline_post: 基线区间参数
    
    Returns:
    - delta_F_over_F0: DataFrame, deltaF/F0值
    - fitted_F0_df: DataFrame, 拟合的F0曲线
    - quality_info: dict, 拟合质量信息
    """
    # 使用改进的拟合函数
    fitted_F0_df, fitted_params, r2_scores, model_types = fitted_F_base(
        intensity, intervals, baseline_pre, baseline_post
    )
    
    # 计算delta_F_over_F0
    delta_F_over_F0 = pd.DataFrame(index=intensity.index, columns=intensity.columns)
    
    for roi_idx in range(intensity.shape[0]):
        # 获取原始信号和拟合的基线
        raw_signal = intensity.iloc[roi_idx].values
        baseline = fitted_F0_df.iloc[roi_idx].values
        
        # 检查基线是否有非常小的值（可能导致除法问题）
        min_baseline = np.percentile(baseline, 5)  # 使用第5百分位数作为最小阈值
        safe_baseline = np.maximum(baseline, min_baseline * 0.1)  # 确保基线不会太接近0
        
        # 计算deltaF/F0
        delta_f = (raw_signal - safe_baseline)
        delta_f_f0 = delta_f / (safe_baseline - background_noise)
        delta_F_over_F0.iloc[roi_idx] = delta_f_f0
        delta_F_over_F0.iloc[roi_idx] = delta_f_f0.astype(np.float32)
    
    delta_F_over_F0 = delta_F_over_F0.astype(np.float32)
    # 收集质量信息
    quality_info = {
        'r2_scores': r2_scores,
        'model_types': model_types,
        'mean_r2': np.mean(r2_scores),
        'median_r2': np.median(r2_scores),
        'fit_params': fitted_params
    }
    
    return delta_F_over_F0, fitted_F0_df, quality_info

def fitted_F_base(intensity, intervals, baseline_pre, baseline_post):
    '''改进版本的fitted_F_base函数'''
    # baseline_intervals = [(start - baseline_pre, start - baseline_post) for start, _ in intervals]
    # improved
    baseline_intervals = []
    if intervals:
        first_start = intervals[0][0]
        baseline_intervals.append((max(0, first_start - 100), max(0, first_start - 10)))
        baseline_intervals.extend([(start - baseline_pre, start - baseline_post) for start, _ in intervals[1:]])
    else:
        baseline_intervals = []
        
    # 确保基线区间不超出数据的时间范围
    baseline_intervals = [(max(0, start), max(0, end)) for start, end in baseline_intervals]

    baseline_timepoints = []
    baseline_values = []

    time_axis = intensity.columns.astype(int)

    for roi_idx in range(intensity.shape[0]):
        roi_baseline_time = []
        roi_baseline_value = []
        for start, end in baseline_intervals:
            start = max(0, start)
            end = min(intensity.shape[1], end)

            time_points_in_baseline = np.arange(start, end)
            baseline_data = intensity.iloc[roi_idx, time_points_in_baseline]

            roi_baseline_time.extend(time_points_in_baseline)
            roi_baseline_value.extend(baseline_data)
        baseline_timepoints.append(np.array(roi_baseline_time))
        baseline_values.append(np.array(roi_baseline_value))
    
    fitted_params = []
    fitted_F0_curves = []
    r2_scores = []
    model_types = []  # 记录每个ROI使用的模型类型
    
    for roi_idx in range(intensity.shape[0]):
        x_data = baseline_timepoints[roi_idx]
        y_data = baseline_values[roi_idx]
        
        if len(x_data) == 0 or len(y_data) == 0:
            print(f"ROI {roi_idx} has no baseline data, using mean value.")
            fitted_F0 = np.ones(len(time_axis)) * np.mean(intensity.iloc[roi_idx])
            fitted_F0_curves.append(fitted_F0)
            fitted_params.append([0, 0, np.mean(intensity.iloc[roi_idx])])
            r2_scores.append(0)
            model_types.append("mean")
            continue
        
        # 排序确保时间序列有序
        sorted_indices = np.argsort(x_data)
        x_data = x_data[sorted_indices]
        y_data = y_data[sorted_indices]
        
        # 去除异常值（可选）
        y_mean = np.mean(y_data)
        y_std = np.std(y_data)
        valid_idx = np.where(np.abs(y_data - y_mean) < 3 * y_std)[0]  # 3-sigma规则
        x_data = x_data[valid_idx]
        y_data = y_data[valid_idx]
        
        # 自动选择最佳拟合模型
        best_func, best_params, r2 = fit_baseline(x_data, y_data, model_type='auto')
        
        # 确定模型类型
        if best_func.__name__ == 'exponential_decay':
            model_type = "exponential"
            # 转换为原始格式以兼容
            popt = best_params
        elif best_func.__name__ == 'polynomial':
            model_type = "polynomial"
            # 为了兼容原有代码，我们将多项式模型参数转换为指数衰减的形式
            # 这里只是一个近似处理
            popt = [0, 0, np.mean(y_data)]  # 简单近似
        elif best_func.__name__ == 'double_exponential':
            model_type = "double_exponential"
            # 简化处理
            a1, b1, a2, b2, c = best_params
            popt = [a1 + a2, (b1 + b2)/2, c]  # 近似处理
        else:
            model_type = "moving_average"
            popt = [0, 0, np.mean(y_data)]
        
        # 使用选定的模型生成整个时间轴上的F0曲线
        fitted_F0 = np.array([best_func(t, *best_params) for t in time_axis])
        
        fitted_F0_curves.append(fitted_F0)
        fitted_params.append(popt)  # 保持兼容性
        r2_scores.append(r2)
        model_types.append(model_type)
    
    # 将拟合的F0曲线转换为DataFrame
    fitted_F0_df = pd.DataFrame(fitted_F0_curves, columns=time_axis)
    
    return fitted_F0_df, fitted_params, r2_scores, model_types

# 拟合基线的函数
def fit_baseline(x_data, y_data, model_type='exp'):
    """
    根据数据特性选择最佳拟合模型
    
    Parameters:
    - x_data: 时间点数组
    - y_data: 相应的强度值
    - model_type: 模型类型，可选 'exp'(指数), 'poly'(多项式), 'auto'(自动选择)
    
    Returns:
    - best_func: 最佳拟合函数
    - best_params: 最佳拟合参数
    - r2_score: 拟合优度
    """
    # 指数衰减模型
    def exponential_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    # 多项式模型（3阶）
    def polynomial(t, a, b, c, d):
        return a * t**3 + b * t**2 + c * t + d
    
    # 双指数模型（用于复杂衰减特征）
    def double_exponential(t, a1, b1, a2, b2, c):
        return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t) + c
    
    best_r2 = -np.inf
    best_func = None
    best_params = None
    
    if model_type == 'exp' or model_type == 'auto':
        try:
            initial_guess = [np.max(y_data)*1.1, 0.001, np.min(y_data)*0.9]
            popt, pcov = curve_fit(exponential_decay, x_data, y_data, p0=initial_guess, maxfev=10000)
            y_pred = exponential_decay(x_data, *popt)
            r2 = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            if r2 > best_r2:
                best_r2 = r2
                best_func = exponential_decay
                best_params = popt
        except RuntimeError:
            pass
    
    if model_type == 'poly' or model_type == 'auto':
        try:
            initial_guess = [0, 0, 0, np.mean(y_data)]
            popt, pcov = curve_fit(polynomial, x_data, y_data, p0=initial_guess, maxfev=10000)
            y_pred = polynomial(x_data, *popt)
            r2 = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            if r2 > best_r2:
                best_r2 = r2
                best_func = polynomial
                best_params = popt
        except RuntimeError:
            pass
    
    if model_type == 'auto':
        try:
            initial_guess = [np.max(y_data)*0.6, 0.01, np.max(y_data)*0.4, 0.001, np.min(y_data)*0.9]
            popt, pcov = curve_fit(double_exponential, x_data, y_data, p0=initial_guess, maxfev=10000)
            y_pred = double_exponential(x_data, *popt)
            r2 = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            if r2 > best_r2:
                best_r2 = r2
                best_func = double_exponential
                best_params = popt
        except RuntimeError:
            pass
    
    # 如果所有拟合都失败，使用移动平均作为备选
    if best_func is None:
        def moving_average(t, window_size):
            # 实现移动平均函数
            return np.convolve(y_data, np.ones(window_size)/window_size, mode='same')
        
        window_size = max(3, len(x_data) // 10)  # 合理的窗口大小
        best_func = lambda t, w=window_size: moving_average(t, w)
        best_params = [window_size]
        best_r2 = 0  # 无法直接计算R2
    
    return best_func, best_params, best_r2

def fitting_curve_vs_original(intensity, key, intervals, fitted_F0_df, r2_scores, model_types, baseline_pre=10, baseline_post=0):
    """
    Compare the fitted curve with the original intensity curve.
    Plots are sorted by fit quality (best to worst) according to R² scores.
    
    Parameters:
    - intensity: Original data for each ROI (2D array-like, shape: [ROIs, time_points])
    - intervals: List of stimulation intervals (tuples of (start, end))
    - fitted_F0_df: DataFrame containing fitted F0 curves (each row corresponds to an ROI)
    - r2_scores: List or array of R² scores for each ROI
    - model_types: List of model types used for each ROI
    - baseline_pre, baseline_post: Baseline interval parameters (not used in plotting but can be added)
    """

    
    # Sort indices by R^2 score in descending order (best fitting first)
    sorted_indices = np.argsort(-np.array(r2_scores))
    
    # Create a unique PDF filename using timestamp to avoid overwriting
    pdf_filename = f'fitting_curve_vs_original_{key}.pdf'
    pdf = PdfPages(pdf_filename)
    
    # Set number of plots per page
    plots_per_page = 3
    fig_rows = 3
    fig_cols = 1
    
    # Loop through sorted indices and plot in batches
    total_rois = len(sorted_indices)
    for i in range(0, total_rois, plots_per_page):
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(12, 10))
        axes = axes.flatten()
        
        for j in range(plots_per_page):
            if i + j >= total_rois:
                axes[j].axis('off')
                continue
            
            roi_idx = sorted_indices[i + j]
            ax = axes[j]
            
            # Plot original intensity curve(intensity is a dataframe)
            ax.plot(intensity.iloc[roi_idx], label='Original', color='blue', alpha=0.7)
            
            # Plot fitted curve (red)
            fitted_curve = fitted_F0_df.iloc[roi_idx].values
            ax.plot(fitted_curve, label='Fitted', color='red', alpha=0.7)
            
            # Draw vertical spans for stimulus intervals
            for start, end in intervals:
                ax.axvspan(start, end, color='yellow', alpha=0.2)
            
            # Set title and labels
            title = f"ROI {roi_idx}: R² = {r2_scores[roi_idx]:.3f}, Model: {model_types[roi_idx]}"
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Intensity')
            ax.legend(fontsize='small', loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    pdf.close()
    print(f"Fitting comparison visualization saved to {pdf_filename}")

def visualize_fitting_quality(intensity, key, intervals, fitted_F0_df, r2_scores, model_types, baseline_pre=10, baseline_post=0):
    """
    Visualize fitting quality and help identify problematic neurons
    
    Parameters:

    - intensity: Original data for each neuron
    - intervals: Time intervals of interest
    - fitted_F0_df: DataFrame containing the fitted F0 curves
    - r2_scores: R^2 scores for each ROI
    - model_types: Model types used for each ROI
    - baseline_pre, baseline_post: Baseline interval parameters
    - key: Key value for the selected worm
    """
    
    # Calculate baseline intervals
    baseline_intervals = [(start - baseline_pre, start - baseline_post) for start, _ in intervals]
    baseline_intervals = [(max(0, start), max(0, end)) for start, end in baseline_intervals]
    
    # Get baseline points for each neuron
    roi_baseline_points = []
    for roi_idx in range(intensity.shape[0]):
        time_points = []
        value_points = []
        for start, end in baseline_intervals:
            start = max(0, start)
            end = min(intensity.shape[1], end)
            time_points.extend(np.arange(start, end))
            value_points.extend(intensity.iloc[roi_idx, start:end])
        roi_baseline_points.append((np.array(time_points), np.array(value_points)))
    
    # Create fitting quality assessment report
    with PdfPages(f'fitting_quality_report_{key}.pdf') as pdf:
        # 1. Overall fitting quality distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(r2_scores, bins=20, alpha=0.7)
        plt.axvline(0.8, color='r', linestyle='--', label='R^2=0.8 (good fit threshold)')
        plt.xlabel('R^2 Score')
        plt.ylabel('Number of Neurons')
        plt.title('Fitting Quality Distribution')
        plt.legend()
        pdf.savefig()
        plt.close()
        
        # 2. Model type statistics
        plt.figure(figsize=(10, 6))
        model_counts = pd.Series(model_types).value_counts()
        plt.bar(model_counts.index, model_counts.values)
        plt.xlabel('Model Type')
        plt.ylabel('Number of Neurons')
        plt.title('Model Type Usage Statistics')
        plt.xticks(rotation=45)
        pdf.savefig()
        plt.close()
        
        # 3. Show the best and worst fits by R^2
        sorted_indices = np.argsort(r2_scores)
        worst_indices = sorted_indices[:min(5, len(sorted_indices))]
        best_indices = sorted_indices[-min(5, len(sorted_indices)):]
        
        # Visualize the neurons with the worst fits
        plt.figure(figsize=(15, 10))
        plt.suptitle('Neurons with the Worst Fits', fontsize=16)
        
        for i, idx in enumerate(worst_indices):
            plt.subplot(len(worst_indices), 1, i+1)
            time_points, values = roi_baseline_points[idx]
            plt.scatter(time_points, values, color='blue', alpha=0.5, label='Baseline Points')
            
            # Show fitted curve
            full_time = intensity.columns.astype(int)
            plt.plot(full_time, fitted_F0_df.iloc[idx], 'r-', label=f'Fitted Curve (R^2={r2_scores[idx]:.3f}, Model: {model_types[idx]})')
            
            plt.title(f'Neuron #{idx}')
            plt.xlabel('Time')
            plt.ylabel('Intensity')
            plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig()
        plt.close()
        
        # Visualize the neurons with the best fits
        plt.figure(figsize=(15, 10))
        plt.suptitle('Neurons with the Best Fits', fontsize=16)
        
        for i, idx in enumerate(best_indices):
            plt.subplot(len(best_indices), 1, i+1)
            time_points, values = roi_baseline_points[idx]
            plt.scatter(time_points, values, color='blue', alpha=0.5, label='Baseline Points')
            
            # Show fitted curve
            full_time = intensity.columns.astype(int)
            plt.plot(full_time, fitted_F0_df.iloc[idx], 'r-', label=f'Fitted Curve (R^2={r2_scores[idx]:.3f}, Model: {model_types[idx]})')
            
            plt.title(f'Neuron #{idx}')
            plt.xlabel('Time')
            plt.ylabel('Intensity')
            plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig()
        plt.close()
        
        # 4. Detailed inspection of each neuron's fitting
        for roi_idx in range(intensity.shape[0]):
            # Only show detailed plots for neurons with R^2 below the threshold
            if r2_scores[roi_idx] > 0.8:
                continue
                
            plt.figure(figsize=(12, 6))
            time_points, values = roi_baseline_points[roi_idx]
            plt.scatter(time_points, values, color='blue', alpha=0.5, label='Baseline Points')
            
            # Show fitted curve
            full_time = intensity.columns.astype(int)
            plt.plot(full_time, fitted_F0_df.iloc[roi_idx], 'r-', label=f'Fitted Curve')
            
            # Add stimulus interval markers
            for start, end in intervals:
                plt.axvspan(start, end, color='gray', alpha=0.2)
            
            plt.title(f'Neuron #{roi_idx} (R^2={r2_scores[roi_idx]:.3f}, Model: {model_types[roi_idx]})')
            plt.xlabel('Time')
            plt.ylabel('Intensity')
            plt.legend()
            
            pdf.savefig()
            plt.close()
    
    print(f"Fitting quality report generated: fitting_quality_report_{key}.pdf")
    
    # Return indices of poorly fitted neurons for further analysis
    poor_fit_indices = np.where(np.array(r2_scores) < 0.8)[0]
    return poor_fit_indices