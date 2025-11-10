import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
import warnings

def split_dataframe(df, n_seg):
    """Split a dataframe into column-wise segments using a sequence definition.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to split.
    n_seg : array-like or None
        Column counts for each segment. Values that cannot be converted to
        positive integers are ignored.

    Returns
    -------
    list[pandas.DataFrame]
        A list of dataframe copies, each containing a consecutive slice of the
        original columns. When ``n_seg`` is empty or ``None`` the list contains
        a single copy of ``df``. Any remaining columns after consuming
        ``n_seg`` are returned as an additional segment so that all columns are
        preserved.
    """

    if df is None or df.empty:
        return []

    total_cols = df.shape[1]

    segment_lengths = []
    if n_seg is not None:
        try:
            flat_values = np.atleast_1d(n_seg).tolist()
        except Exception:
            flat_values = list(n_seg) if isinstance(n_seg, (list, tuple)) else [n_seg]
        for value in flat_values:
            try:
                length = int(value)
            except (TypeError, ValueError):
                continue
            if length > 0:
                segment_lengths.append(length)

    split_dfs = []
    current_col_index = 0

    for num in segment_lengths:
        if current_col_index >= total_cols:
            print("Warning: n_seg exceeds the number of columns in the dataframe.")
            break
        end_col_index = current_col_index + num
        if end_col_index > total_cols:
            print(f"Warning: end_col_index {end_col_index} exceeds total columns {total_cols}. Adjusting to fit.")
            end_col_index = total_cols

        split_df = df.iloc[:, current_col_index:end_col_index].copy()
        if not split_df.empty:
            split_dfs.append(split_df)
        current_col_index = end_col_index

    if not split_dfs:
        split_dfs.append(df.copy())

    return split_dfs


def _infer_segment_offset(columns):
    """Best-effort conversion of the first column label to an integer offset."""
    if columns is None or len(columns) == 0:
        return 0
    first_label = columns[0]
    try:
        return int(first_label)
    except (TypeError, ValueError):
        try:
            return int(float(first_label))
        except (TypeError, ValueError):
            return 0


def _prepare_segment_intervals(intervals, segments):
    """Map global stimulus intervals to each segmented dataframe.

    The helper supports two interval formats:
    - flat list of ``(start, end)`` tuples that span the full recording
    - list of lists where each inner list already corresponds to a segment
    """

    if not segments:
        return []

    if not intervals:
        return [[] for _ in segments]

    first_entry = intervals[0]
    is_nested = (
        isinstance(first_entry, (list, tuple))
        and bool(first_entry)
        and isinstance(first_entry[0], (list, tuple))
    )
    if is_nested:
        if len(intervals) != len(segments):
            raise ValueError("Number of interval sets does not match the number of segments.")
        return [
            [(int(start), int(end)) for start, end in segment_intervals]
            for segment_intervals in intervals
        ]

    try:
        flat_intervals = [(int(start), int(end)) for start, end in intervals]
    except (TypeError, ValueError):
        flat_intervals = []

    segment_intervals = []
    for segment_df in segments:
        columns = segment_df.columns
        offset = _infer_segment_offset(columns)
        seg_length = segment_df.shape[1]
        seg_end = offset + seg_length
        seg_list = []
        for start, end in flat_intervals:
            if end <= offset or start >= seg_end:
                continue
            seg_start = max(start, offset) - offset
            seg_end_adj = min(end, seg_end) - offset
            if seg_end_adj > seg_start:
                seg_list.append((seg_start, seg_end_adj))
        segment_intervals.append(seg_list)

    return segment_intervals


def _clean_outliers(values, index, threshold=10):
    """Replace large-magnitude spikes with interpolated values to stabilise dF/F."""
    series = pd.Series(values, index=index, dtype=np.float64)
    mask = np.abs(series.values) > threshold
    if mask.any():
        series = series.mask(mask)
        series = series.interpolate(method='linear', limit_direction='both')
        series = series.ffill().bfill()
    return series.astype(np.float32)


def _compute_segment_delta(intensity_segment, fitted_segment, background_noise, outlier_threshold=5):
    """Compute delta F/F0 for a single segment."""
    delta_df = pd.DataFrame(
        np.nan,
        index=intensity_segment.index,
        columns=intensity_segment.columns,
        dtype=np.float32,
    )

    for roi_idx in range(intensity_segment.shape[0]):
        raw_signal = intensity_segment.iloc[roi_idx].to_numpy(dtype=np.float64)
        baseline = fitted_segment.iloc[roi_idx].to_numpy(dtype=np.float64)

        if baseline.size == 0:
            continue

        min_baseline = np.percentile(baseline, 5)
        safe_baseline = np.maximum(baseline, min_baseline * 0.1)

        denominator = safe_baseline - background_noise
        denominator = np.where(
            np.abs(denominator) < 1e-6,
            np.where(denominator < 0, -1e-6, 1e-6),
            denominator,
        )

        delta_f = raw_signal - safe_baseline
        delta_f_f0 = delta_f / denominator

        cleaned_series = _clean_outliers(delta_f_f0, intensity_segment.columns, threshold=outlier_threshold)
        delta_df.iloc[roi_idx] = cleaned_series.to_numpy(dtype=np.float32)

    return delta_df
        
        
def calculate_delta_F_over_F0(
    intensity,
    intervals,
    baseline_pre=6,
    baseline_post=1,
    vps_setting=1,
    background_noise=102,
    n_seq=None,
):
    """Compute delta F/F0 with optional segmentation support.

    When ``n_seq`` is provided the intensity dataframe is split column-wise and
    curve fitting is executed independently for each segment. The outputs are
    then merged back to match the original dataframe layout. Fitting quality
    metrics are tracked per segment and aggregated globally.
    """

    if intensity is None or intensity.empty:
        empty_df = pd.DataFrame()
        empty_quality = {
            'r2_scores': [],
            'model_types': [],
            'mean_r2': np.nan,
            'median_r2': np.nan,
            'fit_params': [],
            'segments': {},
        }
        return empty_df, empty_df, empty_quality

    split_intensity_df_list = split_dataframe(intensity, n_seq)
    if not split_intensity_df_list:
        split_intensity_df_list = [intensity.copy()]

    segment_intervals = _prepare_segment_intervals(intervals, split_intensity_df_list)

    combined_delta = pd.DataFrame(
        np.nan,
        index=intensity.index,
        columns=intensity.columns,
        dtype=np.float32,
    )
    combined_fitted = pd.DataFrame(
        np.nan,
        index=intensity.index,
        columns=intensity.columns,
        dtype=np.float32,
    )

    all_r2_scores = []
    all_model_types = []
    all_fit_params = []
    segment_quality = {}

    for seg_idx, segment_df in enumerate(split_intensity_df_list):
        original_columns = segment_df.columns
        working_segment = segment_df.copy()
        working_segment.columns = pd.RangeIndex(working_segment.shape[1])

        seg_intervals = segment_intervals[seg_idx] if seg_idx < len(segment_intervals) else []

        fitted_F0_df, fitted_params, r2_scores, model_types = fitted_F_base(
            working_segment,
            seg_intervals,
            baseline_pre,
            baseline_post,
            vps_setting=vps_setting,
        )
        fitted_F0_df = fitted_F0_df.astype(np.float32, copy=False)
        delta_segment = _compute_segment_delta(
            working_segment,
            fitted_F0_df,
            background_noise=background_noise,
        )

        fitted_F0_df.columns = original_columns
        delta_segment.columns = original_columns

        combined_delta.loc[:, original_columns] = delta_segment
        combined_fitted.loc[:, original_columns] = fitted_F0_df

        seg_r2_list = [float(r) for r in list(r2_scores)] if len(r2_scores) else []
        seg_model_types = list(model_types)
        seg_fit_params = list(fitted_params)

        segment_quality[seg_idx] = {
            'r2_scores': seg_r2_list,
            'model_types': seg_model_types,
            'mean_r2': float(np.nanmean(seg_r2_list)) if seg_r2_list else np.nan,
            'median_r2': float(np.nanmedian(seg_r2_list)) if seg_r2_list else np.nan,
            'fit_params': seg_fit_params,
            'column_labels': list(original_columns),
            'intervals': seg_intervals,
            'offset': _infer_segment_offset(original_columns),
        }

        all_r2_scores.extend(seg_r2_list)
        all_model_types.extend(seg_model_types)
        all_fit_params.extend(seg_fit_params)

    combined_delta = combined_delta.loc[:, intensity.columns].astype(np.float32)
    combined_fitted = combined_fitted.loc[:, intensity.columns].astype(np.float32)

    global_mean_r2 = float(np.nanmean(all_r2_scores)) if all_r2_scores else np.nan
    global_median_r2 = float(np.nanmedian(all_r2_scores)) if all_r2_scores else np.nan

    quality_info = {
        'r2_scores': all_r2_scores,
        'model_types': all_model_types,
        'mean_r2': global_mean_r2,
        'median_r2': global_median_r2,
        'fit_params': all_fit_params,
        'segments': segment_quality,
    }

    return combined_delta, combined_fitted, quality_info

def fitted_F_base(intensity, intervals, baseline_pre, baseline_post, vps_setting=1):
    # baseline_intervals = [(start - baseline_pre, start - baseline_post) for start, _ in intervals]
    # improved
    baseline_intervals = []
    if intervals:
        first_start = intervals[0][0]
        baseline_intervals.append((max(0, first_start - 20*vps_setting), max(0, first_start - 3)))
        baseline_intervals.extend([(start - baseline_pre*vps_setting, start - baseline_post*vps_setting) for start, _ in intervals[1:]])
    else:
        baseline_intervals = []
    
    # 添加最后一个buffer的最后一部分
    baseline_intervals.append((intensity.shape[1]-20*vps_setting, intensity.shape[1]))

    # 确保基线区间不超出数据的时间范围
    baseline_intervals = [(max(0, start), min(end, intensity.shape[1])) for start, end in baseline_intervals]

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
        
        finite_mask = np.isfinite(y_data)
        x_data = x_data[finite_mask]
        y_data = y_data[finite_mask]

        if len(x_data) == 0 or len(y_data) == 0:
            mean_val = np.nanmean(intensity.iloc[roi_idx])
            fitted_F0 = np.ones(len(time_axis)) * mean_val
            fitted_F0_curves.append(fitted_F0)
            fitted_params.append([0, 0, mean_val])
            r2_scores.append(0)
            model_types.append("mean")
            continue

        # 排序确保时间序列有序
        sorted_indices = np.argsort(x_data)
        x_data = x_data[sorted_indices]
        y_data = y_data[sorted_indices]
        
        # 去除异常值: 使用3-sigma规则
        y_mean = np.mean(y_data)
        y_std = np.std(y_data)
        if np.isnan(y_std) or y_std == 0:
            valid_idx = np.arange(len(y_data))
        else:
            valid_idx = np.where(np.abs(y_data - y_mean) < 3 * y_std)[0]

        x_data = x_data[valid_idx]
        y_data = y_data[valid_idx]

        if len(x_data) == 0 or len(y_data) == 0:
            mean_val = np.nanmean(intensity.iloc[roi_idx])
            fitted_F0 = np.ones(len(time_axis)) * mean_val
            fitted_F0_curves.append(fitted_F0)
            fitted_params.append([0, 0, mean_val])
            r2_scores.append(0)
            model_types.append("mean")
            continue
        
        # 自动选择最佳拟合模型
        best_func, best_params, r2 = fit_baseline(x_data, y_data, model_type='auto')
        
        # 确定模型类型
        if best_func.__name__ == 'exponential_decay':
            model_type = "exponential"
            # 转换为原始格式以兼容
            popt = best_params
        elif best_func.__name__ == 'polynomial':
            model_type = "polynomial"
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

def fit_baseline(x_data, y_data, model_type='exp'):
    """
    根据数据特性选择最佳拟合模型
    
    Parameters:
    - x_data: 时间点数组
    - y_data: 相应的强度值
    - model_type: 模型类型，可选 'exp'(指数), 'poly'(多项式), 'auto'(自动选择)
    
    Returns:
    - best_func: 最佳拟合函数 (可调用, f(t, *params))
    - best_params: 最佳拟合参数 (numpy 数组)
    - r2_score: 拟合优度 (R²)
    """
    
    # --- 模型定义 ---
    def exponential_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    def polynomial(t, a, b, c, d):
        return a * t**3 + b * t**2 + c * t + d
    
    def double_exponential(t, a1, b1, a2, b2, c):
        return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t) + c
    
    # --- 内部辅助函数，用于拟合和评分 ---
    def _fit_and_score(model, x, y, p0, bounds=(-np.inf, np.inf)):
        try:
            with warnings.catch_warnings():
                # 忽略 curve_fit 可能抛出的 OptimizeWarning
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)
            
            y_pred = model(x, *popt)
            # 使用 sklearn 计算 R²，更稳健
            r2 = r2_score(y, y_pred)
            return model, popt, r2
        except RuntimeError:
            # 拟合失败
            return None, None, -np.inf
        except ValueError:
            # 边界或 p0 维度不匹配等问题
            return None, None, -np.inf

    # --- 拟合逻辑 ---
    best_r2 = -np.inf
    best_func = None
    best_params = None

    models_to_try = []

    # 准备要尝试的模型、初始值和边界
    if len(x_data) == 0 or len(y_data) == 0:
        mean_val = float(np.nanmean(y_data) if len(y_data) else 0.0)
        best_func = lambda t, c=mean_val: np.full_like(np.asarray(t), c, dtype=np.float64)
        best_params = np.array([mean_val])
        return best_func, best_params, 0.0

    if len(x_data) == 1:
        mean_val = float(y_data[0])
        best_func = lambda t, c=mean_val: np.full_like(np.asarray(t), c, dtype=np.float64)
        best_params = np.array([mean_val])
        return best_func, best_params, 0.0

    y_min, y_max, y_mean = np.min(y_data), np.max(y_data), np.mean(y_data)
    
    if model_type == 'exp' or model_type == 'auto':
        p0_exp = [y_max, 0.001, y_min]
        # 边界 (a, b, c): 强制 b (衰减率) > 0
        bounds_exp = ([-np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf])
        models_to_try.append(("exp", exponential_decay, p0_exp, bounds_exp))

    if model_type == 'poly' or model_type == 'auto':
        p0_poly = [0, 0, 0, y_mean]
        models_to_try.append(("poly", polynomial, p0_poly, (-np.inf, np.inf)))
        
    if model_type == 'auto':
        p0_double_exp = [y_max*0.5, 0.01, y_max*0.5, 0.001, y_min]
        # 边界 (a1, b1, a2, b2, c): 强制 b1, b2 > 0
        bounds_double_exp = ([-np.inf, 1e-9, -np.inf, 1e-9, -np.inf], 
                             [np.inf, np.inf, np.inf, np.inf, np.inf])
        models_to_try.append(("double_exp", double_exponential, p0_double_exp, bounds_double_exp))

    # 循环尝试所有模型
    for name, model, p0, bounds in models_to_try:
        if len(x_data) < len(np.atleast_1d(p0)):
            continue
        func, params, r2 = _fit_and_score(model, x_data, y_data, p0, bounds)
        if r2 > best_r2:
            best_r2 = r2
            best_func = func
            best_params = params

    # --- 稳健的后备方案 ---
    if best_func is None:
        if len(x_data) >= 2:
            try:
                best_params = np.polyfit(x_data, y_data, 1)

                def linear_fit(t, a, b):
                    return a * t + b

                best_func = linear_fit
                y_pred_lin = linear_fit(x_data, *best_params)
                best_r2 = r2_score(y_data, y_pred_lin)

            except np.linalg.LinAlgError:
                mean_val = float(np.mean(y_data))
                best_func = lambda t, c=mean_val: np.full_like(np.asarray(t), c, dtype=np.float64)
                best_params = np.array([mean_val])
                best_r2 = 0.0
        else:
            mean_val = float(np.mean(y_data))
            best_func = lambda t, c=mean_val: np.full_like(np.asarray(t), c, dtype=np.float64)
            best_params = np.array([mean_val])
            best_r2 = 0.0

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

def visualize_fitting_quality(intensity, key, intervals, fitted_F0_df, r2_scores, model_types, baseline_pre=5, baseline_post=0):
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