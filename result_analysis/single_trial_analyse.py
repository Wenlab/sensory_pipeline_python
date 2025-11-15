# @ Cheng Yukun
# 分布分析与可视化
import json
import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from matplotlib.colors import to_hex

from data_load.curve_fit import calculate_delta_F_over_F0
from data_load.get_stimulus_info import extract_intervals, get_stimulus_info
from data_load.load_worm_data import load_worm_ID
from utils.HDF5Toolkit import load_h5file

try:
    from result_plot.visweb import generate_compound_color_scheme
except ImportError:  # pragma: no cover - optional dependency
    generate_compound_color_scheme = None


# %%
# --- 聚类、排序和可视化 ---
from scipy.signal import correlate, correlation_lags
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.graph_objs as go
import plotly.subplots as psub
import scipy.cluster.hierarchy as sch
import numpy as np

def cluster_by_corr_then_sort_by_delay(signal_dict, max_lag=10, ref_method='max_var'):
    # 步骤1: 预处理（标准化）
    def standardize(signal):
        return (signal - np.mean(signal))/max(np.mean(signal),1)

    std_signals = {ch: standardize(sig) for ch, sig in signal_dict.items()}
    channels = list(signal_dict.keys())
    n_channels = len(channels)

    # 步骤2-3: 计算互相关并提取延迟
    delay_matrix = np.zeros((n_channels, n_channels))   # j 相对于 i 的延迟
    corr_matrix = np.zeros((n_channels, n_channels))    # 最大互相关值

    for i, ch_i in enumerate(channels):
        for j, ch_j in enumerate(channels):
            # 获取标准化信号
            sig_i = std_signals[ch_i]
            sig_j = std_signals[ch_j]
            
            # 计算互相关
            corr = correlate(sig_i, sig_j, mode='full')
            lags = correlation_lags(len(sig_i), len(sig_j), mode='full')
            
            # 限制延迟范围
            mask = (np.abs(lags) <= max_lag)
            corr_sub = corr[mask]
            lags_sub = lags[mask]
            
            # 找到峰值位置（忽略边界噪音）
            peak_idx = np.argmax(np.abs(corr_sub))  # 或取正峰值 np.argmax(corr_sub)
            delay = lags_sub[peak_idx]
            max_corr = corr_sub[peak_idx]
            
            # 存储结果
            delay_matrix[i, j] = delay
            corr_matrix[i, j] = max_corr
            # if ch_i ==27 and ch_j == 72:
            #     print(f"Debug: ch_i={ch_i}, ch_j={ch_j}, delay={delay}, max_corr={max_corr}")
            #     plt.figure()
            #     plt.plot(sig_i)
            #     plt.plot(sig_j)
            #     plt.figure()
            #     plt.plot(sig_i, sig_j)
            #     plt.figure()
            #     plt.plot(lags_sub, corr_sub)
            #     plt.title(f"Cross-correlation between {ch_i} and {ch_j}")
            #     plt.xlabel("Lag")
            #     plt.ylabel("Correlation")
            #     plt.axvline(x=delay, color='r', linestyle='--', label=f"Delay={delay}")
            #     plt.legend()
            #     plt.show()
    # 1. 基于相关矩阵进行层次聚类
    # corr_abs = corr_matrix.copy()  # 使用绝对值表示相似度
    corr_abs = np.max(corr_matrix) - corr_matrix  # 转换为距离矩阵
    np.fill_diagonal(corr_abs, 0)   # 忽略对角线(自相关)
    # print(corr_abs)
    Z = linkage(corr_abs, method='ward')  # 使用Ward方法进行层次聚类

    # 自动确定聚类数量（基于距离阈值）
    max_dist = np.max(Z[:, 2]) * 0.5  # 使用最大距离的50%作为阈值
    clusters = fcluster(Z, max_dist, criterion='distance')

    # 2. 计算每个通道的方差（原始信号，未标准化）
    variances = {ch: np.var(signal) for ch, signal in signal_dict.items()}

    # 3. 在每个聚类内部选取参考通道并排序
    # 支持两种参考通道选择方法: 'max_var'（最大方差）或 'center'（聚类中心）
    def get_ref_channel_idx(cluster_indices, method='max_var'):
        if method == 'max_var':
            # 方差最大
            cluster_variances = {idx: variances[channels[idx]] for idx in cluster_indices}
            return max(cluster_variances, key=cluster_variances.get)
        elif method == 'center':
            # 距离聚类中心最近
            # 计算聚类内相关矩阵的均值（中心）
            sub_corr = corr_matrix[np.ix_(cluster_indices, cluster_indices)]
            center_idx = np.argmax(np.sum(sub_corr, axis=1))
            return cluster_indices[center_idx]
        else:
            raise ValueError("method must be 'max_var' or 'center'")

    cluster_refs = {}  # 存储每个聚类的参考通道
    sorted_indices = []  # 排序后的通道索引列表
    cluster_refs_idx = {}  # 存储每个聚类参考通道的索引

    for cluster_id in np.unique(clusters):
        # 获取当前聚类中的通道索引
        cluster_indices = np.where(clusters == cluster_id)[0]
        
        # 检查聚类中是否只有一个通道
        if len(cluster_indices) == 1:
            # 单通道聚类：直接添加到排序列表
            sorted_indices.extend(cluster_indices)
            # 使用自身作为参考
            ref_channel_idx = cluster_indices[0]
            ref_channel_name = channels[ref_channel_idx]
            cluster_refs[cluster_id] = ref_channel_name
            continue
        
        # # 在聚类内部选取方差最大的通道作为参考
        # # 计算聚类内每个通道的方差
        # cluster_variances = {idx: variances[channels[idx]] for idx in cluster_indices}
        # # 找到方差最大的通道索引
        # ref_channel_idx = max(cluster_variances, key=cluster_variances.get)

        ref_channel_idx = get_ref_channel_idx(cluster_indices, method=ref_method)
        ref_channel_name = channels[ref_channel_idx]
        cluster_refs[cluster_id] = ref_channel_name
        cluster_refs_idx[cluster_id] = ref_channel_idx
        
        # 获取当前聚类内所有通道相对于参考通道的延迟
        cluster_delays = [delay_matrix[ref_channel_idx, idx] for idx in cluster_indices]
        
        # 按延迟从小到大排序（提前的在前，滞后的在后）
        # 同时保留通道原始索引
        sorted_cluster = sorted(zip(cluster_indices, cluster_delays), key=lambda x: x[1])
        
        # 添加到总体排序列表
        sorted_indices.extend([idx for idx, _ in sorted_cluster])

    # 4. 应用排序结果
    # 新的通道顺序
    sorted_channels = [channels[i] for i in sorted_indices]

    # 重排相关矩阵和延迟矩阵
    sorted_corr_matrix = corr_matrix[sorted_indices, :][:, sorted_indices]
    sorted_delay_matrix = delay_matrix[sorted_indices, :][:, sorted_indices]
    
    # return (sorted_channels, sorted_corr_matrix, sorted_delay_matrix, clusters, cluster_refs, Z, max_dist)

    # 5. 使用 plotly 绘制排序后的热图
    # 创建子图
    fig_corr = go.FigureWidget(psub.make_subplots(rows=1, cols=2, subplot_titles=('排序后的通道互相关矩阵', '排序后的通道延迟矩阵')))

    # 相关矩阵热图
    # print(sorted_corr_matrix)
    heatmap_corr = go.Heatmap(
        z=sorted_corr_matrix,
        y=[str(id) for id in sorted_channels],
        x=[str(id) for id in sorted_channels],
        colorscale='RdBu_r',
        # colorbar=dict(title='互相关值'),
        zmid=0
    )
    fig_corr.add_trace(heatmap_corr, row=1, col=1)

    # 延迟矩阵热图
    heatmap_delay = go.Heatmap(
        z=sorted_delay_matrix,
        y=[str(id) for id in sorted_channels],
        x=[str(id) for id in sorted_channels],
        colorscale='RdBu_r',
        colorbar=dict(title='延迟(样本数)'),
        zmid=0
    )
    fig_corr.add_trace(heatmap_delay, row=1, col=2)

    # 添加聚类分隔线和参考通道标记
    cumulative_size = 0
    cluster_ids = np.unique(clusters)
    cluster_sizes = [np.sum(clusters == cid) for cid in cluster_ids]

    for cluster_id, size in zip(cluster_ids, cluster_sizes):
        # 分隔线位置
        if cumulative_size > 0:
            for i in [1, 2]:
                fig_corr.add_shape(
                    type="line",
                    x0=cumulative_size - 0.5, x1=cumulative_size - 0.5,
                    y0=-0.5, y1=len(sorted_channels)-0.5,
                    line=dict(color="black", width=2),
                    xref=f"x{i}", yref=f"y{i}"
                )
                fig_corr.add_shape(
                    type="line",
                    x0=-0.5, x1=len(sorted_channels)-0.5,
                    y0=cumulative_size - 0.5, y1=cumulative_size - 0.5,
                    line=dict(color="black", width=2),
                    xref=f"x{i}", yref=f"y{i}"
                )
        # 参考通道标记
        # ref_pos = cumulative_size + size / 2 - 0.5
        for i in [1, 2]:
            fig_corr.add_trace(
                go.Scatter(
                    # x=[sorted_channels[int(ref_pos)]],
                    # y=[sorted_channels[int(ref_pos)]],
                    x= [cluster_refs[cluster_id]],
                    y= [cluster_refs[cluster_id]],
                    mode='markers',
                    marker=dict(size=8, symbol='square-open', color='gold', line=dict(width=2)),
                    name=f'类{cluster_id}参考' if (i == 1 and cluster_id == 1) else "",
                    showlegend=bool(i == 1 and cluster_id == 1)
                ),
                row=1, col=i
            )
        cumulative_size += size

    fig_corr.update_xaxes(tickangle=90)
    fig_corr.update_layout(height=600, width=1200, title_text="排序后的通道互相关与延迟矩阵", showlegend=True)
    # display(fig_corr)

    # 6. 输出聚类排序结果
    print("\n聚类和排序结果：")
    print(f"检测到 {len(np.unique(clusters))} 个信号聚类")
    print("通道排序 (按聚类分组并类内按延迟排序):")

    cumulative_size = 0
    for cluster_id, size in zip(cluster_ids, cluster_sizes):
        cluster_group = sorted_channels[cumulative_size:cumulative_size + size]
        ref_channel = cluster_refs[cluster_id]
        
        print(f"\n聚类 #{cluster_id} (包含 {size} 个通道):")
        print(f"参考通道: {ref_channel} (方差最大)")
        
        for ch in cluster_group:
            if ch == ref_channel:
                print(f"  - {ch} (参考通道): 参考延迟")
            else:
                orig_idx = sorted_indices[sorted_channels.index(ch)]
                rel_delay = delay_matrix[channels.index(ref_channel), orig_idx]
                print(f"  - {ch}: 相对于参考通道 {ref_channel} 的延迟 = {rel_delay:.1f} 样本")
        cumulative_size += size

    # 7. 使用 plotly 可视化聚类树状图
    dendro = sch.dendrogram(Z, labels=channels, orientation='top', no_plot=True)
    icoord = np.array(dendro['icoord'])
    dcoord = np.array(dendro['dcoord'])
    labels = dendro['ivl']

    fig_delay = go.FigureWidget()
    for xs, ys in zip(icoord, dcoord):
        fig_delay.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='black')))
    fig_delay.update_layout(
        title='通道聚类树状图',
        xaxis=dict(
            tickvals=[(xs[1]+xs[2])/2 for xs in icoord],
            ticktext=labels,
            tickangle=90,
            title='通道标识'
        ),
        yaxis=dict(title='聚类距离'),
        height=400, width=900
    )
    fig_delay.add_shape(
        type="line",
        x0=0, x1=10*len(labels),
        y0=max_dist, y1=max_dist,
        line=dict(color="red", dash="dash"),
    )
    # display(fig_delay)
    
    return (sorted_channels, sorted_corr_matrix, sorted_delay_matrix, fig_corr, fig_delay)



# %%
# --- Single trial analysis pipeline ---

def _load_single_trial_intensity(
    h5_file_path: str,
    root_name: Optional[str] = None,
):
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"H5 file not found: {h5_file_path}")

    resolved_root = root_name
    if resolved_root is None:
        with h5py.File(h5_file_path, "r") as h5_file:
            keys = list(h5_file.keys())
            if not keys:
                raise ValueError(f"No root groups found in {h5_file_path}")
            resolved_root = keys[0]

    data = load_h5file(h5_file_path, resolved_root)

    intensity = np.asarray(data.get("intensity", []), dtype=np.float32)
    if intensity.ndim == 1:
        intensity = intensity[np.newaxis, :]
    if intensity.size == 0:
        raise ValueError(f"No intensity dataset found under '{resolved_root}' in {h5_file_path}")

    digital_ids = data.get("ID")
    if digital_ids is None:
        digital_ids = np.arange(intensity.shape[0])
    digital_ids = np.asarray(digital_ids)

    n_seq = data.get("n_seg")
    if n_seq is None:
        n_seq = np.array([])
    else:
        n_seq = np.asarray(n_seq)

    return intensity, digital_ids, n_seq, resolved_root


def _to_python_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _to_python_scalar(value.item())
        return [
            _to_python_scalar(element)
            for element in value.tolist()
        ]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (np.generic,)):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fallback path
            return value
    return value


def _normalize_neuron_ids(raw_ids: Sequence) -> List[str]:
    normalized = []
    for idx, value in enumerate(raw_ids):
        converted = _to_python_scalar(value)
        if converted is None or (isinstance(converted, float) and np.isnan(converted)):
            converted = idx
        normalized.append(str(converted))
    return normalized


def _resolve_stimulus_metadata(
    labjack_excel_path: Optional[str],
    exp_name: str,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    if not labjack_excel_path:
        return [], []

    experiment_df, stimulus_lists = get_stimulus_info(
        labjack_excel_path,
        if_generate_stimulus_lists=True,
    )
    stimulus_intervals, _ = extract_intervals(experiment_df, exp_name)
    stimulus_list = stimulus_lists.get(exp_name, []) if stimulus_lists else []
    return stimulus_intervals, stimulus_list


def _prepare_delta_f(
    intensity: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    n_seq: Sequence,
    baseline_pre: int,
    baseline_post: int,
    vps_setting: int,
    background_noise: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    intensity_df = pd.DataFrame(intensity)
    delta_df, fitted_df, quality = calculate_delta_F_over_F0(
        intensity_df,
        list(intervals),
        baseline_pre=baseline_pre,
        baseline_post=baseline_post,
        vps_setting=vps_setting,
        background_noise=background_noise,
        n_seq=n_seq,
    )

    delta = delta_df.to_numpy(dtype=np.float32, copy=True)
    fitted = fitted_df.to_numpy(dtype=np.float32, copy=True) if fitted_df is not None else None
    return delta, fitted, quality


def _safe_nanmax(values: np.ndarray) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmax(values))


def _safe_nansum(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    return float(np.nansum(values))


def _iter_interval_segments(trace: np.ndarray, intervals: Sequence[Tuple[int, int]]):
    if not intervals:
        yield trace
        return
    length = trace.shape[0]
    for start, end in intervals:
        start_idx = int(max(0, np.floor(start)))
        end_idx = int(min(length, np.ceil(end)))
        if end_idx <= start_idx:
            continue
        yield trace[start_idx:end_idx]


def _resolve_stimulus_colors(
    stimulus_list: Sequence[str],
    provided_colors: Optional[Dict[str, str]],
) -> Dict[str, str]:
    if provided_colors:
        return dict(provided_colors)

    if not stimulus_list:
        return {}

    unique = list(dict.fromkeys(stimulus_list))

    if generate_compound_color_scheme is not None:
        try:
            palette = generate_compound_color_scheme(unique)
            if isinstance(palette, dict):
                return {key: to_hex(value) if not isinstance(value, str) else value for key, value in palette.items()}
        except Exception:  # pragma: no cover - fallback palette
            pass

    cmap = plt.get_cmap("tab20")
    return {
        stimulus: to_hex(cmap(idx / max(1, len(unique) - 1)))
        for idx, stimulus in enumerate(unique)
    }