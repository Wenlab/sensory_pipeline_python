import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go

def _color_to_rgba(color, alpha=0.1):
    """
    使用 Matplotlib 将任何颜色格式转换为带透明度的 rgba 字符串。
    
    :param color: 颜色 (支持 'red', '#ff0000', '#f00', 'rgb(255,0,0)', (1,0,0))
    :param alpha: 强制设置的透明度
    :return: 'rgba(255, 0, 0, 0.1)' 格式的字符串
    """
    # 1. 将任何格式的颜色转换为 (R, G, B, A) 元组，值范围 0-1
    #    (注意：mcolors.to_rgba 会保留原始的 alpha)
    r, g, b, _ = mcolors.to_rgba(color)
    
    # 2. 转换回 0-255 并应用新的 alpha
    r_int, g_int, b_int = int(r * 255), int(g * 255), int(b * 255)
    
    return f'rgba({r_int},{g_int},{b_int},{alpha})'

def add_regions_to_fig(fig, intervals, stimulus_list=None, 
                         name=None, color='rgba(170,0,0,0.1)', 
                         y0=0, y1=1, xref='x', yref='paper', 
                         showlegend=True, alpha=0.1, **kwargs):
    """
    直接向 Plotly Figure 添加高亮区域 (使用 fig.add_vrect)。

    :param fig: 要修改的 Plotly Figure 对象
    :param intervals: 时间间隔列表 (N*2 数组或元组列表)
    :param stimulus_list: (可选) 与 intervals 等长的刺激名称列表
    :param name: (单一模式) 区域名称
    :param color: (单一模式) 区域颜色。或 (多重模式) 颜色列表
    :param alpha: 强制设置的透明度
    """
    
    color_map = {}
    
    if stimulus_list is None:
        region_name = name if name is not None else 'Region'
        final_color = _color_to_rgba(color, alpha)
        color_map[region_name] = final_color
        names_to_loop = [region_name] * len(intervals)
    
    else:
        if len(intervals) != len(stimulus_list):
            print(f"Warnings, the length of intervals ({len(intervals)}) does not match the length of stimulus_list ({len(stimulus_list)}).")
            stimulus_list = stimulus_list[:min(len(intervals), len(stimulus_list))]
            intervals = intervals[:min(len(intervals), len(stimulus_list))]
        
        unique_stimuli = sorted(list(set(stimulus_list)))
        n_stim = len(unique_stimuli)
        
        if isinstance(color, list):
            if len(color) < n_stim:
                print(f"Warning: Not enough colors provided. Using colormap.")
                cmap = cm.get_cmap('tab10')
                for i, stim in enumerate(unique_stimuli):
                    color_map[stim] = _color_to_rgba(cmap(i % 10), alpha)
            else:
                for i, stim in enumerate(unique_stimuli):
                    color_map[stim] = _color_to_rgba(color[i], alpha)
        else:
            cmap = cm.get_cmap('tab10') 
            for i, stim in enumerate(unique_stimuli):
                color_map[stim] = _color_to_rgba(cmap(i % 10), alpha)
        
        names_to_loop = stimulus_list
    
    for (start, end), stim_name in zip(intervals, names_to_loop):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color_map[stim_name],
            layer="below",
            line_width=1,
            showlegend=False,
            **kwargs
        )

    if showlegend:
        unique_stimuli = sorted(set(names_to_loop))
        for stim_name in unique_stimuli:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color_map[stim_name],
                    symbol='square'
                ),
                name=stim_name,
                showlegend=True,
                hoverinfo='skip'
            ))


def draw_waterfall_plot(x, y_dict, y_offset, id_list=None, fill_area=True, fig=None, **kwargs):
    if fig is None:
        fig = go.FigureWidget()
    # else:
    #     fig.data = ()  # 清空现有数据
    if id_list is None:
        id_list = sorted(y_dict.keys())
    colors = cm.hsv(np.linspace(0, 1, len(id_list)))  # 使用渐变色
    id_name_dict = kwargs.pop('id_name_dict', {})
    # ymax = 0
    traces = []
    for i, neuron_id in enumerate(id_list):
        y = y_dict[int(neuron_id)]
        y_with_offset = y - min(y) + y_offset * (len(id_list) - i)
        # ymax = max(ymax, y_with_offset.max())
        # 创建辅助轨迹 - 目标水平线
        if fill_area:
            traces.append(go.Scatter(
                x=x,
                y=[y_offset * (len(id_list) - i)] * len(x),  # 创建一条与x等长的水平线
                mode='lines',
                line=dict(color='rgba(255,255,255,0)'),  # 完全透明
                showlegend=False
            ))
        traces.append(
            go.Scatter(
                x=x,
                y=y_with_offset,
                line=dict(
                    width=1.5,
                    color=f"rgba({colors[i][0]*255}, {colors[i][1]*255}, {colors[i][2]*255}, 0.8)",
                ),
                name=f"Neuron: {neuron_id if neuron_id not in id_name_dict else id_name_dict[neuron_id]}",
                hovertemplate=f"Neuron: {neuron_id if neuron_id not in id_name_dict else str(id_name_dict[neuron_id])+"("+str(neuron_id)+")"}"+'<br>data: %{customdata:.2f}<br>volume: %{x}<extra></extra>',
                customdata=y,
                fill="tonexty" if fill_area else 'none',
                **kwargs,
            )
        )
        
    layout = go.Layout(
        template="simple_white", # "plotly_dark"
        paper_bgcolor="white", # "black"
        plot_bgcolor="white", # "black"
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=0),  # Adjusted top margin to hide titles
        # yaxis=dict(range=[0, ymax])
    )
    overwrite = True if len(fig.data)!=len(traces) else False
    fig.update({'data': traces, 'layout': layout}, overwrite=overwrite)

    return fig