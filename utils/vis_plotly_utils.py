import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
from PIL import ImageColor
def _color_to_rgba(color, alpha=None)-> str:
    """
    将颜色字符串转换为 RGB 或 RGBA 格式。
    支持的输入格式为ImageColor.getrgb(color)可以解析的格式
    :param color: 颜色字符串，例如 'red', '#ff0000', 'rgb(255,0,0)', 'hsl(0,100%,50%)' 等。若格式中含有a值, 则返回rgba格式。
    也可以是 matplotlib 生成的 tuple/list/ndarray, 如 (0.1, 0.2, 0.3, 1.0)
    :param alpha: 可选的透明度值，范围为 0-255。如果提供了该值, 则返回 RGBA 格式。
    :return: RGB 或 RGBA 格式的字符串，例如 'rgb(255,0,0)' 或 'rgba(255,0,0,128)'。
    :raises ValueError: 如果颜色字符串无法解析或格式不正确。
    例子:
    >>> to_rgb('red')
    'rgb (255, 0, 0)'
    >>> to_rgb('#00ff00', 128)
    'rgba (0, 255, 0, 128)'
    >>> to_rgb('hsl(240,100%,50%)')
    'rgb (0, 0, 255)'
    >>> to_rgb('hsl(240,100%,50%)', 200)
    'rgba (0, 0, 255, 200)'
    """
    try:
        if isinstance(color, (tuple, list, np.ndarray)):
            # Handle tuple/list/ndarray input (e.g. from matplotlib)
            vals = [float(x) for x in color]
            if len(vals) < 3:
                 raise ValueError(f"Color tuple too short: {color}")
            
            r, g, b = vals[0], vals[1], vals[2]
            # Convert 0-1 float to 0-255 int
            if all(0 <= x <= 1.0 for x in [r, g, b]):
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
            else:
                r, g, b = int(r), int(g), int(b)
            
            rgb_tuple = (r, g, b)
            if len(vals) > 3:
                rgb_tuple += (vals[3],)
        else:
            rgb_tuple = ImageColor.getrgb(color)

        if len(rgb_tuple) == 3 and alpha is None:
            return f"rgb({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]})"
        elif len(rgb_tuple) == 4:
            if alpha is not None:
                rgb_tuple = rgb_tuple[:3] + (alpha,)
            return f"rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},{rgb_tuple[3]})"
        elif len(rgb_tuple) == 3 and alpha is not None:
            return f"rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},{alpha})"
        else:
            raise ValueError(f"无法解析颜色: {color}")
    except Exception as e:
        raise ValueError(f"无法解析颜色: {color}") from e

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
    # add hover
    if 'yaxis2' not in fig.layout:
        fig.update_layout(
            yaxis2=dict(
                range=[0, 1],
                overlaying='y',
                visible=False,
                showgrid=False,
                showticklabels=False
            )
        )

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
            if len(color) == len(stimulus_list):
                for stim_val, color_val in zip(stimulus_list, color):
                    if stim_val not in color_map:
                        color_map[stim_val] = _color_to_rgba(color_val, alpha)
            elif len(color) >= n_stim:
                for i, stim in enumerate(unique_stimuli):
                    color_map[stim] = _color_to_rgba(color[i], alpha)
            else:
                print(f"Warning: Not enough colors provided. Using colormap.")
                cmap = cm.get_cmap('tab10')
                for i, stim in enumerate(unique_stimuli):
                    color_map[stim] = _color_to_rgba(cmap(i % 10), alpha)
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
        hover_color = color_map[stim_name]
        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[0.99, 0.99],
            yaxis='y2',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=5),
            opacity=0,
            marker=dict(color='rgba(0,0,0,0)'),
            hovertemplate=f'<b>Stimulus: {stim_name}</b><br>Start: {start:.2f}<br>End: {end:.2f}<extra></extra>',
            hoverlabel=dict(bgcolor=hover_color, align='left'),
            showlegend=False
        ))

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
    
    id_list = list(reversed(id_list)) # 反转顺序以从下到上绘制
    # colors = cm.hsv(np.linspace(0, 1, len(id_list)))  # 使用渐变色
    colors = [_color_to_rgba(f'hsl({i*330/len(id_list)},100%,{kwargs.pop("brightness", "35%")})',0.8) for i in range(len(id_list))]
    id_name_dict = kwargs.pop('id_name_dict', {})
    # ymax = 0
    traces = []
    for i, neuron_id in enumerate(id_list):
        y = y_dict[int(neuron_id)]
        # y_with_offset = y - min(y) + y_offset * (len(id_list) - i)
        y_min = np.nanmin(y) if not np.all(np.isnan(y)) else 0
        y_with_offset = y - y_min + y_offset * i
        # ymax = max(ymax, y_with_offset.max())
        # 创建辅助轨迹 - 目标水平线
        if fill_area:
            traces.append(go.Scatter(
                x=x,
                # y=[y_offset * (len(id_list) - i)] * len(x),  # 创建一条与x等长的水平线
                y=[y_offset * i] * len(x),
                mode='lines',
                line=dict(color='rgba(255,255,255,0)'),  # 完全透明
                showlegend=False
            ))
        display_name = neuron_id
        if neuron_id in id_name_dict:
            name_from_dict = str(id_name_dict[neuron_id])

            if not name_from_dict.isdigit():
                display_name = f"{name_from_dict} ({neuron_id})"
        traces.append(
            go.Scatter(
                x=x,
                y=y_with_offset,
                line=dict(
                    width=1.5,
                    # color=f"rgba({colors[i][0]*255}, {colors[i][1]*255}, {colors[i][2]*255}, 0.8)",
                    color=colors[i],
                ),
                name=f"Neuron: {display_name}",
                hovertemplate=f"Neuron: {display_name}"+'<br>data: %{customdata:.2f}<br>volume: %{x}<extra></extra>',
                customdata=y,
                fill="tonexty" if fill_area else 'none',
                **kwargs,
            )
        )
        
    layout = go.Layout(
        template="simple_white", # "plotly_dark"
        paper_bgcolor="white", # "black"
        plot_bgcolor="white", # "black"
        # template="plotly_dark", # "plotly_dark"
        # paper_bgcolor="black", # "black"
        # plot_bgcolor="black", # "black"
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=0),  # Adjusted top margin to hide titles
        # yaxis=dict(range=[0, ymax])
    )
    overwrite = True if len(fig.data)!=len(traces) else False
    fig.update({'data': traces, 'layout': layout}, overwrite=overwrite)

    return fig
