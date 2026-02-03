import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from result_analysis.single_trial_analyse import cluster_by_corr_then_sort_by_delay, _load_single_trial_intensity, _resolve_stimulus_metadata, _prepare_delta_f
from utils.interpolate import interpolate_over_nans
from utils.vis_plotly_utils import add_regions_to_fig, draw_waterfall_plot
from data_load.preprocessing import detect_and_mask_step_drops

def prepare_signal_dict(h5_file_path, root_name, labjack_excel_path=None, raw_data=False):
    intensity, _, n_seq, _ = _load_single_trial_intensity(h5_file_path, root_name)
    if labjack_excel_path is not None:
        stimulus_intervals, stimulus_list = _resolve_stimulus_metadata(labjack_excel_path, root_name)
    else:
        stimulus_intervals, stimulus_list = [], []

    if raw_data:
        delta_f = intensity - np.nanmean(intensity, axis=1, keepdims=True)
        delta_f_interp, t_inter = interpolate_over_nans(delta_f)
    else:
        intensity = detect_and_mask_step_drops(intensity)
        delta_f, fitted, quality = _prepare_delta_f(intensity, stimulus_intervals, n_seq, baseline_pre=6, baseline_post=1, vps_setting=1, background_noise=102)
        delta_f_interp, t_inter = interpolate_over_nans(delta_f)

    signal_dict = {id:sig for id, sig in enumerate(delta_f_interp)}
    return signal_dict, stimulus_intervals, stimulus_list

def map_stim(stimulus_list, color_scheme, stim_name):
    stimulus_name_list = [stim_name[str(s)] for s in stimulus_list]
    color_list = [color_scheme.get(stimulus, "#000000") for stimulus in stimulus_list]
    return stimulus_name_list, color_list

def map_id(id_excel_path, root_name):
    df = pd.read_excel(id_excel_path, sheet_name=root_name)
    id_name_dict = {}
    digital_list = df["digital"].tolist()
    biological_list = df["biological"].tolist()
    for digital_id, biological_name in zip(digital_list, biological_list):
        if pd.isna(biological_name):
            id_name_dict[digital_id] = digital_id
        else:
            id_name_dict[digital_id] = biological_name
    return id_name_dict

def prepare_single_signal_plotly(h5_file_path, root_name, labjack_excel_path=None, id_excel_path=None, color_scheme=None, stim_name=None, raw_data=False):
    signal_dict, stimulus_intervals, stimulus_list = prepare_signal_dict(h5_file_path, root_name, labjack_excel_path, raw_data=raw_data)
    if id_excel_path is None:
        id_name_dict = {id:id for id in signal_dict.keys()}
    else:
        id_name_dict = map_id(id_excel_path, root_name)
    if color_scheme is None or stim_name is None:
        stimulus_name_list = [str(s) for s in stimulus_list]
        color_list = None
    else:
        stimulus_name_list, color_list = map_stim(stimulus_list, color_scheme, stim_name)

    return signal_dict, stimulus_intervals, stimulus_name_list, color_list, id_name_dict

def create_plotly_fig(h5_file_path, root_name, labjack_excel_path=None, id_excel_path=None, cluster=False, color_scheme=None, stim_name=None, raw_data=False, **kwargs):
    signal_dict, stimulus_intervals, stimulus_name_list, color_list, id_name_dict = prepare_single_signal_plotly(h5_file_path, root_name, labjack_excel_path, id_excel_path, color_scheme, stim_name, raw_data=raw_data)
    if cluster:
        sorted_ids, sorted_corr_matrix, sorted_delay_matrix, fig_corr, fig_delay = cluster_by_corr_then_sort_by_delay(signal_dict, max_lag=5,ref_method='center')
    else:
        sorted_ids = sorted(signal_dict.keys())
        fig_corr, fig_delay = None, None
    region_alpha = kwargs.pop('region_alpha', 0.3)
    show_region_legend = kwargs.pop('show_region_legend', True)
    fig_signal = draw_waterfall_plot(
        x= np.arange(signal_dict[0].shape[0]),
        y_dict=signal_dict,
        y_offset= kwargs.pop('y_offset', 1.5),
        id_list=sorted_ids,
        fill_area= kwargs.pop('fill_area', True),
        id_name_dict=id_name_dict,
        **kwargs
    )
    add_regions_to_fig(
        fig_signal,
        stimulus_intervals,
        stimulus_name_list,
        color=color_list,
        alpha=region_alpha,
        showlegend=show_region_legend,
    )

    return fig_signal, fig_corr, fig_delay

if __name__ == "__main__":
    h5_file_path = r"H:\Process_temporary\WJH\olfactory\infer_result\20251105\20251105.h5"
    labjack_excel_path = r"H:\Process_temporary\WJH\olfactory\labjack_result\20251105_bac\output_volumes_merged.xlsx"
    ID_path = r"H:\Process_temporary\WJH\olfactory\ID\result\20251105\20251105_1.xlsx"
    import json
    with open("H:\Process_temporary\WJH\sensory_pipeline_python\data/config/bacteria_color_scheme.json", "r") as f:
        color_scheme = json.load(f)
    with open("H:\Process_temporary\WJH\sensory_pipeline_python\data/config/bacteria.json", "r") as f:
        stim_name = json.load(f)
    from result_plot.draw_single_signal_plotly import create_plotly_fig
    fig_args = {
        "y_offset": 1.5,
        "fill_area": True,
        "brightness": "30%",
        "region_alpha": 0.3,
        "show_region_legend": True
    }

    fig_signal, fig_corr, fig_delay = create_plotly_fig(
        h5_file_path,
        "w4",
        labjack_excel_path,
        id_excel_path=None,
        cluster=True,
        color_scheme=color_scheme,
        stim_name=stim_name
    )