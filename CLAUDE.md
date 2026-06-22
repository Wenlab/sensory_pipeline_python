# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 使用中文回答。

## Environment

This project uses **[pixi](https://pixi.sh)** for package management. Dependencies are defined in `pixi.toml`. The pixi environment is in `.pixi/` (gitignored).

```bash
# Install dependencies
pixi install

# Run a script within the pixi environment
pixi run python <script.py>
```

Tests use `pytest` (no existing project tests — test infrastructure is TBD).

## Project overview

Processing pipeline for **fluorescence calcium imaging data** from _C. elegans_ sensory neurons. Raw TIFF time-series from a two-camera microscope (red/green channels) are converted to fluorescence intensity traces, artifact-corrected, normalized to dF/F0, and analyzed via PCA/dPCA, clustering, and interactive visualization.

Data flows through **HDF5 files** at each major stage. The primary outputs are processed HDF5 files, analysis results, and Dash-based interactive plots.

## Pipeline stages

### 1. Stimulus extraction (`channel_info_get/`)
- `extract_channel_info.py` — `ExtractChannelInfo` class: reads LabJack digital channel logs from TIFF metadata or Excel, maps 8-bit channel states to stimulus labels (Buffer, Odor1–5, Control1–2, All Off).
- `stimulus_config_builder.py` — Generates channel configuration JSON and stimulus metadata (color schemes, odor names) for downstream use.

### 2. Raw data conversion (`in_experiment/`)
- `transfer_tiff2npy.py` — Converts raw multi-TIFF stacks to `.npy` volumes using dask for memory efficiency. Groups folders by worm ID. Handles both red and green camera channels.
- `process_labjack_data.py` — Extracts trial timing from LabJack data; generates stimulus interval Excel files used by the data loading stage.

### 3. Neuron identification (`id_identify/`)
- `id_identify_in_napari.py` — Opens `.npy` volumes in napari for manual neuron ROI annotation. Draws bounding boxes around identified neurons.
- `trace_check_in_napari.py` — Visualizes extracted traces overlaying the image to validate neuron identification.

### 4. Data loading & preprocessing (`data_load/`)
- `load_worm_data.py` — **Main entry point**: loads HDF5 intensity data, applies artifact denoising, computes dF/F0 via exponential curve fitting. Uses `n_seg` to split data into recording segments.
- `preprocessing.py` — **Artifact denoising**: detects short V/inverse-V reversal artifacts via robust MAD-based thresholding. Two-pass: per-neuron detection then frame-level synchronization check. Repairs short artifacts with linear interpolation. This replaced the older step-drop detection approach.
- `curve_fit.py` — Computes dF/F0: fits baseline fluorescence F0 for each neuron, then `(F - F0) / F0`. Handles segment-aware fitting via `n_seg`.
- `get_stimulus_info.py` — Parses experiment Excel files to extract stimulus/buffer intervals and generate stimulus name lists.
- `process_worm_data.py` — Assembles trial-level data from per-worm dF/F0 data into `neuron_segments_dict` structure: `{neuron_group: {stimulus_type: [trial_dict, ...]}}` — the standard data format for all downstream analysis.
- `interpolation.py` — Replaces bad-data regions in HDF5 intensity arrays via cubic spline, polynomial, moving average, or linear interpolation.
- `df_sort.py` — Utility for sorting DataFrames by stimulus order.

### 5. Analysis (`result_analysis/`)
- `baseline_correction.py` — Pre-stimulus baseline subtraction for individual trials. **Active — do not archive.**

### 6. Visualization (`result_plot/`)
- `visweb.py` — **Main Dash app**: interactive web dashboard for exploring neuronal responses. Supports neuron/stimulus selection, time window cropping (stimulus ON → OFF+10s), and Plotly-based figures.
- `draw_single_signal_plotly.py` — Single-trial waterfall plots with stimulus shading and preprocessing mode selection (none/conservative/aggressive).
- `draw_trials_heatmap.py` — Heatmap visualization of trial responses across neurons and stimuli.
- `draw_raw_signal.py` — Raw intensity trace plotting.
- `draw_mean_signal.py` — Mean ± SD response plots.
- `radar_heatmap_plot.py` — Radar/heatmap plots for multi-dimensional response profiles.
- `vis_metabolism_pca.py` — PCA dashboard for metabolism-related analyses.

### 7. Utilities (`utils/`)
- `HDF5Toolkit.py` — Generic HDF5 save/load with type preservation (numpy, torch, dicts, scalars).
- `interpolate.py` — NaN interpolation over time dimension with max-gap awareness.
- `parse_stimulus_info.py` — Groups stimulus labels by compound name, parses concentration strings (`%`, `uM`, `mM`, scientific notation), generates sort keys.
- `vis_plotly_utils.py` — Shared Plotly helpers (waterfall plots, stimulus region shading).
- `read_vols_using_dask.py` — Dask-based lazy reading of large multi-TIFF volumes.
- `copy_tiff_using_dask.py` — Dask-based copying of TIFF stacks.
- `tiff_to_video_with_labels.py` — Converts TIFF stacks to labeled videos.
- `interpolate_neuron.py` — Neuron-specific interpolation.
- `load_albert_data.py` — Data loader for Albert lab format.
- `concentration_calculation.py` — Concentration math utilities.
- `prints.py` — Print formatting helpers.
- `html_generate.py` — HTML report generation.

### 8. Usage scripts (`usage/`)
- `context.py` — Context for usage examples.
- `merge_h5.py` — Merge multiple HDF5 files.
- `check_alignment.py` — Alignment verification between channels.
- `plot_dout_8tuple.py` — Specialized 8-tuple visualization.

## Key data structures

**HDF5 layout** (per worm group, e.g. `w1`):
```
w1/
  intensity: (n_neurons, n_timepoints) float64
  n_seg: [seg0_len, seg1_len, ...]  — recording segment boundaries
```

**neuron_segments_dict** (standard analysis format):
```python
{
  neuron_group_id: {           # e.g. "AWA", "AWB", "0"
    stimulus_code: [           # e.g. "c1_1", "Buffer"
      {
        "deltaFoverF_0": np.ndarray,  # dF/F0 trace
        "start_time": int,            # stimulus onset frame
        "end_time": int,              # stimulus offset frame
        "worm_key": str,              # source worm
        "segment_index": int,         # trial index
        "stim_name": str,             # human-readable name
        "stim_color": str,            # hex color
      },
      ...
    ]
  }
}
```

## Artifact denoising (important detail)

The `preprocessing.py` module replaced older step-drop detection. Key functions:

- `_detect_reversal_artifacts(data, diff_factor, max_run)` — Per-neuron V/inverse-V spike detection using robust MAD of first differences. No smoothing applied — preserves temporal fidelity.
- `detect_shared_bad_frames(...)` — Frames where ≥`shared_threshold` fraction of neurons show artifacts are flagged as shared bad frames.
- `repair_short_artifacts(intensity, bad_mask, max_interp_gap)` — Linear interpolation across short artifact runs (≤`max_interp_gap`). Longer runs and edge runs are left untouched.
- `denoise_frame_artifacts(...)` — **Recommended entry point**: combines detection + repair. `repair_all_neurons_at_shared_frames=True` repairs all neurons at frames where many neurons spike simultaneously (assumed systematic).

Two preset modes:
- **Conservative**: `max_interp_gap=4` — only very short artifacts
- **Aggressive**: `max_interp_gap=32` — wider artifact repair (used in production)

## Platform notes

Developed on Windows. Uses `osx-arm64` as a secondary pixi platform for Mac compatibility. The `numba` dependency is platform-conditional (win-64 uses PyPI, osx-arm64 uses conda-forge). Path separators in the codebase use raw strings with backslashes — use `pathlib` for new code.

## Git Workflow

After completing every task that modifies files, ask the user: **"Commit these changes to git?"**

If the user says yes, invoke the `/commit` skill to generate a Conventional Commits message and commit.
