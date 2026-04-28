import numpy as np
import pandas as pd

def _robust_diff_scale(data, min_scale=1e-6):
    """Return one robust first-derivative scale per trace."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 2:
        return np.full(data.shape[0], min_scale, dtype=float)

    diff = np.diff(data, axis=1)
    scales = np.empty(data.shape[0], dtype=float)
    for i, row in enumerate(diff):
        valid = row[np.isfinite(row)]
        if valid.size == 0:
            scales[i] = min_scale
            continue

        median_diff = np.median(valid)
        abs_dev = np.abs(valid - median_diff)
        mad = np.median(abs_dev)
        scale = 1.4826 * mad

        use_floor_scale = False
        lower_quartile_scale = 1.4826 * np.percentile(abs_dev, 25)
        if lower_quartile_scale > min_scale:
            scale = min(scale, lower_quartile_scale * 2)
        elif np.count_nonzero(abs_dev <= min_scale) >= valid.size / 2:
            scale = min_scale
            use_floor_scale = True

        if (not np.isfinite(scale) or scale <= min_scale) and not use_floor_scale:
            scale = np.std(valid) + min_scale
        scales[i] = max(scale, min_scale)
    return scales


def _find_runs(mask):
    """Yield inclusive-exclusive runs of True values from a 1D boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            yield start, i
            start = None
    if start is not None:
        yield start, len(mask)


def _detect_reversal_artifacts(data, diff_factor=6, max_run=4):
    """
    Detect short V/inverse-V artifacts per neuron.

    This targets frame-level acquisition/tracking artifacts that jump away from
    the local trace and recover within a few frames. It deliberately avoids
    smoothing and does not classify sustained biological rises/decays as noise.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_neurons, n_time = data.shape
    bad = np.zeros((n_neurons, n_time), dtype=bool)
    if n_time < 3:
        return bad

    scales = _robust_diff_scale(data)
    thresholds = diff_factor * scales

    left_jump = data[:, 1:-1] - data[:, :-2]
    right_jump = data[:, 2:] - data[:, 1:-1]
    threshold_2d = thresholds[:, np.newaxis]
    finite_triplet = (
        np.isfinite(data[:, :-2])
        & np.isfinite(data[:, 1:-1])
        & np.isfinite(data[:, 2:])
    )
    spike = (left_jump > threshold_2d) & (right_jump < -threshold_2d)
    trough = (left_jump < -threshold_2d) & (right_jump > threshold_2d)
    bad[:, 1:-1] |= finite_triplet & (spike | trough)

    # Catch short flat-topped/flat-bottomed bursts that a strict 3-point check
    # cannot see, e.g. [100, 100, 500, 500, 100, 100].
    diff = np.diff(data, axis=1)
    for neuron_idx in range(n_neurons):
        threshold = thresholds[neuron_idx]
        row_diff = diff[neuron_idx]
        large_jump_indices = np.where(np.isfinite(row_diff) & (np.abs(row_diff) > threshold))[0]
        for start_diff_idx in large_jump_indices:
            first_jump = row_diff[start_diff_idx]
            jump_sign = np.sign(first_jump)
            if jump_sign == 0:
                continue

            search_stop = min(row_diff.size, start_diff_idx + max_run + 1)
            for end_diff_idx in range(start_diff_idx + 1, search_stop):
                second_jump = row_diff[end_diff_idx]
                if not np.isfinite(second_jump):
                    continue
                if np.sign(second_jump) != -jump_sign or abs(second_jump) <= threshold:
                    continue

                left_value = data[neuron_idx, start_diff_idx]
                right_value = data[neuron_idx, end_diff_idx + 1]
                if not (np.isfinite(left_value) and np.isfinite(right_value)):
                    break

                recovery_error = abs(left_value - right_value)
                artifact_size = max(abs(first_jump), abs(second_jump))
                if recovery_error <= max(threshold, artifact_size * 0.35):
                    bad[neuron_idx, start_diff_idx + 1 : end_diff_idx + 1] = True
                    break

    return bad


def _restore_input_type(data, template, columns=None, index=None, original_shape=None):
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(data, index=index, columns=columns)
    if original_shape is not None and len(original_shape) == 1:
        return data.flatten()
    return data


def detect_shared_bad_frames(
    intensity_input,
    diff_factor=6,
    shared_threshold=0.35,
    max_run=4,
    return_score=False,
):
    """
    Detect frames with synchronized short artifacts across neurons.

    Parameters:
    - intensity_input: pd.DataFrame or np.ndarray, shaped (neurons, time)
    - diff_factor: multiplier on each neuron's robust diff MAD scale
    - shared_threshold: fraction of neurons that must show a local artifact
    - max_run: maximum short burst length considered for reversal artifacts
    - return_score: if True, also return per-frame shared artifact fractions
    """
    data = np.asarray(
        intensity_input.values if isinstance(intensity_input, pd.DataFrame) else intensity_input,
        dtype=float,
    )
    if data.ndim == 1:
        data = data[np.newaxis, :]

    per_neuron_bad = _detect_reversal_artifacts(
        data, diff_factor=diff_factor, max_run=max_run
    )
    shared_score = per_neuron_bad.mean(axis=0)
    shared_bad_frames = shared_score >= shared_threshold
    if return_score:
        return shared_bad_frames, shared_score, per_neuron_bad
    return shared_bad_frames


def repair_short_artifacts(intensity_input, bad_mask, max_interp_gap=4):
    """
    Replace short marked artifacts with local linear interpolation.

    Long marked runs and edge runs are left unchanged because there is not enough
    local evidence to reconstruct them safely.
    """
    is_df = isinstance(intensity_input, pd.DataFrame)
    if is_df:
        data = intensity_input.values.astype(float).copy()
        columns = intensity_input.columns
        index = intensity_input.index
    else:
        data = np.array(intensity_input, dtype=float, copy=True)
        columns = None
        index = None

    original_shape = data.shape
    if data.ndim == 1:
        data = data[np.newaxis, :]

    mask = np.asarray(bad_mask, dtype=bool)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[np.newaxis, :], data.shape).copy()
    elif mask.shape != data.shape:
        raise ValueError(
            f"bad_mask must have shape {(data.shape[1],)} or {data.shape}, got {mask.shape}"
        )

    for neuron_idx in range(data.shape[0]):
        trace = data[neuron_idx]
        for start, stop in _find_runs(mask[neuron_idx]):
            if stop - start > max_interp_gap:
                continue
            left_idx = start - 1
            right_idx = stop
            if left_idx < 0 or right_idx >= trace.size:
                continue
            if not (np.isfinite(trace[left_idx]) and np.isfinite(trace[right_idx])):
                continue
            trace[start:stop] = np.interp(
                np.arange(start, stop),
                [left_idx, right_idx],
                [trace[left_idx], trace[right_idx]],
            )

    return _restore_input_type(
        data, intensity_input, columns=columns, index=index, original_shape=original_shape
    )


def denoise_frame_artifacts(
    intensity_input,
    diff_factor=6,
    shared_threshold=0.35,
    max_run=4,
    max_interp_gap=4,
    repair_local=True,
    repair_all_neurons_at_shared_frames=True,
    return_info=False,
):
    """
    Denoise short frame-level artifacts without smoothing or time shifting.

    The detector first finds local V/inverse-V reversals per neuron, then scores
    frames by how many neurons show that pattern. Synchronized bad frames are
    repaired by short local interpolation. Optional per-neuron local repair
    handles isolated spikes that are not shared across the population.
    """
    is_df = isinstance(intensity_input, pd.DataFrame)
    if is_df:
        data = intensity_input.values.astype(float).copy()
        columns = intensity_input.columns
        index = intensity_input.index
    else:
        data = np.array(intensity_input, dtype=float, copy=True)
        columns = None
        index = None

    original_shape = data.shape
    if data.ndim == 1:
        data = data[np.newaxis, :]

    shared_frames, shared_score, per_neuron_bad = detect_shared_bad_frames(
        data,
        diff_factor=diff_factor,
        shared_threshold=shared_threshold,
        max_run=max_run,
        return_score=True,
    )

    repair_mask = per_neuron_bad.copy() if repair_local else np.zeros_like(per_neuron_bad)
    if repair_all_neurons_at_shared_frames:
        short_shared_frames = shared_frames.copy()
        for start, stop in _find_runs(shared_frames):
            if stop - start > max_interp_gap:
                short_shared_frames[start:stop] = False
        repair_mask[:, short_shared_frames] = True
    else:
        repair_mask |= per_neuron_bad & shared_frames[np.newaxis, :]

    repaired = repair_short_artifacts(data, repair_mask, max_interp_gap=max_interp_gap)
    if return_info:
        repaired_array = repaired.values if isinstance(repaired, pd.DataFrame) else repaired
        info = {
            "shared_bad_frames": np.where(shared_frames)[0],
            "shared_score": shared_score,
            "per_neuron_bad_mask": per_neuron_bad,
            "repair_mask": repair_mask,
            "n_repaired_points": int(
                np.sum(~np.isclose(np.asarray(repaired_array, dtype=float), data, equal_nan=True))
            ),
        }
        return repaired, info

    return _restore_input_type(
        np.asarray(repaired, dtype=float),
        intensity_input,
        columns=columns,
        index=index,
        original_shape=original_shape,
    )


def _despike_trace(trace, mad_factor=5, tolerance_factor=2, max_run=4):
    """Internal helper to remove short reversible artifacts."""
    n = len(trace)
    if n < 3: return trace

    artifact_mask = _detect_reversal_artifacts(
        trace[np.newaxis, :], diff_factor=mad_factor, max_run=max_run
    )[0]
    if artifact_mask.any():
        trace = repair_short_artifacts(
            trace, artifact_mask, max_interp_gap=max_run
        )
    
    # Compute MAD of diff for local thresholding
    diff = np.diff(trace)
    valid_diff = diff[np.isfinite(diff)]
    if len(valid_diff) == 0: return trace
    
    median_diff = np.median(valid_diff)
    mad = np.median(np.abs(valid_diff - median_diff))
    if mad == 0: 
        # If signal is very flat, use a small fraction of the median intensity as floor
        mad = np.nanmedian(np.abs(trace)) * 0.01 + 1e-6
    
    # We look for both "V" (drop-recovery) and "Inverse-V" (spike-decay)
    # But for step detection errors, "V" is the primary target
    for t in range(1, n - 1):
        # Case: Drop followed by recovery
        if (trace[t-1] - trace[t] > mad_factor * mad and 
            trace[t+1] - trace[t] > mad_factor * mad):
            if abs(trace[t-1] - trace[t+1]) < tolerance_factor * mad:
                trace[t] = (trace[t-1] + trace[t+1]) / 2.0
        # Case: Spike followed by decay (optional but good for stability)
        elif (trace[t] - trace[t-1] > mad_factor * mad and 
              trace[t] - trace[t+1] > mad_factor * mad):
            if abs(trace[t-1] - trace[t+1]) < tolerance_factor * mad:
                trace[t] = (trace[t-1] + trace[t+1]) / 2.0
    return trace

