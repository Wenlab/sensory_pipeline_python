from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def derive_trial_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable trial identifier based on worm, segment, and date."""
    required = {"worm_key", "segment_index", "date"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns required for trial IDs: {missing_text}")

    enriched = df.copy()
    enriched["trial_id"] = (
        enriched["worm_key"].astype(str)
        + "_"
        + enriched["segment_index"].astype(str)
        + "_"
        + enriched["date"].astype(str)
    )
    return enriched


def derive_core_panel(
    df: pd.DataFrame,
    min_trials_per_stimulus: int = 20,
) -> list[str]:
    """Return neurons with enough trial support in every observed stimulus."""
    enriched = derive_trial_ids(df)
    trial_counts = (
        enriched.groupby(["neuron", "stimulus"])["trial_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    keep_mask = trial_counts.min(axis=1) >= min_trials_per_stimulus
    return sorted(trial_counts.index[keep_mask].tolist())


def build_trial_trajectory_payload(
    df: pd.DataFrame,
    core_panel: Iterable[str],
    baseline_window: tuple[int, int] = (0, 4),
    crop_window: tuple[int, int] = (5, 20),
    min_full_trials_per_stimulus: int = 10,
) -> dict:
    """Build a fully aligned trial tensor restricted to a core neuron panel."""
    core_panel = sorted(list(core_panel))
    if not core_panel:
        raise ValueError("core_panel must contain at least one neuron.")

    enriched = derive_trial_ids(df)
    grouped = (
        enriched.groupby(["stimulus", "trial_id", "neuron", "time_point"], as_index=False)[
            "delta_F_over_F0"
        ]
        .mean()
    )
    grouped = grouped[grouped["neuron"].isin(core_panel)].copy()

    baseline_times = sorted(
        grouped.loc[
            grouped["time_point"].between(baseline_window[0], baseline_window[1]),
            "time_point",
        ].unique()
    )
    crop_times = sorted(
        grouped.loc[
            grouped["time_point"].between(crop_window[0], crop_window[1]),
            "time_point",
        ].unique()
    )
    if not baseline_times:
        raise ValueError("No baseline time points found inside baseline_window.")
    if not crop_times:
        raise ValueError("No crop time points found inside crop_window.")

    required_times = sorted(set(baseline_times) | set(crop_times))
    windowed = grouped[grouped["time_point"].isin(required_times)].copy()

    windowed_non_null = windowed.dropna(subset=["delta_F_over_F0"])
    completeness = (
        windowed_non_null.groupby(["stimulus", "trial_id", "neuron"])["time_point"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=core_panel, fill_value=0)
    )
    valid_rows = completeness.index[
        (completeness == len(required_times)).all(axis=1)
    ]

    trial_ids_by_stimulus: dict[str, list[str]] = {}
    for stimulus, trial_id in valid_rows.tolist():
        trial_ids_by_stimulus.setdefault(stimulus, []).append(trial_id)

    kept_stimuli = sorted(
        stimulus
        for stimulus, trial_ids in trial_ids_by_stimulus.items()
        if len(trial_ids) >= min_full_trials_per_stimulus
    )
    if not kept_stimuli:
        raise ValueError("No stimuli retained after full-trial gating.")

    kept_trial_ids = {
        stimulus: sorted(trial_ids_by_stimulus[stimulus])
        for stimulus in kept_stimuli
    }
    max_trials = max(len(trial_ids) for trial_ids in kept_trial_ids.values())
    trial_tensor = np.full(
        (len(kept_stimuli), len(core_panel), len(crop_times), max_trials),
        np.nan,
        dtype=np.float64,
    )

    value_lookup = {
        (row.stimulus, row.trial_id, row.neuron, row.time_point): float(row.delta_F_over_F0)
        for row in windowed_non_null.itertuples(index=False)
    }

    for stim_idx, stimulus in enumerate(kept_stimuli):
        for trial_slot, trial_id in enumerate(kept_trial_ids[stimulus]):
            for neuron_idx, neuron in enumerate(core_panel):
                baseline_values = np.array(
                    [value_lookup[(stimulus, trial_id, neuron, time_point)] for time_point in baseline_times],
                    dtype=np.float64,
                )
                crop_values = np.array(
                    [value_lookup[(stimulus, trial_id, neuron, time_point)] for time_point in crop_times],
                    dtype=np.float64,
                )
                trial_tensor[stim_idx, neuron_idx, :, trial_slot] = crop_values - baseline_values.mean()

    return {
        "stimuli": kept_stimuli,
        "core_panel": core_panel,
        "baseline_time_points": baseline_times,
        "crop_time_points": crop_times,
        "trial_ids_by_stimulus": kept_trial_ids,
        "trial_tensor": trial_tensor,
        "full_trial_counts": {stimulus: len(trial_ids) for stimulus, trial_ids in kept_trial_ids.items()},
    }


def compute_mean_tensor(trial_tensor: np.ndarray) -> np.ndarray:
    """Average trial tensor over the trial axis while ignoring NaN padding."""
    return np.nanmean(trial_tensor, axis=3)


def compute_neutral_pca_spectrum(trial_tensor: np.ndarray) -> dict:
    """Fit a full PCA spectrum across all valid trial-time observations."""
    if trial_tensor.ndim != 4:
        raise ValueError("trial_tensor must have shape (S, N, T, R).")

    trial_major = np.transpose(trial_tensor, (0, 3, 2, 1))  # (S, R, T, N)
    valid_mask = ~np.all(np.isnan(trial_major), axis=(2, 3))
    valid_trajectories = trial_major[valid_mask]
    if len(valid_trajectories) == 0:
        raise ValueError("No valid trial trajectories available for PCA fitting.")

    flattened = valid_trajectories.reshape(-1, valid_trajectories.shape[-1])
    scaler = StandardScaler()
    standardized = scaler.fit_transform(flattened)
    actual_components = min(standardized.shape[0], standardized.shape[1])
    if actual_components < 1:
        raise ValueError("Neutral PCA spectrum requires at least one PCA component.")

    pca = PCA(n_components=actual_components)
    pca.fit(standardized)
    return {
        "scaler": scaler,
        "pca": pca,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components": actual_components,
    }


def fit_neutral_trajectory_space(
    trial_tensor: np.ndarray,
    n_components: int = 3,
) -> dict:
    """Fit a shared PCA trajectory space across all valid trial-time observations."""
    if trial_tensor.ndim != 4:
        raise ValueError("trial_tensor must have shape (S, N, T, R).")

    trial_major = np.transpose(trial_tensor, (0, 3, 2, 1))  # (S, R, T, N)
    valid_mask = ~np.all(np.isnan(trial_major), axis=(2, 3))
    valid_trajectories = trial_major[valid_mask]
    if len(valid_trajectories) == 0:
        raise ValueError("No valid trial trajectories available for PCA fitting.")

    flattened = valid_trajectories.reshape(-1, valid_trajectories.shape[-1])
    scaler = StandardScaler()
    standardized = scaler.fit_transform(flattened)
    actual_components = min(n_components, standardized.shape[0], standardized.shape[1])
    if actual_components < 1:
        raise ValueError("Neutral trajectory space requires at least one PCA component.")

    pca = PCA(n_components=actual_components)
    transformed = pca.fit_transform(standardized).reshape(
        valid_trajectories.shape[0],
        valid_trajectories.shape[1],
        actual_components,
    )

    trajectory_space = np.full(
        (trial_major.shape[0], trial_major.shape[1], trial_major.shape[2], actual_components),
        np.nan,
        dtype=np.float64,
    )
    trajectory_space[valid_mask] = transformed

    return {
        "scaler": scaler,
        "pca": pca,
        "trajectories": trajectory_space,
        "valid_mask": valid_mask,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def trajectory_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """Compute a time-aligned mean Euclidean trajectory distance."""
    if traj_a.shape != traj_b.shape:
        raise ValueError("Trajectory shapes must match.")

    valid_time_mask = ~(
        np.isnan(traj_a).any(axis=1) | np.isnan(traj_b).any(axis=1)
    )
    if not np.any(valid_time_mask):
        return math.nan

    distances = np.linalg.norm(
        traj_a[valid_time_mask] - traj_b[valid_time_mask],
        axis=1,
    )
    return float(np.mean(distances))


def compute_dataset_verification_metrics(
    trial_trajectories: np.ndarray,
    stimulus_names: list[str],
    split_size: int = 5,
    n_repeats: int = 100,
    random_state: int | None = None,
) -> dict:
    """Estimate trial reproducibility metrics and compare them to a null baseline."""
    groups, kept_stimulus_names = _extract_valid_groups(trial_trajectories, stimulus_names, split_size)
    if len(groups) < 2:
        raise ValueError("Need at least two stimuli with enough trials for verification.")

    flat_trials = np.concatenate(groups, axis=0)
    trial_counts = [len(group) for group in groups]
    rng = np.random.default_rng(random_state)

    true_retrieval: list[float] = []
    true_margin: list[float] = []
    true_rdm: list[float] = []
    null_retrieval: list[float] = []
    null_margin: list[float] = []
    null_rdm: list[float] = []

    for _ in range(n_repeats):
        half_a, half_b = _sample_split_halves(groups, split_size, rng)
        true_retrieval.append(_retrieval_accuracy(half_a, half_b))
        true_margin.append(_distance_margin(half_a, half_b))
        true_rdm.append(_rdm_correlation(half_a, half_b))

        shuffled = flat_trials[rng.permutation(len(flat_trials))]
        shuffled_groups = _reshape_groups(shuffled, trial_counts)
        null_a, null_b = _sample_split_halves(shuffled_groups, split_size, rng)
        null_retrieval.append(_retrieval_accuracy(null_a, null_b))
        null_margin.append(_distance_margin(null_a, null_b))
        null_rdm.append(_rdm_correlation(null_a, null_b))

    retrieval_accuracy = float(np.mean(true_retrieval))
    distance_margin = float(np.mean(true_margin))
    rdm_correlation = float(np.mean(true_rdm))
    null_retrieval_mean = float(np.mean(null_retrieval))
    null_distance_margin_mean = float(np.mean(null_margin))
    null_rdm_correlation_mean = float(np.mean(null_rdm))
    null_retrieval_std = float(np.std(null_retrieval, ddof=0))
    null_distance_margin_std = float(np.std(null_margin, ddof=0))
    null_rdm_correlation_std = float(np.std(null_rdm, ddof=0))
    retrieval_z = _z_score(retrieval_accuracy, null_retrieval_mean, null_retrieval_std)
    distance_margin_z = _z_score(distance_margin, null_distance_margin_mean, null_distance_margin_std)
    rdm_correlation_z = _z_score(rdm_correlation, null_rdm_correlation_mean, null_rdm_correlation_std)

    return {
        "stimulus_names": kept_stimulus_names,
        "retrieval_accuracy": retrieval_accuracy,
        "null_retrieval_mean": null_retrieval_mean,
        "null_retrieval_std": null_retrieval_std,
        "retrieval_z": retrieval_z,
        "distance_margin": distance_margin,
        "null_distance_margin_mean": null_distance_margin_mean,
        "null_distance_margin_std": null_distance_margin_std,
        "distance_margin_z": distance_margin_z,
        "rdm_correlation": rdm_correlation,
        "null_rdm_correlation_mean": null_rdm_correlation_mean,
        "null_rdm_correlation_std": null_rdm_correlation_std,
        "rdm_correlation_z": rdm_correlation_z,
        "passes_verification": bool(
            retrieval_z >= 2.0 and distance_margin_z >= 2.0 and rdm_correlation_z >= 2.0
        ),
    }


def compute_mean_trajectories(trial_trajectories: np.ndarray) -> np.ndarray:
    """Average trajectories over the trial axis while ignoring NaN padding."""
    return np.nanmean(trial_trajectories, axis=1)


def compute_cluster_coherence(labels: np.ndarray, mean_trajectories: np.ndarray) -> float:
    """Return between-minus-within trajectory distance for a clustering."""
    distance_matrix = _trajectory_distance_matrix(mean_trajectories)
    within: list[float] = []
    between: list[float] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within.append(distance_matrix[i, j])
            else:
                between.append(distance_matrix[i, j])
    if not within or not between:
        return math.nan
    return float(np.mean(between) - np.mean(within))


def sample_split_mean_tensors(
    trial_tensor: np.ndarray,
    split_size: int = 5,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample two balanced mean tensors from a NaN-padded trial tensor."""
    rng = np.random.default_rng(random_state)
    mean_a = np.full(trial_tensor.shape[:3], np.nan, dtype=np.float64)
    mean_b = np.full(trial_tensor.shape[:3], np.nan, dtype=np.float64)

    for stim_idx in range(trial_tensor.shape[0]):
        valid_trial_indices = np.where(
            ~np.all(np.isnan(trial_tensor[stim_idx]), axis=(0, 1))
        )[0]
        if len(valid_trial_indices) < 2 * split_size:
            raise ValueError(
                f"Stimulus index {stim_idx} does not have {2 * split_size} valid trials."
            )
        chosen = rng.choice(valid_trial_indices, size=2 * split_size, replace=False)
        half_a_idx = chosen[:split_size]
        half_b_idx = chosen[split_size:]
        mean_a[stim_idx] = np.nanmean(trial_tensor[stim_idx, :, :, half_a_idx], axis=2)
        mean_b[stim_idx] = np.nanmean(trial_tensor[stim_idx, :, :, half_b_idx], axis=2)

    return mean_a, mean_b


def _extract_valid_groups(
    trial_trajectories: np.ndarray,
    stimulus_names: list[str],
    split_size: int,
) -> tuple[list[np.ndarray], list[str]]:
    groups: list[np.ndarray] = []
    kept_names: list[str] = []
    for stimulus_idx, stimulus_name in enumerate(stimulus_names):
        stimulus_trials = trial_trajectories[stimulus_idx]
        valid_mask = ~np.all(np.isnan(stimulus_trials), axis=(1, 2))
        valid_trials = stimulus_trials[valid_mask]
        if len(valid_trials) >= 2 * split_size:
            groups.append(valid_trials)
            kept_names.append(stimulus_name)
    return groups, kept_names


def _sample_split_halves(
    groups: list[np.ndarray],
    split_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    half_a: list[np.ndarray] = []
    half_b: list[np.ndarray] = []
    for group in groups:
        chosen = rng.choice(len(group), size=2 * split_size, replace=False)
        half_a.append(np.mean(group[chosen[:split_size]], axis=0))
        half_b.append(np.mean(group[chosen[split_size:]], axis=0))
    return np.stack(half_a), np.stack(half_b)


def _reshape_groups(flat_trials: np.ndarray, counts: list[int]) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    start = 0
    for count in counts:
        groups.append(flat_trials[start : start + count])
        start += count
    return groups


def _trajectory_distance_matrix(mean_trajectories: np.ndarray) -> np.ndarray:
    n_items = mean_trajectories.shape[0]
    distance_matrix = np.zeros((n_items, n_items), dtype=np.float64)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            distance = trajectory_distance(mean_trajectories[i], mean_trajectories[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    return distance_matrix


def _retrieval_accuracy(half_a: np.ndarray, half_b: np.ndarray) -> float:
    distance_matrix = np.zeros((len(half_a), len(half_b)), dtype=np.float64)
    for i in range(len(half_a)):
        for j in range(len(half_b)):
            distance_matrix[i, j] = trajectory_distance(half_a[i], half_b[j])
    predictions = np.argmin(distance_matrix, axis=1)
    truth = np.arange(len(half_a))
    return float(np.mean(predictions == truth))


def _distance_margin(half_a: np.ndarray, half_b: np.ndarray) -> float:
    distance_matrix = np.zeros((len(half_a), len(half_b)), dtype=np.float64)
    for i in range(len(half_a)):
        for j in range(len(half_b)):
            distance_matrix[i, j] = trajectory_distance(half_a[i], half_b[j])
    same = np.diag(distance_matrix)
    different = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)]
    return float(np.mean(different) - np.mean(same))


def _rdm_correlation(half_a: np.ndarray, half_b: np.ndarray) -> float:
    rdm_a = _trajectory_distance_matrix(half_a)
    rdm_b = _trajectory_distance_matrix(half_b)
    tri_a = rdm_a[np.triu_indices(len(half_a), k=1)]
    tri_b = rdm_b[np.triu_indices(len(half_b), k=1)]
    if len(tri_a) < 2:
        return 1.0
    correlation, _ = spearmanr(tri_a, tri_b)
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def _z_score(value: float, mean: float, std: float) -> float:
    if std == 0:
        if value > mean:
            return float("inf")
        return 0.0
    return float((value - mean) / std)
