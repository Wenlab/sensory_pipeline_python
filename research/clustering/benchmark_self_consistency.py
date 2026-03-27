from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score, silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from result_analysis.representation_clustering.latent_cluster import cluster_latent_space
from result_analysis.representation_clustering.verification import (
    build_trial_trajectory_payload,
    compute_cluster_coherence,
    compute_dataset_verification_metrics,
    compute_mean_tensor,
    compute_mean_trajectories,
    compute_neutral_pca_spectrum,
    derive_core_panel,
    fit_neutral_trajectory_space,
    trajectory_distance,
)


@dataclass(frozen=True)
class SelfConsistencyConfig:
    neural_data_path: str
    output_root: str = "results/tmp"
    min_trials_per_stimulus: int = 20
    min_full_trials_per_stimulus: int = 10
    baseline_start: int = 0
    baseline_end: int = 4
    crop_start: int = 5
    crop_end: int = 20
    split_size: int = 5
    repeats: int = 100
    trajectory_components: int = 3
    scoring: str = "gap"


def _candidate_repo_roots() -> list[Path]:
    here = Path(__file__).resolve()
    worktree_root = here.parents[2]
    roots = [worktree_root]
    if worktree_root.parent.name == ".worktrees":
        roots.append(worktree_root.parent.parent)
    return roots


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path

    for root in _candidate_repo_roots():
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve path: {path_str}")


def resolve_output_root(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_candidate_repo_roots()[0] / path).resolve()


def load_neural_dataframe(path_str: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_repo_path(path_str))


def create_self_consistency_config(
    neural_data_path: str,
    output_root: str = "results/tmp",
    min_trials_per_stimulus: int = 20,
    min_full_trials_per_stimulus: int = 10,
    baseline_window: tuple[int, int] = (0, 4),
    crop_window: tuple[int, int] = (5, 20),
    split_size: int = 5,
    repeats: int = 100,
    trajectory_components: int = 3,
    scoring: str = "gap",
) -> dict:
    return asdict(
        SelfConsistencyConfig(
            neural_data_path=neural_data_path,
            output_root=output_root,
            min_trials_per_stimulus=min_trials_per_stimulus,
            min_full_trials_per_stimulus=min_full_trials_per_stimulus,
            baseline_start=baseline_window[0],
            baseline_end=baseline_window[1],
            crop_start=crop_window[0],
            crop_end=crop_window[1],
            split_size=split_size,
            repeats=repeats,
            trajectory_components=trajectory_components,
            scoring=scoring,
        )
    )


def create_method_grid() -> list[dict]:
    configs: list[dict] = []

    def add_config(method: str, config_id: str, **kwargs) -> None:
        configs.append({"config_id": config_id, "method": method, **kwargs})

    add_config("pca", "pca_none", scaling="none", scaling_orientation="stimuluswise")
    add_config("pca", "pca_standard_stimuluswise", scaling="standard", scaling_orientation="stimuluswise")
    add_config("pca", "pca_standard_neuronwise", scaling="standard", scaling_orientation="neuronwise")

    add_config("tca", "tca_none", scaling="none", scaling_orientation="stimuluswise")
    add_config("tca", "tca_standard_stimuluswise", scaling="standard", scaling_orientation="stimuluswise")
    add_config("tca", "tca_standard_neuronwise", scaling="standard", scaling_orientation="neuronwise")
    add_config("tca", "tca_soft", scaling="soft", scaling_orientation="stimuluswise")

    add_config("dpca", "dpca_raw_component_embedding", use_reconstruction=False, scaling_orientation="neuronwise")
    add_config("dpca", "dpca_reconstruction_stimuluswise", use_reconstruction=True, scaling_orientation="stimuluswise")
    add_config("dpca", "dpca_reconstruction_neuronwise", use_reconstruction=True, scaling_orientation="neuronwise")

    return configs


def prepare_self_consistency_dataset(
    neural_data_path: str,
    min_trials_per_stimulus: int = 20,
    min_full_trials_per_stimulus: int = 10,
    baseline_window: tuple[int, int] = (0, 4),
    crop_window: tuple[int, int] = (5, 20),
    split_size: int = 5,
    repeats: int = 100,
    trajectory_components: int = 3,
    random_state: int = 0,
) -> dict:
    neural_df = load_neural_dataframe(neural_data_path)
    core_panel = derive_core_panel(neural_df, min_trials_per_stimulus=min_trials_per_stimulus)
    payload = build_trial_trajectory_payload(
        neural_df,
        core_panel=core_panel,
        baseline_window=baseline_window,
        crop_window=crop_window,
        min_full_trials_per_stimulus=min_full_trials_per_stimulus,
    )
    mean_tensor = compute_mean_tensor(payload["trial_tensor"])
    neutral_pca_spectrum = compute_neutral_pca_spectrum(payload["trial_tensor"])
    neutral_space = fit_neutral_trajectory_space(
        payload["trial_tensor"], n_components=trajectory_components
    )
    verification_metrics = compute_dataset_verification_metrics(
        neutral_space["trajectories"],
        stimulus_names=payload["stimuli"],
        split_size=split_size,
        n_repeats=repeats,
        random_state=random_state,
    )
    stimulus_plot_names = derive_stimulus_plot_names(neural_df, payload["stimuli"])

    return {
        "neural_df": neural_df,
        "core_panel": core_panel,
        "payload": payload,
        "trial_tensor": payload["trial_tensor"],
        "mean_tensor": mean_tensor.astype(np.float64),
        "neutral_pca_spectrum": neutral_pca_spectrum,
        "neutral_space": neutral_space,
        "mean_trajectories": compute_mean_trajectories(neutral_space["trajectories"]),
        "verification_metrics": verification_metrics,
        "stimulus_plot_names": stimulus_plot_names,
        "baseline_window": baseline_window,
        "crop_window": crop_window,
        "split_size": split_size,
        "trajectory_components": trajectory_components,
    }


def run_method_configs(
    configs: list[dict],
    prepared: dict,
    seeds: Iterable[int],
    scoring: str = "gap",
) -> pd.DataFrame:
    rows: list[dict] = []
    for config in configs:
        for seed in seeds:
            row = {
                "config_id": config["config_id"],
                "method": config["method"],
                "seed": seed,
            }
            try:
                row.update(
                    execute_self_consistency_config(
                        config,
                        prepared=prepared,
                        seed=seed,
                        scoring=scoring,
                    )
                )
                row.setdefault("error", None)
            except Exception as exc:  # pragma: no cover - exercised by smoke execution
                row.update(
                    {
                        "split_half_label_stability": np.nan,
                        "trajectory_coherence_margin": np.nan,
                        "dropout_stability": np.nan,
                        "silhouette": np.nan,
                        "best_k": np.nan,
                        "runtime_sec": np.nan,
                        "internal_score": np.nan,
                        "label_signature": None,
                        "error": str(exc),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def execute_self_consistency_config(
    config: dict,
    prepared: dict,
    seed: int = 0,
    scoring: str = "gap",
    split_repeats: int = 5,
    dropout_repeats: int = 5,
    dropout_fraction: float = 0.2,
) -> dict:
    start = time.perf_counter()
    np.random.seed(seed)

    mean_tensor = prepared["mean_tensor"]
    trial_tensor = prepared["trial_tensor"]
    labels, best_k, best_score, _, components = cluster_latent_space(
        mean_tensor,
        tensor_trial=trial_tensor if config["method"] == "dpca" else None,
        method=config["method"],
        n_comp=3,
        n_iterations=5,
        metric="euclidean",
        scoring=scoring,
        scaling=config.get("scaling", "none"),
        scaling_orientation=config.get("scaling_orientation", "stimuluswise"),
        use_reconstruction=config.get("use_reconstruction", True),
        var_cum_threshold=0.9,
    )

    split_half_label_stability = compute_split_half_label_stability(
        config,
        trial_tensor,
        scoring=scoring,
        split_size=prepared["split_size"],
        n_repeats=split_repeats,
        random_state=seed,
    )
    dropout_stability = compute_neuron_dropout_stability(
        config,
        mean_tensor,
        trial_tensor,
        base_labels=labels,
        scoring=scoring,
        n_repeats=dropout_repeats,
        dropout_fraction=dropout_fraction,
        random_state=seed,
    )
    trajectory_coherence_margin = compute_cluster_coherence(
        labels,
        prepared["mean_trajectories"],
    )
    silhouette = np.nan
    if 1 < len(np.unique(labels)) < len(labels):
        silhouette = float(silhouette_score(components, labels))

    runtime_sec = time.perf_counter() - start
    return {
        "split_half_label_stability": split_half_label_stability,
        "trajectory_coherence_margin": trajectory_coherence_margin,
        "dropout_stability": dropout_stability,
        "silhouette": silhouette,
        "best_k": int(best_k),
        "runtime_sec": runtime_sec,
        "internal_score": float(best_score),
        "label_signature": json.dumps([int(label) for label in labels.tolist()]),
    }


def compute_split_half_label_stability(
    config: dict,
    trial_tensor: np.ndarray,
    scoring: str,
    split_size: int = 5,
    n_repeats: int = 5,
    random_state: int = 0,
) -> float:
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    for _ in range(n_repeats):
        half_a_tensor, half_b_tensor, half_a_mean, half_b_mean = sample_split_trial_tensors(
            trial_tensor,
            split_size=split_size,
            rng=rng,
        )
        labels_a, *_ = cluster_latent_space(
            half_a_mean,
            tensor_trial=half_a_tensor if config["method"] == "dpca" else None,
            method=config["method"],
            n_comp=3,
            n_iterations=5,
            metric="euclidean",
            scoring=scoring,
            scaling=config.get("scaling", "none"),
            scaling_orientation=config.get("scaling_orientation", "stimuluswise"),
            use_reconstruction=config.get("use_reconstruction", True),
            var_cum_threshold=0.9,
        )
        labels_b, *_ = cluster_latent_space(
            half_b_mean,
            tensor_trial=half_b_tensor if config["method"] == "dpca" else None,
            method=config["method"],
            n_comp=3,
            n_iterations=5,
            metric="euclidean",
            scoring=scoring,
            scaling=config.get("scaling", "none"),
            scaling_orientation=config.get("scaling_orientation", "stimuluswise"),
            use_reconstruction=config.get("use_reconstruction", True),
            var_cum_threshold=0.9,
        )
        scores.append(float(adjusted_rand_score(labels_a, labels_b)))
    return float(np.mean(scores)) if scores else np.nan


def compute_neuron_dropout_stability(
    config: dict,
    mean_tensor: np.ndarray,
    trial_tensor: np.ndarray,
    base_labels: np.ndarray,
    scoring: str,
    n_repeats: int = 5,
    dropout_fraction: float = 0.2,
    random_state: int = 0,
) -> float:
    rng = np.random.default_rng(random_state)
    n_neurons = mean_tensor.shape[1]
    n_drop = max(1, int(round(n_neurons * dropout_fraction)))
    scores: list[float] = []
    for _ in range(n_repeats):
        drop_indices = rng.choice(n_neurons, size=n_drop, replace=False)
        keep_mask = np.ones(n_neurons, dtype=bool)
        keep_mask[drop_indices] = False
        dropped_mean = mean_tensor[:, keep_mask, :]
        dropped_trial = trial_tensor[:, keep_mask, :, :]
        labels, *_ = cluster_latent_space(
            dropped_mean,
            tensor_trial=dropped_trial if config["method"] == "dpca" else None,
            method=config["method"],
            n_comp=3,
            n_iterations=5,
            metric="euclidean",
            scoring=scoring,
            scaling=config.get("scaling", "none"),
            scaling_orientation=config.get("scaling_orientation", "stimuluswise"),
            use_reconstruction=config.get("use_reconstruction", True),
            var_cum_threshold=0.9,
        )
        scores.append(float(adjusted_rand_score(base_labels, labels)))
    return float(np.mean(scores)) if scores else np.nan


def sample_split_trial_tensors(
    trial_tensor: np.ndarray,
    split_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_stimuli, n_neurons, n_timepoints, _ = trial_tensor.shape
    half_a_trials = np.full((n_stimuli, n_neurons, n_timepoints, split_size), np.nan, dtype=np.float64)
    half_b_trials = np.full((n_stimuli, n_neurons, n_timepoints, split_size), np.nan, dtype=np.float64)
    half_a_mean = np.full((n_stimuli, n_neurons, n_timepoints), np.nan, dtype=np.float64)
    half_b_mean = np.full((n_stimuli, n_neurons, n_timepoints), np.nan, dtype=np.float64)

    for stim_idx in range(n_stimuli):
        valid_trial_indices = np.where(
            ~np.all(np.isnan(trial_tensor[stim_idx]), axis=(0, 1))
        )[0]
        if len(valid_trial_indices) < 2 * split_size:
            raise ValueError(
                f"Stimulus index {stim_idx} does not have {2 * split_size} valid trials."
            )
        chosen = rng.choice(valid_trial_indices, size=2 * split_size, replace=False)
        a_idx = chosen[:split_size]
        b_idx = chosen[split_size:]
        half_a_trials[stim_idx] = trial_tensor[stim_idx][:, :, a_idx]
        half_b_trials[stim_idx] = trial_tensor[stim_idx][:, :, b_idx]
        half_a_mean[stim_idx] = np.nanmean(half_a_trials[stim_idx], axis=2)
        half_b_mean[stim_idx] = np.nanmean(half_b_trials[stim_idx], axis=2)

    return half_a_trials, half_b_trials, half_a_mean, half_b_mean


def aggregate_method_runs(run_results: pd.DataFrame) -> pd.DataFrame:
    if run_results.empty:
        return pd.DataFrame()

    successful = run_results[run_results["error"].isna()].copy()
    if successful.empty:
        return pd.DataFrame(columns=["config_id", "method"])

    summary = (
        successful.groupby(["config_id", "method"], as_index=False)
        .agg(
            split_half_label_stability=("split_half_label_stability", "mean"),
            trajectory_coherence_margin=("trajectory_coherence_margin", "mean"),
            dropout_stability=("dropout_stability", "mean"),
            silhouette=("silhouette", "mean"),
            best_k=("best_k", "mean"),
            runtime_sec=("runtime_sec", "mean"),
            internal_score=("internal_score", "mean"),
            successful_runs=("seed", "nunique"),
        )
    )
    stability = compute_seed_stability(run_results)
    return summary.merge(stability, on="config_id", how="left")


def compute_seed_stability(run_results: pd.DataFrame) -> pd.DataFrame:
    successful = run_results[run_results["error"].isna()].copy()
    rows: list[dict] = []
    for config_id, group in successful.groupby("config_id"):
        label_sets = [json.loads(signature) for signature in group["label_signature"].dropna().tolist()]
        pair_aris: list[float] = []
        for i in range(len(label_sets)):
            for j in range(i + 1, len(label_sets)):
                pair_aris.append(adjusted_rand_score(label_sets[i], label_sets[j]))
        rows.append(
            {
                "config_id": config_id,
                "seed_stability": float(np.mean(pair_aris)) if pair_aris else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_self_consistency_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    method_priority = summary_df["method"].map({"pca": 0, "tca": 1, "dpca": 2}).fillna(9)
    ranked = summary_df.assign(method_priority=method_priority).sort_values(
        by=[
            "split_half_label_stability",
            "trajectory_coherence_margin",
            "seed_stability",
            "dropout_stability",
            "method_priority",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked.drop(columns=["method_priority"])


def derive_stimulus_plot_names(neural_df: pd.DataFrame, stimuli: list[str]) -> list[str]:
    if "stim_name" not in neural_df.columns:
        return list(stimuli)

    mapping: dict[str, str] = {}
    for stimulus in stimuli:
        rows = neural_df.loc[neural_df["stimulus"] == stimulus, "stim_name"].dropna()
        if rows.empty:
            mapping[stimulus] = stimulus
            continue
        unique_names = rows.astype(str).unique().tolist()
        mapping[stimulus] = unique_names[0]

    counts = pd.Series([mapping[stimulus] for stimulus in stimuli]).value_counts()
    labels: list[str] = []
    for stimulus in stimuli:
        label = mapping[stimulus]
        if counts[label] > 1:
            labels.append(f"{label} ({stimulus})")
        else:
            labels.append(label)
    return labels


def write_self_consistency_outputs(
    output_dir: Path | str,
    prepared: dict,
    run_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    config: dict,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    verification_metrics = dict(prepared["verification_metrics"])
    verification_metrics["core_panel"] = prepared["core_panel"]
    verification_metrics["full_trial_counts"] = prepared["payload"]["full_trial_counts"]
    verification_metrics["crop_time_points"] = prepared["payload"]["crop_time_points"]
    verification_metrics["baseline_time_points"] = prepared["payload"]["baseline_time_points"]

    (output_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_path / "dataset_verification_summary.json").write_text(
        json.dumps(_to_jsonable(verification_metrics), indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                key: value
                for key, value in verification_metrics.items()
                if not isinstance(value, (list, dict))
            }
        ]
    ).to_csv(output_path / "dataset_verification_summary.csv", index=False)

    run_results.to_csv(output_path / "method_run_results.csv", index=False)
    summary_df.to_csv(output_path / "method_summary.csv", index=False)
    ranking_df.to_csv(output_path / "method_ranking_summary.csv", index=False)

    summary_lines = [
        "# Self-Consistency Benchmark",
        f"- Retained stimuli: {len(prepared['payload']['stimuli'])}",
        f"- Core panel size: {len(prepared['core_panel'])}",
        f"- Verification pass: {prepared['verification_metrics']['passes_verification']}",
    ]
    if not ranking_df.empty:
        summary_lines.append(f"- Top method: {ranking_df.iloc[0]['config_id']}")
    (output_path / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return output_path


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def plot_phase2_verification_figures(
    trial_trajectories: np.ndarray,
    stimulus_names: list[str],
    output_dir: Path | str,
    split_size: int = 5,
    n_repeats: int = 100,
    random_state: int = 0,
    time_points: list[int] | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    groups, kept_stimuli = _extract_plot_groups(trial_trajectories, stimulus_names, split_size)
    if len(groups) < 2:
        raise ValueError("Need at least two stimuli with enough trials to plot Phase 2 figures.")

    rng = np.random.default_rng(random_state)
    same_distances: list[float] = []
    different_distances: list[float] = []
    distance_matrices: list[np.ndarray] = []
    repeat_same_means: list[float] = []
    repeat_different_means: list[float] = []
    representative_half_a: np.ndarray | None = None
    representative_half_b: np.ndarray | None = None

    for _ in range(n_repeats):
        half_a, half_b = _sample_plot_split_halves(groups, split_size, rng)
        if representative_half_a is None:
            representative_half_a = half_a.copy()
            representative_half_b = half_b.copy()
        distance_matrix = _cross_trajectory_distance_matrix(half_a, half_b)
        distance_matrices.append(distance_matrix)
        same_values = np.diag(distance_matrix)
        different_values = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)]
        same_distances.extend(same_values.tolist())
        different_distances.extend(different_values.tolist())
        repeat_same_means.append(float(np.mean(same_values)))
        repeat_different_means.append(float(np.mean(different_values)))

    same_array = np.asarray(same_distances, dtype=np.float64)
    different_array = np.asarray(different_distances, dtype=np.float64)
    mean_distance_matrix = np.mean(np.stack(distance_matrices, axis=0), axis=0)
    repeat_same_array = np.asarray(repeat_same_means, dtype=np.float64)
    repeat_different_array = np.asarray(repeat_different_means, dtype=np.float64)
    stat = wilcoxon(repeat_same_array, repeat_different_array, alternative="less", zero_method="wilcox")
    ordered_matrix, ordered_labels = _cluster_order_distance_matrix(mean_distance_matrix, kept_stimuli)

    summary = {
        "stimulus_names": ordered_labels,
        "same_mean": float(np.mean(same_array)),
        "same_median": float(np.median(same_array)),
        "same_sem": float(np.std(repeat_same_array, ddof=0) / np.sqrt(len(repeat_same_array))),
        "different_mean": float(np.mean(different_array)),
        "different_median": float(np.median(different_array)),
        "different_sem": float(np.std(repeat_different_array, ddof=0) / np.sqrt(len(repeat_different_array))),
        "p_value": float(stat.pvalue),
        "test_statistic": float(stat.statistic),
        "n_repeats": int(n_repeats),
        "split_size": int(split_size),
    }

    _plot_same_vs_different_bar(
        summary,
        repeat_same_array,
        repeat_different_array,
        output_path / "05_phase2_same_vs_different_bar.png",
    )
    _plot_distance_matrix_heatmap(
        ordered_matrix,
        ordered_labels,
        output_path / "06_phase2_distance_matrix_heatmap.png",
    )
    if representative_half_a is not None and representative_half_b is not None:
        order_lookup = {label: idx for idx, label in enumerate(kept_stimuli)}
        order = [order_lookup[label] for label in ordered_labels]
        _plot_split_half_3d_trajectories(
            representative_half_a[order],
            representative_half_b[order],
            ordered_labels,
            output_path / "07_phase2_split_half_3d.png",
        )
        _plot_split_half_3d_html(
            representative_half_a[order],
            representative_half_b[order],
            ordered_labels,
            output_path / "07_phase2_split_half_3d.html",
        )
        _plot_split_half_centers_3d(
            representative_half_a[order],
            representative_half_b[order],
            ordered_labels,
            output_path / "08_phase2_split_half_centers_3d.png",
        )
        key_indices, key_labels = _select_key_time_indices(
            n_timepoints=representative_half_a.shape[1],
            time_points=time_points,
        )
        _plot_split_half_key_timepoints_3d(
            representative_half_a[order],
            representative_half_b[order],
            ordered_labels,
            key_indices,
            key_labels,
            output_path / "09_phase2_split_half_key_timepoints_3d.png",
        )
    (output_path / "phase2_figure_summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def plot_neutral_pca_scree(
    explained_variance_ratio: np.ndarray,
    output_dir: Path | str,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ratios = np.asarray(explained_variance_ratio, dtype=np.float64)
    components = np.arange(1, len(ratios) + 1)
    cumulative = np.cumsum(ratios)
    summary = {
        f"pc{idx + 1}": float(value) for idx, value in enumerate(ratios)
    }
    summary["cumulative_top_3"] = float(cumulative[min(2, len(cumulative) - 1)])
    summary["cumulative_top_5"] = float(cumulative[min(4, len(cumulative) - 1)])
    summary["cumulative_total"] = float(cumulative[-1])

    fig, ax1 = plt.subplots(figsize=(6.0, 4.2))
    ax1.bar(components, ratios, color="#4C78A8", edgecolor="black", linewidth=0.8)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_title(
        f"Neutral PCA scree plot (3D captures {summary['cumulative_top_3']:.1%} variance)"
    )
    ax1.set_xticks(components)

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="#F58518", marker="o", linewidth=1.6)
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, max(1.0, cumulative[-1] * 1.05))

    for idx, label in [(3, "3D"), (5, "5D")]:
        if idx <= len(cumulative):
            x_pos = idx
            y_pos = cumulative[idx - 1]
            ax2.scatter([x_pos], [y_pos], color="#C0392B", s=30, zorder=4)
            ax2.annotate(
                f"{label}: {y_pos:.1%}",
                xy=(x_pos, y_pos),
                xytext=(8, -14 if idx == 3 else 10),
                textcoords="offset points",
                fontsize=9,
                color="#7F1D1D",
                arrowprops={"arrowstyle": "-", "color": "#7F1D1D", "lw": 0.8},
            )

    fig.tight_layout()
    fig.savefig(output_path / "04_neutral_pca_scree.png", dpi=200)
    plt.close(fig)

    (output_path / "neutral_pca_summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def _extract_plot_groups(
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


def _sample_plot_split_halves(
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


def _cross_trajectory_distance_matrix(half_a: np.ndarray, half_b: np.ndarray) -> np.ndarray:
    distance_matrix = np.zeros((len(half_a), len(half_b)), dtype=np.float64)
    for i in range(len(half_a)):
        for j in range(len(half_b)):
            distance_matrix[i, j] = trajectory_distance(half_a[i], half_b[j])
    return distance_matrix


def _cluster_order_distance_matrix(
    distance_matrix: np.ndarray,
    stimulus_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if distance_matrix.shape[0] <= 2:
        return distance_matrix, stimulus_names

    symmetric = (distance_matrix + distance_matrix.T) / 2.0
    np.fill_diagonal(symmetric, 0.0)
    condensed = squareform(symmetric, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    return distance_matrix[np.ix_(order, order)], [stimulus_names[idx] for idx in order]


def _select_key_time_indices(
    n_timepoints: int,
    time_points: list[int] | None,
) -> tuple[list[int], list[str]]:
    if n_timepoints < 1:
        raise ValueError("n_timepoints must be at least 1.")

    if time_points is not None and len(time_points) != n_timepoints:
        raise ValueError("time_points length must match the trajectory time axis.")

    if n_timepoints <= 4:
        indices = list(range(n_timepoints))
    else:
        fractions = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
        indices = sorted({int(round(fraction * (n_timepoints - 1))) for fraction in fractions})
        while len(indices) < 4:
            for candidate in range(n_timepoints):
                if candidate not in indices:
                    indices.append(candidate)
                if len(indices) == 4:
                    break
        indices = sorted(indices[:4])

    if time_points is None:
        labels = [f"t{idx + 1}" for idx in indices]
    else:
        labels = [str(time_points[idx]) for idx in indices]
    return indices, labels


def _plot_same_vs_different_bar(
    summary: dict,
    repeat_same: np.ndarray,
    repeat_different: np.ndarray,
    output_path: Path,
) -> None:
    labels = ["Same stimulus", "Different stimulus"]
    means = [summary["same_mean"], summary["different_mean"]]
    sems = [summary["same_sem"], summary["different_sem"]]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    bars = ax.bar(
        labels,
        means,
        yerr=sems,
        capsize=8,
        color=["#4C78A8", "#F58518"],
        edgecolor="black",
        linewidth=1.0,
    )
    ax.set_ylabel("Mean trajectory distance")
    ax.set_title("Phase 2: same-stimulus halves are closer")

    rng = np.random.default_rng(0)
    for x_pos, values, color in zip(
        [0, 1],
        [repeat_same, repeat_different],
        ["#2F5D8A", "#B45A10"],
    ):
        jitter = rng.normal(loc=0.0, scale=0.035, size=len(values))
        ax.scatter(
            np.full(len(values), x_pos) + jitter,
            values,
            s=18,
            color=color,
            alpha=0.7,
            zorder=3,
        )

    upper = max(means[i] + sems[i] for i in range(2))
    line_y = upper * 1.10 if upper > 0 else 0.1
    text_y = upper * 1.17 if upper > 0 else 0.12
    ax.plot([0, 0, 1, 1], [line_y * 0.98, line_y, line_y, line_y * 0.98], color="black", lw=1.2)
    ax.text(0.5, text_y, _format_p_value(summary["p_value"]), ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, text_y * 1.12 if text_y > 0 else 1.0)

    for bar, mean_value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_distance_matrix_heatmap(
    distance_matrix: np.ndarray,
    stimulus_names: list[str],
    output_path: Path,
) -> None:
    fig_width = max(6.5, len(stimulus_names) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.82))
    image = ax.imshow(distance_matrix, cmap="magma_r", aspect="equal")
    ax.set_title("Phase 2: split-half distance matrix")
    ax.set_xlabel("Half B stimulus")
    ax.set_ylabel("Half A stimulus")
    ax.set_xticks(np.arange(len(stimulus_names)))
    ax.set_xticklabels(stimulus_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(stimulus_names)))
    ax.set_yticklabels(stimulus_names, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(stimulus_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(stimulus_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for idx in range(len(stimulus_names)):
        rect = plt.Rectangle((idx - 0.5, idx - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=0.7)
        ax.add_patch(rect)
    fig.colorbar(image, ax=ax, label="Mean trajectory distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_split_half_3d_trajectories(
    half_a: np.ndarray,
    half_b: np.ndarray,
    stimulus_names: list[str],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, len(stimulus_names)))

    for idx, (traj_a, traj_b, color) in enumerate(zip(half_a, half_b, colors)):
        ax.plot(
            traj_a[:, 0],
            traj_a[:, 1],
            traj_a[:, 2],
            color=color,
            linewidth=1.8,
            alpha=0.9,
        )
        ax.plot(
            traj_b[:, 0],
            traj_b[:, 1],
            traj_b[:, 2],
            color=color,
            linewidth=1.6,
            linestyle="--",
            alpha=0.75,
        )
        center_a = np.nanmean(traj_a, axis=0)
        center_b = np.nanmean(traj_b, axis=0)
        ax.scatter(center_a[0], center_a[1], center_a[2], color=color, s=18, alpha=0.95)
        ax.scatter(center_b[0], center_b[1], center_b[2], color=color, s=18, alpha=0.95, marker="^")
        ax.plot(
            [center_a[0], center_b[0]],
            [center_a[1], center_b[1]],
            [center_a[2], center_b[2]],
            color=color,
            linewidth=0.9,
            alpha=0.5,
        )

    ax.set_title("Phase 2: representative split-halves in 3D neutral PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=24, azim=-55)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", label="Half A"),
        Line2D([0], [0], color="black", lw=1.6, linestyle="--", label="Half B"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_split_half_3d_html(
    half_a: np.ndarray,
    half_b: np.ndarray,
    stimulus_names: list[str],
    output_path: Path,
) -> None:
    colors = plt.cm.tab20(np.linspace(0, 1, len(stimulus_names)))
    figure = go.Figure()

    for traj_a, traj_b, color, stimulus_name in zip(half_a, half_b, colors, stimulus_names):
        rgb = tuple(int(round(channel * 255)) for channel in color[:3])
        color_text = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        figure.add_trace(
            go.Scatter3d(
                x=traj_a[:, 0],
                y=traj_a[:, 1],
                z=traj_a[:, 2],
                mode="lines",
                line={"color": color_text, "width": 5},
                name=f"{stimulus_name} | Half A",
                hovertemplate=f"{stimulus_name}<br>Half A<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<br>PC3=%{{z:.2f}}<extra></extra>",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=traj_b[:, 0],
                y=traj_b[:, 1],
                z=traj_b[:, 2],
                mode="lines",
                line={"color": color_text, "width": 4, "dash": "dash"},
                name=f"{stimulus_name} | Half B",
                hovertemplate=f"{stimulus_name}<br>Half B<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<br>PC3=%{{z:.2f}}<extra></extra>",
                showlegend=False,
            )
        )

    figure.update_layout(
        title="Phase 2: representative split-halves in 3D neutral PCA",
        scene={
            "xaxis_title": "PC1",
            "yaxis_title": "PC2",
            "zaxis_title": "PC3",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        template="plotly_white",
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def _plot_split_half_centers_3d(
    half_a: np.ndarray,
    half_b: np.ndarray,
    stimulus_names: list[str],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, len(stimulus_names)))

    for traj_a, traj_b, color in zip(half_a, half_b, colors):
        center_a = np.nanmean(traj_a, axis=0)
        center_b = np.nanmean(traj_b, axis=0)
        ax.scatter(center_a[0], center_a[1], center_a[2], color=color, s=42, alpha=0.95)
        ax.scatter(center_b[0], center_b[1], center_b[2], color=color, s=46, alpha=0.95, marker="^")
        ax.plot(
            [center_a[0], center_b[0]],
            [center_a[1], center_b[1]],
            [center_a[2], center_b[2]],
            color=color,
            linewidth=1.2,
            alpha=0.65,
        )

    ax.set_title("Phase 2: split-half centers in 3D neutral PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=24, azim=-55)

    legend_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=6, label="Half A center"),
        Line2D([0], [0], color="black", marker="^", linestyle="None", markersize=7, label="Half B center"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_split_half_key_timepoints_3d(
    half_a: np.ndarray,
    half_b: np.ndarray,
    stimulus_names: list[str],
    key_indices: list[int],
    key_labels: list[str],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, len(stimulus_names)))
    marker_sizes = np.linspace(18, 42, len(key_indices))

    for traj_a, traj_b, color in zip(half_a, half_b, colors):
        key_a = traj_a[key_indices]
        key_b = traj_b[key_indices]
        ax.plot(key_a[:, 0], key_a[:, 1], key_a[:, 2], color=color, linewidth=1.5, alpha=0.9)
        ax.plot(key_b[:, 0], key_b[:, 1], key_b[:, 2], color=color, linewidth=1.3, linestyle="--", alpha=0.8)
        for point_idx, size in enumerate(marker_sizes):
            ax.scatter(key_a[point_idx, 0], key_a[point_idx, 1], key_a[point_idx, 2], color=color, s=size, alpha=0.95)
            ax.scatter(
                key_b[point_idx, 0],
                key_b[point_idx, 1],
                key_b[point_idx, 2],
                color=color,
                s=size,
                alpha=0.95,
                marker="^",
            )

    ax.set_title("Phase 2: key timepoints in 3D neutral PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=24, azim=-55)

    half_legend = [
        Line2D([0], [0], color="black", marker="o", linestyle="-", markersize=5, label="Half A"),
        Line2D([0], [0], color="black", marker="^", linestyle="--", markersize=6, label="Half B"),
    ]
    time_legend = [
        Line2D([0], [0], color="gray", marker="o", linestyle="None", markersize=4 + idx * 1.4, label=f"t={label}")
        for idx, label in enumerate(key_labels)
    ]
    ax.legend(handles=half_legend + time_legend, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _format_p_value(p_value: float) -> str:
    if p_value < 1e-3:
        return "p < 0.001"
    return f"p = {p_value:.3g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark clustering methods by trial self-consistency without external labels."
    )
    parser.add_argument("--data", required=True, help="Path to the neural parquet dataset.")
    parser.add_argument("--output-root", default="results/tmp", help="Directory for outputs.")
    parser.add_argument("--scoring", default="gap", choices=["gap", "silhouette"])
    parser.add_argument("--min-trials-per-stimulus", type=int, default=20)
    parser.add_argument("--min-full-trials-per-stimulus", type=int, default=10)
    parser.add_argument("--baseline-start", type=int, default=0)
    parser.add_argument("--baseline-end", type=int, default=4)
    parser.add_argument("--crop-start", type=int, default=5)
    parser.add_argument("--crop-end", type=int, default=20)
    parser.add_argument("--split-size", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trajectory-components", type=int, default=3)
    args = parser.parse_args()

    config = create_self_consistency_config(
        neural_data_path=args.data,
        output_root=args.output_root,
        min_trials_per_stimulus=args.min_trials_per_stimulus,
        min_full_trials_per_stimulus=args.min_full_trials_per_stimulus,
        baseline_window=(args.baseline_start, args.baseline_end),
        crop_window=(args.crop_start, args.crop_end),
        split_size=args.split_size,
        repeats=args.repeats,
        trajectory_components=args.trajectory_components,
        scoring=args.scoring,
    )
    prepared = prepare_self_consistency_dataset(
        neural_data_path=args.data,
        min_trials_per_stimulus=args.min_trials_per_stimulus,
        min_full_trials_per_stimulus=args.min_full_trials_per_stimulus,
        baseline_window=(args.baseline_start, args.baseline_end),
        crop_window=(args.crop_start, args.crop_end),
        split_size=args.split_size,
        repeats=args.repeats,
        trajectory_components=args.trajectory_components,
        random_state=0,
    )

    run_results = pd.DataFrame()
    summary_df = pd.DataFrame()
    ranking_df = pd.DataFrame()
    if prepared["verification_metrics"]["passes_verification"]:
        run_results = run_method_configs(
            create_method_grid(),
            prepared=prepared,
            seeds=[0, 1, 2],
            scoring=args.scoring,
        )
        summary_df = aggregate_method_runs(run_results)
        ranking_df = rank_self_consistency_results(summary_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_output_root(args.output_root) / f"{timestamp}_self_consistency_benchmark"
    write_self_consistency_outputs(
        output_dir,
        prepared=prepared,
        run_results=run_results,
        summary_df=summary_df,
        ranking_df=ranking_df,
        config=config,
    )
    plot_neutral_pca_scree(
        explained_variance_ratio=prepared["neutral_pca_spectrum"]["explained_variance_ratio"],
        output_dir=output_dir,
    )
    plot_phase2_verification_figures(
        trial_trajectories=prepared["neutral_space"]["trajectories"],
        stimulus_names=prepared["stimulus_plot_names"],
        output_dir=output_dir,
        split_size=args.split_size,
        n_repeats=args.repeats,
        random_state=0,
        time_points=prepared["payload"]["crop_time_points"],
    )

    print(f"Self-consistency benchmark completed. Output directory: {output_dir}")
    print(json.dumps(prepared["verification_metrics"], indent=2))
    if not ranking_df.empty:
        print(ranking_df[["rank", "config_id", "method"]].to_string(index=False))


if __name__ == "__main__":
    main()
