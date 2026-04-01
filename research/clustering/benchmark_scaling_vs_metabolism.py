from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from result_analysis.representation_clustering.bacteria_metabolism_clustering import (
    build_reference_clustering,
    load_metabolism_data,
)
from result_analysis.representation_clustering.latent_cluster import cluster_latent_space
from result_analysis.representation_clustering.tensor_prep import (
    prepare_chemo_tensor,
    prepare_chemo_trial_tensor,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    neural_data_path: str
    metabolism_path: str
    output_root: str = "results/tmp"
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


def load_neural_dataframe(path_str: str) -> pd.DataFrame:
    path = resolve_repo_path(path_str)
    return pd.read_parquet(path)


def resolve_output_root(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_candidate_repo_roots()[0] / path).resolve()


def normalize_neural_samples(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "sample_id" in normalized.columns:
        return normalized

    if "sample" in normalized.columns:
        normalized["sample_id"] = normalized["sample"].astype(str)
    elif "stim_name" in normalized.columns:
        normalized["sample_id"] = (
            normalized["stim_name"]
            .astype(str)
            .str.split()
            .str[0]
        )
    elif "stimulus" in normalized.columns:
        normalized["sample_id"] = normalized["stimulus"].astype(str)
    else:
        normalized["sample_id"] = pd.Series(dtype="object")

    if "stimulus" in normalized.columns:
        normalized["original_stimulus"] = normalized["stimulus"]
        normalized["stimulus"] = normalized["sample_id"]

    return normalized


def infer_sample_list(df: pd.DataFrame, candidate_columns: Iterable[str] | None = None) -> list[str]:
    columns = list(candidate_columns or ("sample_id", "sample", "bacteria", "sample_list", "stimulus"))
    for column in columns:
        if column in df.columns:
            values = df[column].dropna().tolist()
            return list(dict.fromkeys(values))
    return []


def load_metabolism_reference(
    metabolism_path: str,
    sample_list: list[str] | None,
    scoring: str = "gap",
) -> dict:
    metabolism_df = load_metabolism_data(resolve_repo_path(metabolism_path), sample_list=sample_list)
    return build_reference_clustering(metabolism_df, scoring=scoring)


def create_benchmark_config(
    neural_data_path: str,
    metabolism_path: str,
    output_root: str = "results/tmp",
    scoring: str = "gap",
) -> dict:
    return asdict(
        BenchmarkConfig(
            neural_data_path=neural_data_path,
            metabolism_path=metabolism_path,
            output_root=output_root,
            scoring=scoring,
        )
    )


def create_benchmark_grid() -> list[dict]:
    configs: list[dict] = []

    def add_config(method: str, scaling_mode: str, **kwargs) -> None:
        config_id = f"{method}_{scaling_mode}"
        configs.append(
            {
                "config_id": config_id,
                "method": method,
                "scaling_mode": scaling_mode,
                **kwargs,
            }
        )

    add_config("pca", "none", scaling="none", scaling_orientation="stimuluswise")
    add_config("pca", "standard_stimuluswise", scaling="standard", scaling_orientation="stimuluswise")
    add_config("pca", "standard_neuronwise", scaling="standard", scaling_orientation="neuronwise")

    add_config("tca", "none", scaling="none", scaling_orientation="stimuluswise")
    add_config("tca", "standard_stimuluswise", scaling="standard", scaling_orientation="stimuluswise")
    add_config("tca", "standard_neuronwise", scaling="standard", scaling_orientation="neuronwise")
    add_config("tca", "soft", scaling="soft", scaling_orientation="stimuluswise")

    add_config("dpca", "raw_component_embedding", use_reconstruction=False, scaling_orientation="neuronwise")
    add_config("dpca", "reconstruction_stimuluswise", use_reconstruction=True, scaling_orientation="stimuluswise")
    add_config("dpca", "reconstruction_neuronwise", use_reconstruction=True, scaling_orientation="neuronwise")

    return configs


def run_benchmark_configs(
    configs: list[dict],
    seeds: Iterable[int],
    runner,
) -> pd.DataFrame:
    rows: list[dict] = []

    for config in configs:
        for seed in seeds:
            row = {
                "config_id": config["config_id"],
                "method": config["method"],
                "scaling_mode": config["scaling_mode"],
                "seed": seed,
            }
            try:
                metrics = runner(config, seed)
                row.update(metrics)
                row.setdefault("error", None)
            except Exception as exc:  # pragma: no cover - exercised through tests
                row.update(
                    {
                        "ari": np.nan,
                        "nmi": np.nan,
                        "v_measure": np.nan,
                        "silhouette": np.nan,
                        "best_k": np.nan,
                        "runtime_sec": np.nan,
                        "error": str(exc),
                    }
                )
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_benchmark_runs(run_results: pd.DataFrame) -> pd.DataFrame:
    if run_results.empty:
        return pd.DataFrame()

    successful = run_results[run_results["error"].isna()].copy()
    if successful.empty:
        return (
            run_results[["config_id", "method", "scaling_mode"]]
            .drop_duplicates()
            .assign(
                ari=np.nan,
                nmi=np.nan,
                v_measure=np.nan,
                silhouette=np.nan,
                best_k=np.nan,
                runtime_sec=np.nan,
                seed_count=0,
                error_count=run_results.groupby("config_id")["error"].transform("count"),
            )
        )

    agg_map: dict[str, tuple[str, str]] = {"seed_count": ("seed", "nunique")}
    for metric in ["ari", "nmi", "v_measure", "silhouette", "best_k", "runtime_sec"]:
        if metric in successful.columns:
            agg_map[metric] = (metric, "mean")

    grouped = successful.groupby(
        ["config_id", "method", "scaling_mode"], as_index=False
    ).agg(**agg_map)

    error_counts = (
        run_results.assign(has_error=run_results["error"].notna().astype(int))
        .groupby("config_id", as_index=False)["has_error"]
        .sum()
        .rename(columns={"has_error": "error_count"})
    )
    merged = grouped.merge(error_counts, on="config_id", how="left")

    for metric in ["ari", "nmi", "v_measure", "silhouette", "best_k", "runtime_sec"]:
        if metric not in merged.columns:
            merged[metric] = np.nan

    return merged


def rank_benchmark_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.sort_values(
        by=["ari", "nmi", "v_measure"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def write_benchmark_outputs(
    output_dir: Path | str,
    benchmark_results: pd.DataFrame,
    stability_results: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    metabolism_reference: pd.DataFrame,
    summary_lines: list[str] | None = None,
    config: dict | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark_results.to_csv(output_path / "benchmark_results.csv", index=False)
    benchmark_results.to_parquet(output_path / "benchmark_results.parquet", index=False)
    stability_results.to_csv(output_path / "stability_results.csv", index=False)
    ranking_summary.to_csv(output_path / "ranking_summary.csv", index=False)
    sample_manifest.to_csv(output_path / "sample_manifest.csv", index=False)
    metabolism_reference.to_csv(output_path / "metabolism_reference.csv", index=False)

    if config is not None:
        (output_path / "config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

    summary_text = "\n".join(summary_lines or ["Benchmark run completed."])
    (output_path / "summary.md").write_text(summary_text + "\n", encoding="utf-8")
    return output_path


def build_sample_manifest(
    neural_df: pd.DataFrame,
    metabolism_df: pd.DataFrame,
) -> pd.DataFrame:
    neural_samples = set(infer_sample_list(neural_df))
    matrix_samples = set(metabolism_df.index.astype(str).tolist())
    ordered_samples = sorted(neural_samples | matrix_samples)

    stim_name_map = {}
    if "sample_id" in neural_df.columns and "stim_name" in neural_df.columns:
        stim_name_map = (
            neural_df[["sample_id", "stim_name"]]
            .dropna()
            .drop_duplicates(subset=["sample_id"])
            .set_index("sample_id")["stim_name"]
            .to_dict()
        )

    return pd.DataFrame(
        {
            "sample_id": ordered_samples,
            "display_name": [stim_name_map.get(sample, sample) for sample in ordered_samples],
            "in_neural": [sample in neural_samples for sample in ordered_samples],
            "in_metabolism": [sample in matrix_samples for sample in ordered_samples],
            "in_overlap": [sample in neural_samples and sample in matrix_samples for sample in ordered_samples],
        }
    )


def prepare_benchmark_dataset(
    neural_data_path: str,
    metabolism_path: str,
    scoring: str = "gap",
) -> dict:
    neural_df = normalize_neural_samples(load_neural_dataframe(neural_data_path))
    metabolism_df = load_metabolism_data(resolve_repo_path(metabolism_path), sample_list=None)
    sample_manifest = build_sample_manifest(neural_df, metabolism_df)

    overlap_samples = sample_manifest.loc[sample_manifest["in_overlap"], "sample_id"].tolist()
    if len(overlap_samples) < 2:
        raise ValueError("Need at least 2 overlapping neural/metabolism samples for benchmarking.")

    filtered_neural_df = neural_df[neural_df["sample_id"].isin(overlap_samples)].copy()
    filtered_neural_df["stimulus"] = filtered_neural_df["sample_id"]
    if "stim_name" in filtered_neural_df.columns:
        filtered_neural_df["stim_name"] = filtered_neural_df["sample_id"]
    if "stim_color" in filtered_neural_df.columns:
        filtered_neural_df["stim_color"] = "#4c6ef5"

    _, tensor_3d, stimuli, neurons, stimulus_info, time_pts, on_window, off_window = prepare_chemo_tensor(
        filtered_neural_df
    )
    tensor_trial, _, _ = prepare_chemo_trial_tensor(filtered_neural_df)
    reference = load_metabolism_reference(metabolism_path, sample_list=stimuli, scoring=scoring)

    reference_df = pd.DataFrame(
        {
            "sample_id": reference["samples"],
            "label": reference["labels"],
        }
    )

    return {
        "neural_df": filtered_neural_df,
        "metabolism_df": metabolism_df,
        "sample_manifest": sample_manifest,
        "overlap_samples": stimuli,
        "tensor_3d": tensor_3d.astype(np.float64),
        "tensor_trial": tensor_trial.astype(np.float64),
        "neurons": neurons,
        "time_pts": time_pts,
        "stimulus_info": stimulus_info,
        "on_window": on_window,
        "off_window": off_window,
        "reference": reference,
        "reference_df": reference_df,
    }


def execute_neural_config(
    config: dict,
    seed: int,
    tensor_3d: np.ndarray,
    tensor_trial: np.ndarray,
    reference_labels: list,
    scoring: str = "gap",
    artifact_store: dict | None = None,
    sample_names: list[str] | None = None,
) -> dict:
    start = time.perf_counter()
    np.random.seed(seed)

    labels, best_k, best_score, Z, components = cluster_latent_space(
        tensor_3d,
        tensor_trial=tensor_trial if config["method"] == "dpca" else None,
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

    runtime_sec = time.perf_counter() - start
    labels_list = [int(label) for label in labels.tolist()]
    silhouette = np.nan
    if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels):
        silhouette = float(silhouette_score(components, labels))

    if artifact_store is not None:
        artifact_store[(config["config_id"], seed)] = {
            "components": components,
            "labels": labels,
            "linkage": Z,
            "samples": sample_names or [],
        }

    return {
        "ari": float(adjusted_rand_score(reference_labels, labels)),
        "nmi": float(normalized_mutual_info_score(reference_labels, labels)),
        "v_measure": float(v_measure_score(reference_labels, labels)),
        "silhouette": silhouette,
        "best_k": int(best_k),
        "runtime_sec": runtime_sec,
        "internal_score": float(best_score),
        "label_signature": json.dumps(labels_list),
    }


def compute_stability_results(run_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    successful = run_results[run_results["error"].isna()].copy()

    for config_id, group in successful.groupby("config_id"):
        label_sets = [json.loads(sig) for sig in group["label_signature"].dropna().tolist()]
        pair_ari: list[float] = []
        pair_nmi: list[float] = []

        for i in range(len(label_sets)):
            for j in range(i + 1, len(label_sets)):
                pair_ari.append(adjusted_rand_score(label_sets[i], label_sets[j]))
                pair_nmi.append(normalized_mutual_info_score(label_sets[i], label_sets[j]))

        rows.append(
            {
                "config_id": config_id,
                "stability_ari": float(np.mean(pair_ari)) if pair_ari else np.nan,
                "stability_nmi": float(np.mean(pair_nmi)) if pair_nmi else np.nan,
                "best_k_std": float(group["best_k"].std(ddof=0)) if len(group) > 0 else np.nan,
                "successful_runs": int(len(group)),
            }
        )

    return pd.DataFrame(rows)


def plot_metabolism_reference(reference: dict, output_path: Path) -> None:
    embedding = reference["embedding"].copy()
    x = embedding.iloc[:, 0]
    y = embedding.iloc[:, 1] if embedding.shape[1] > 1 else np.zeros(len(embedding))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=60)
    for sample, x_val, y_val in zip(reference["samples"], x, y):
        ax.annotate(sample, (x_val, y_val), fontsize=8)
    ax.set_title("Metabolism Reference Embedding")
    ax.set_xlabel(embedding.columns[0] if len(embedding.columns) > 0 else "PC1")
    ax.set_ylabel(embedding.columns[1] if len(embedding.columns) > 1 else "PC2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    pivot = summary_df.pivot(index="method", columns="scaling_mode", values="ari").fillna(np.nan)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("ARI Heatmap")
    fig.colorbar(im, ax=ax, label="ARI")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_stability_summary(stability_df: pd.DataFrame, output_path: Path) -> None:
    if stability_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(stability_df["config_id"], stability_df["stability_ari"])
    ax.set_title("Stability Summary")
    ax.set_ylabel("Pairwise ARI")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_method_scaling_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(summary_df["config_id"], summary_df["ari"])
    ax.set_title("Method and Scaling Comparison")
    ax.set_ylabel("Mean ARI")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_top_run_artifacts(
    best_config_id: str,
    run_results: pd.DataFrame,
    artifact_store: dict,
    output_dir: Path,
) -> None:
    successful = run_results[
        (run_results["config_id"] == best_config_id) & run_results["error"].isna()
    ].sort_values("ari", ascending=False)
    if successful.empty:
        return

    best_run = successful.iloc[0]
    artifact = artifact_store.get((best_config_id, int(best_run["seed"])))
    if artifact is None:
        return

    top_dir = output_dir / "top_runs"
    top_dir.mkdir(exist_ok=True)

    components = artifact["components"]
    labels = artifact["labels"]
    samples = artifact["samples"]
    if components.shape[1] == 1:
        x = components[:, 0]
        y = np.zeros(len(x))
    else:
        x = components[:, 0]
        y = components[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=labels, cmap="tab20", s=70)
    for sample, x_val, y_val in zip(samples, x, y):
        ax.annotate(sample, (x_val, y_val), fontsize=8)
    ax.set_title(f"Top Run Embedding: {best_config_id}")
    fig.tight_layout()
    fig.savefig(top_dir / f"{best_config_id}_embedding.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(artifact["linkage"], labels=samples, ax=ax, leaf_rotation=90)
    ax.set_title(f"Top Run Dendrogram: {best_config_id}")
    fig.tight_layout()
    fig.savefig(top_dir / f"{best_config_id}_dendrogram.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark neural clustering methods against metabolism reference labels.")
    parser.add_argument("--data", required=True, help="Path to the neural parquet dataset.")
    parser.add_argument("--metabolism", required=True, help="Path to the metabolism matrix Excel file.")
    parser.add_argument("--output-root", default="results/tmp", help="Directory for benchmark outputs.")
    parser.add_argument("--scoring", default="gap", choices=["gap", "silhouette"], help="Reference cluster-count scoring mode.")
    args = parser.parse_args()

    config = create_benchmark_config(args.data, args.metabolism, output_root=args.output_root, scoring=args.scoring)
    prepared = prepare_benchmark_dataset(args.data, args.metabolism, scoring=args.scoring)
    benchmark_grid = create_benchmark_grid()
    artifact_store: dict = {}
    seeds = [0, 1, 2]

    def runner(run_config: dict, seed: int) -> dict:
        return execute_neural_config(
            run_config,
            seed=seed,
            tensor_3d=prepared["tensor_3d"],
            tensor_trial=prepared["tensor_trial"],
            reference_labels=prepared["reference"]["labels"],
            scoring=args.scoring,
            artifact_store=artifact_store,
            sample_names=prepared["overlap_samples"],
        )

    run_results = run_benchmark_configs(benchmark_grid, seeds=seeds, runner=runner)
    summary_df = aggregate_benchmark_runs(run_results)
    ranking_summary = rank_benchmark_results(summary_df)
    stability_results = compute_stability_results(run_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_output_root(args.output_root) / f"{timestamp}_clustering_scaling_benchmark"

    top_config = ranking_summary.iloc[0]["config_id"] if not ranking_summary.empty else "none"
    summary_lines = [
        "# Clustering Scaling Benchmark",
        f"- Samples in overlap: {len(prepared['overlap_samples'])}",
        f"- Top configuration: {top_config}",
    ]

    write_benchmark_outputs(
        output_dir=output_dir,
        benchmark_results=run_results,
        stability_results=stability_results,
        ranking_summary=ranking_summary,
        sample_manifest=prepared["sample_manifest"],
        metabolism_reference=prepared["reference_df"],
        summary_lines=summary_lines,
        config=config,
    )

    plot_metabolism_reference(prepared["reference"], output_dir / "01_metabolism_reference.png")
    plot_metric_heatmap(summary_df, output_dir / "02_metric_heatmap.png")
    plot_stability_summary(stability_results, output_dir / "03_stability_summary.png")
    plot_method_scaling_comparison(summary_df, output_dir / "04_method_scaling_comparison.png")
    if top_config != "none":
        plot_top_run_artifacts(top_config, run_results, artifact_store, output_dir)

    print(f"Benchmark completed. Output directory: {output_dir}")
    if not ranking_summary.empty:
        print(ranking_summary[["rank", "config_id", "ari", "nmi", "v_measure"]].to_string(index=False))


if __name__ == "__main__":
    main()
