"""Oracle diagnostics and real-data CLI for saved AHC experiment artifacts."""

import argparse
from collections import Counter
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.cluster import _cut_ahc_hierarchy, cluster_embeddings
from eval.der import parse_rttm
from experiments.analyze_partition_quality import associate_reference_windows, partition_quality
from utils.provenance import sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_MANIFEST = ROOT / "results/partition_quality_stability/embedding_cache_verified/manifest.json"
DEFAULT_BENCHMARK = ROOT / "results/natural_vietnamese_benchmark/full_evaluation"
DEFAULT_ANNOTATIONS = ROOT / "results/natural_vietnamese_benchmark/datahive_references/annotation_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "results/partition_quality_stability"


def _composition(labels, members):
    return Counter(labels[index] for index in members if labels[index] is not None)


def _purity(composition):
    total = sum(composition.values())
    return None if not total else max(composition.values()) / total


def _entropy(composition):
    total = sum(composition.values())
    return None if not total else -sum(
        (count / total) * math.log2(count / total)
        for count in composition.values() if count)


def trace_reference_merges(children, distances, reference_labels, selected_n_clusters,
                           max_cut_clusters=10):
    """Trace oracle speaker composition and cut capacity of a complete AHC tree."""
    children = [list(pair) for pair in children]
    distances = list(distances)
    reference_labels = list(reference_labels)
    n_samples = len(reference_labels)
    if n_samples < 1 or len(children) != n_samples - 1 or len(distances) != len(children):
        raise ValueError("complete hierarchy requires n_samples - 1 children and distances")
    if not 1 <= selected_n_clusters <= n_samples:
        raise ValueError("selected_n_clusters must be between one and n_samples")
    if any(not math.isfinite(value) for value in distances) or any(
            right < left - 1e-12 for left, right in zip(distances, distances[1:])):
        raise ValueError("merge distances must be finite and nondecreasing")

    members = {index: [index] for index in range(n_samples)}
    performed = n_samples - selected_n_clusters
    merge_rows = []
    for step, ((left, right), distance) in enumerate(zip(children, distances)):
        if left not in members or right not in members or left == right:
            raise ValueError(f"invalid child identifiers at merge {step}")
        left_members, right_members = members[left], members[right]
        merged_members = left_members + right_members
        left_comp = _composition(reference_labels, left_members)
        right_comp = _composition(reference_labels, right_members)
        merged_comp = left_comp + right_comp
        reliable_before = sum(left_comp.values()) + sum(right_comp.values())
        purity_before = (None if not reliable_before else
                         (max(left_comp.values(), default=0)
                          + max(right_comp.values(), default=0)) / reliable_before)
        left_speakers, right_speakers = set(left_comp), set(right_comp)
        crossing = bool(left_speakers and right_speakers and
                        any(left_label != right_label
                            for left_label in left_speakers for right_label in right_speakers))
        merge_rows.append({
            "merge_index": step,
            "merge_distance": float(distance),
            "left_cluster_id": left,
            "right_cluster_id": right,
            "left_cluster_size": len(left_members),
            "right_cluster_size": len(right_members),
            "merged_cluster_size": len(merged_members),
            "reference_composition": dict(merged_comp),
            "reference_speaker_entropy_bits": _entropy(merged_comp),
            "left_purity": _purity(left_comp),
            "right_purity": _purity(right_comp),
            "purity_before_merge": purity_before,
            "purity_after_merge": _purity(merged_comp),
            "reference_crossing": crossing,
            "introduces_mixing": crossing and len(left_speakers) == len(right_speakers) == 1,
            "before_selected_cut": step < performed,
            "oracle_diagnostic": True,
        })
        members[n_samples + step] = merged_members

    if max_cut_clusters < 1:
        raise ValueError("max_cut_clusters must be positive")
    cuts = []
    for n_clusters in range(1, min(n_samples, max_cut_clusters) + 1):
        predicted = _cut_ahc_hierarchy(children, n_samples, n_clusters).tolist()
        quality, _, _ = partition_quality(reference_labels, predicted)
        cuts.append({"n_clusters": n_clusters, "merges_performed": n_samples - n_clusters,
                     "ari": quality["ari"], "nmi": quality["nmi"],
                     "purity": quality["duration_weighted_cluster_purity"]})
    selected = next(row for row in cuts if row["n_clusters"] == selected_n_clusters)
    valid_ari = [row for row in cuts if row["ari"] is not None]
    oracle_best = max(valid_ari, key=lambda row: (row["ari"], row["purity"], -row["n_clusters"])) if valid_ari else None
    crossing_before = [row for row in merge_rows
                       if row["reference_crossing"] and row["before_selected_cut"]]
    boundary_tied = (0 < performed < len(distances)
                     and math.isclose(distances[performed - 1], distances[performed],
                                      rel_tol=1e-12, abs_tol=1e-12))
    summary = {
        "selected_n_clusters": selected_n_clusters,
        "selected_cut_position": performed,
        "selected_cut_metrics": selected,
        "selected_cut_tied_distance": boundary_tied,
        "earliest_reference_crossing_merge": next(
            (row["merge_index"] for row in merge_rows if row["reference_crossing"]), None),
        "cross_speaker_merges_before_selected_cut": len(crossing_before),
        "introduces_mixing_merges_before_selected_cut": sum(
            row["introduces_mixing"] for row in crossing_before),
        "pure_clusters_before_selected_cut": sum(
            _purity(_composition(reference_labels, cluster_members)) == 1.0
            for cluster_id, cluster_members in _clusters_after(children, n_samples, performed).items()),
        "oracle_best_cut": oracle_best,
        "perfect_reference_partition_available": bool(
            oracle_best is not None and math.isclose(oracle_best["ari"], 1.0)),
        "selected_cut_below_oracle": bool(
            oracle_best is not None and selected["ari"] is not None
            and selected["ari"] < oracle_best["ari"] - 1e-12),
        "selected_cut_ari_loss_vs_oracle": (
            None if oracle_best is None or selected["ari"] is None
            else oracle_best["ari"] - selected["ari"]),
        "failure_type": _failure_type(selected, oracle_best),
        "ambiguous_or_unlabeled_leaves": sum(label is None for label in reference_labels),
        "oracle_diagnostic": True,
    }
    return summary, merge_rows, cuts


def _clusters_after(children, n_samples, merges_performed):
    active = {index: [index] for index in range(n_samples)}
    for step, (left, right) in enumerate(children[:merges_performed]):
        active[n_samples + step] = active.pop(left) + active.pop(right)
    return active


def _failure_type(selected, oracle_best):
    if selected["ari"] is None or oracle_best is None:
        return "unresolved"
    if math.isclose(selected["ari"], 1.0):
        return "none"
    if math.isclose(oracle_best["ari"], 1.0):
        return "incorrect_hierarchy_cut"
    return "incorrect_early_merge"


def _json_rows(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _annotation_records(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("eligible", True):
                records[row["recording_id"]] = row
    return records


def _cache_index(path):
    manifest = _json_rows(path)
    if manifest.get("status") != "complete":
        raise ValueError("embedding cache manifest is not complete")
    return manifest, {(row["mode"], row["recording_id"]): row
                      for row in manifest["caches"]}


def _load_cache(row, manifest_path):
    path = Path(row["cache_path"])
    if not path.is_absolute():
        path = Path(manifest_path).parent / path
    with np.load(path) as arrays:
        embeddings = np.asarray(arrays["embeddings"])
        windows = list(zip(arrays["window_starts"].tolist(), arrays["window_ends"].tolist()))
    if len(windows) != row["embedding_windows"] or len(embeddings) != len(windows):
        raise ValueError(f"cache window count mismatch: {path}")
    return path, embeddings, windows


def _configuration_labels(row, embeddings, configuration, seed, max_speakers):
    ahc = row.get("clustering_diagnostics", {}).get("ahc")
    if ahc and len(ahc.get("labels", [])) == len(embeddings):
        return np.asarray(ahc["labels"], dtype=int), "saved_ahc_trace"
    diagnostics = {}
    with contextlib.redirect_stdout(io.StringIO()):
        labels = cluster_embeddings(
            embeddings, method=configuration["method"], threshold=configuration["threshold"],
            max_speakers=max_speakers, seed=seed, diagnostics=diagnostics,
            count_estimator=configuration["count_estimator"])
    return np.asarray(labels, dtype=int), "cpu_recluster_from_verified_cache"


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_partition(rows, hierarchy_summaries):
    hierarchy_by_key = {
        (row["mode"], row["configuration"], row["recording_id"]): row
        for row in hierarchy_summaries
    }
    groups = {}
    for row in rows:
        groups.setdefault((row["mode"], row["configuration"]), []).append(row)
    summaries = []
    for (mode, configuration), values in sorted(groups.items()):
        reliable = sum(row["reliable_windows"] for row in values)
        weighted = lambda key: (sum(row[key] * row["reliable_windows"] for row in values
                                    if row[key] is not None) / reliable if reliable else None)
        scored = sum(row["scored_duration_seconds"] for row in values)
        hierarchy = [hierarchy_by_key[(mode, configuration, row["recording_id"])]
                     for row in values if (mode, configuration, row["recording_id"])
                     in hierarchy_by_key]
        available_initial = [row for row in values
                             if row["initial_predicted_cluster_count"] is not None]
        summaries.append({
            "mode": mode, "configuration": configuration, "recording_count": len(values),
            "reference_provenance": "silver_reference",
            "initial_count_accuracy": (sum(row["initial_predicted_cluster_count"]
                                           == row["reference_speaker_count"]
                                           for row in available_initial)
                                       / len(available_initial) if available_initial else None),
            "final_count_accuracy": sum(row["final_predicted_speaker_count"]
                                        == row["reference_speaker_count"]
                                        for row in values) / len(values),
            "mean_initial_k": (sum(row["initial_predicted_cluster_count"]
                                   for row in available_initial) / len(available_initial)
                               if available_initial else None),
            "initial_k_available_recordings": len(available_initial),
            "mean_final_k": sum(row["final_predicted_speaker_count"] for row in values) / len(values),
            "ari": weighted("ari"), "nmi": weighted("nmi"),
            "duration_weighted_cluster_purity": weighted("duration_weighted_cluster_purity"),
            "duration_weighted_speaker_fragmentation": weighted(
                "duration_weighted_speaker_fragmentation"),
            "DER": (100 * sum(row["miss_seconds"] + row["false_alarm_seconds"]
                              + row["speaker_error_seconds"] for row in values) / scored
                    if scored else None),
            "speaker_error_seconds": sum(row["speaker_error_seconds"] for row in values),
            "cross_speaker_merges_before_selected_cut": sum(
                row["cross_speaker_merges_before_selected_cut"] for row in hierarchy),
            "introduces_mixing_merges_before_selected_cut": sum(
                row["introduces_mixing_merges_before_selected_cut"] for row in hierarchy),
        })
    return summaries


def _critical_recordings(benchmark_rows):
    full = [row for row in benchmark_rows
            if row["mode"] == "full_pipeline" and row["overlap_condition"] == "included"]
    fixed = {row["recording_id"]: row for row in full
             if row["configuration"] == "fixed_ahc_075"}
    robust = {row["recording_id"]: row for row in full
              if row["configuration"] == "robust_adaptive_ahc_075"}
    regressions = sorted(({
        "recording_id": recording_id,
        "fixed_der": fixed[recording_id]["DER"],
        "robust_der": robust_row["DER"],
        "robust_minus_fixed_der": robust_row["DER"] - fixed[recording_id]["DER"],
    } for recording_id, robust_row in robust.items() if recording_id in fixed),
        key=lambda row: (-row["robust_minus_fixed_der"], row["recording_id"]))
    selected = []
    for recording_id in ["vi_000004"] + [row["recording_id"] for row in regressions[:5]]:
        if recording_id not in selected:
            selected.append(recording_id)
    return selected, regressions[:5]


def _plot_critical(recording_id, output, partition_clusters, hierarchy_summaries,
                   crossing_rows, benchmark_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = "full_pipeline"
    fixed_name, robust_name = "fixed_ahc_075", "robust_adaptive_ahc_075"
    summaries = {(row["configuration"], row["recording_id"]): row
                 for row in hierarchy_summaries if row["mode"] == mode}
    fixed = summaries.get((fixed_name, recording_id))
    robust = summaries.get((robust_name, recording_id))
    if not fixed or not robust:
        return None
    benchmark = next(row for row in benchmark_rows if row["mode"] == mode
                     and row["overlap_condition"] == "included"
                     and row["configuration"] == robust_name
                     and row["recording_id"] == recording_id)
    distances = np.asarray(benchmark["clustering_diagnostics"]["ahc"]["distances"])
    n_samples = len(distances) + 1
    remaining = n_samples - np.arange(1, n_samples)
    mask = remaining <= 15
    cluster_rows = [row for row in partition_clusters if row["mode"] == mode
                    and row["configuration"] == robust_name
                    and row["recording_id"] == recording_id]
    speakers = sorted({speaker for row in cluster_rows
                       for speaker in json.loads(row["reference_distribution_json"])})
    matrix = np.asarray([[json.loads(row["reference_distribution_json"]).get(speaker, 0.)
                          for speaker in speakers] for row in cluster_rows])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(remaining[mask], distances[mask], marker="o", markersize=3)
    crossing = [row for row in crossing_rows if row["mode"] == mode
                and row["configuration"] == robust_name
                and row["recording_id"] == recording_id]
    crossing_x = [n_samples - row["merge_index"] - 1 for row in crossing]
    visible_crossing = [(x, row["merge_distance"]) for x, row in zip(crossing_x, crossing)
                        if x <= 15]
    if visible_crossing:
        axes[0].scatter(*zip(*visible_crossing), marker="x", color="black", s=28,
                        label="Reference-crossing merge", zorder=3)
    axes[0].axvline(fixed["selected_n_clusters"], color="tab:blue", label="Fixed cut")
    axes[0].axvline(robust["selected_n_clusters"], color="tab:red", label="Robust cut")
    axes[0].invert_xaxis()
    axes[0].set(xlabel="Clusters remaining", ylabel="Merge distance", title="AHC hierarchy tail")
    axes[0].legend()
    image = axes[1].imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[1].set_xticks(range(len(speakers)), speakers, rotation=25, ha="right")
    axes[1].set_yticks(range(len(cluster_rows)),
                       [str(row["predicted_cluster"]) for row in cluster_rows])
    axes[1].set(xlabel="Silver-reference speaker", ylabel="Robust initial cluster",
                title="Cluster composition")
    figure.colorbar(image, ax=axes[1], label="Duration share")
    figure.suptitle(recording_id)
    path = output / "figures" / f"{recording_id}_partition_hierarchy.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def _plot_timeline(recording_id, output, reference, benchmark_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = [("Silver reference", reference)]
    for label, configuration in (
            ("Fixed AHC 0.75", "fixed_ahc_075"),
            ("Robust AHC", "robust_adaptive_ahc_075"),
            ("Spectral", "spectral"), ("NME-SC", "nme_sc")):
        path = benchmark_dir / "hypothesis_rttm" / "full_pipeline" / configuration / f"{recording_id}.rttm"
        tracks.append((label, parse_rttm(str(path)).get(recording_id, [])))

    figure, axis = plt.subplots(figsize=(12, 3.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for track_index, (_, segments) in enumerate(tracks):
        speakers = {speaker: index for index, speaker in enumerate(
            sorted({speaker for _, _, speaker in segments}))}
        for start, end, speaker in segments:
            axis.broken_barh([(start, end - start)], (track_index - .35, .7),
                             facecolors=colors(speakers[speaker] % 10), linewidth=0)
    axis.set_yticks(range(len(tracks)), [label for label, _ in tracks])
    axis.set(xlabel="Time (s)", title=f"{recording_id}: aligned final timelines")
    axis.set_ylim(-.7, len(tracks) - .3)
    axis.invert_yaxis()
    path = output / "figures" / f"{recording_id}_timeline.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def run_real_analysis(cache_manifest=DEFAULT_CACHE_MANIFEST, benchmark_dir=DEFAULT_BENCHMARK,
                      annotation_manifest=DEFAULT_ANNOTATIONS, output_dir=DEFAULT_OUTPUT):
    """Analyze frozen predictions; references are consumed only after clustering."""
    cache_manifest = Path(cache_manifest).resolve()
    benchmark_dir, output = Path(benchmark_dir).resolve(), Path(output_dir).resolve()
    cache_metadata, caches = _cache_index(cache_manifest)
    benchmark_manifest = _json_rows(benchmark_dir / "manifest.json")
    if benchmark_manifest.get("status") != "complete":
        raise ValueError("natural benchmark manifest is not complete")
    configurations = {row["name"]: row for row in benchmark_manifest["plan"]["configurations"]}
    seed, max_speakers = benchmark_manifest["plan"]["seed"], benchmark_manifest["plan"]["max_speakers"]
    annotations = _annotation_records(annotation_manifest)
    benchmark_rows = [row for row in _json_rows(benchmark_dir / "recording_results.json")
                      if row["overlap_condition"] == "included"]
    critical_ids, regressions = _critical_recordings(benchmark_rows)

    loaded_caches, references = {}, {}
    partition_rows, cluster_rows, speaker_rows = [], [], []
    merge_rows, crossing_rows, hierarchy_summaries = [], [], []
    for row in benchmark_rows:
        key = row["mode"], row["recording_id"]
        if key not in loaded_caches:
            loaded_caches[key] = _load_cache(caches[key], cache_manifest)
        cache_path, embeddings, windows = loaded_caches[key]
        if row["embedding_sha256"] != caches[key]["embedding_sha256"]:
            raise ValueError(f"embedding digest provenance mismatch for {key}")
        if row["recording_id"] not in references:
            reference_path = Path(annotations[row["recording_id"]]["rttm_path"])
            references[row["recording_id"]] = parse_rttm(str(reference_path))[row["recording_id"]]
        reference = references[row["recording_id"]]
        associated = associate_reference_windows(windows, reference)
        reference_labels = [item["reference_label"] for item in associated]
        hypothesis_path = (benchmark_dir / "hypothesis_rttm" / row["mode"]
                           / row["configuration"] / f"{row['recording_id']}.rttm")
        if sha256(hypothesis_path) != row["hypothesis_sha256"]:
            raise ValueError(f"hypothesis digest mismatch: {hypothesis_path}")
        hypothesis = parse_rttm(str(hypothesis_path)).get(row["recording_id"], [])
        final_k = len({speaker for _, _, speaker in hypothesis})
        if final_k != row["predicted_speakers"]:
            raise ValueError(f"final speaker count mismatch: {hypothesis_path}")
        base = {
            "recording_id": row["recording_id"], "mode": row["mode"],
            "configuration": row["configuration"], "partition_stage": "initial_clustering",
        }
        ahc = row.get("clustering_diagnostics", {}).get("ahc")
        if not ahc or len(ahc.get("labels", [])) != len(embeddings):
            # shortcut: old non-AHC artifacts saved aggregates but not labels; rerun only if
            # future artifacts need per-cluster diagnostics and persist deterministic labels.
            partition_rows.append({
                **base, "reference_speaker_count": row["reference_speaker_count"],
                "initial_predicted_cluster_count": None,
                "final_predicted_speaker_count": final_k,
                "absolute_count_error": abs(final_k - row["reference_speaker_count"]),
                "predicted_identities_eliminated": None,
                "total_windows": row["embedding_windows"],
                "reliable_windows": row["compatible_windows"],
                "excluded_ambiguous_or_unlabeled_windows": (
                    row["embedding_windows"] - row["compatible_windows"]),
                "reference_dependent": True, "ari": row["ARI"], "nmi": row["NMI"],
                "duration_weighted_cluster_purity": row["purity"],
                "speaker_confusion": None if row["purity"] is None else 1 - row["purity"],
                "duration_weighted_cluster_entropy": None,
                "duration_weighted_cluster_fragmentation": None,
                "duration_weighted_speaker_fragmentation": row["fragmentation"],
                "label_source": "saved_initial_aggregate_labels_not_persisted",
                "final_hypothesis_rttm": str(hypothesis_path), "DER": row["DER"],
                "Miss": row["Miss"], "FalseAlarm": row["FalseAlarm"],
                "SpeakerError": row["SpeakerError"],
                "scored_duration_seconds": row["scored_duration_seconds"],
                "miss_seconds": row["miss_seconds"],
                "false_alarm_seconds": row["false_alarm_seconds"],
                "speaker_error_seconds": row["speaker_error_seconds"],
                "embedding_cache": str(cache_path),
                "reference_provenance": row["reference_provenance"],
            })
            continue
        labels, label_source = _configuration_labels(
            row, embeddings, configurations[row["configuration"]], seed, max_speakers)
        initial_k = len(set(labels.tolist()))
        quality, clusters, speakers = partition_quality(
            reference_labels, labels.tolist(), [end - start for start, end in windows],
            initial_predicted_count=initial_k, final_predicted_count=final_k)
        if row["ARI"] is not None and not math.isclose(quality["ari"], row["ARI"], abs_tol=1e-12):
            raise ValueError(f"saved partition mismatch for {key}/{row['configuration']}")
        partition_rows.append({
            **base, **quality, "label_source": label_source,
            "final_hypothesis_rttm": str(hypothesis_path), "DER": row["DER"],
            "Miss": row["Miss"], "FalseAlarm": row["FalseAlarm"],
            "SpeakerError": row["SpeakerError"],
            "scored_duration_seconds": row["scored_duration_seconds"],
            "miss_seconds": row["miss_seconds"], "false_alarm_seconds": row["false_alarm_seconds"],
            "speaker_error_seconds": row["speaker_error_seconds"],
            "embedding_cache": str(cache_path), "reference_provenance": row["reference_provenance"],
        })
        cluster_rows.extend({
            **base, **{key: value for key, value in cluster.items()
                      if key != "reference_distribution"},
            "reference_distribution_json": json.dumps(
                cluster["reference_distribution"], sort_keys=True),
        } for cluster in clusters)
        speaker_rows.extend({
            **base, **{key: value for key, value in speaker.items()
                      if key != "cluster_distribution"},
            "cluster_distribution_json": json.dumps(
                speaker["cluster_distribution"], sort_keys=True),
        } for speaker in speakers)

        if ahc and len(ahc.get("children", [])) == len(labels) - 1:
            summary, merges, cuts = trace_reference_merges(
                ahc["children"], ahc["distances"], reference_labels, initial_k,
                max_cut_clusters=max_speakers)
            hierarchy_base = {"recording_id": row["recording_id"], "mode": row["mode"],
                              "configuration": row["configuration"]}
            hierarchy_summary = {
                **hierarchy_base, **summary,
                "initial_predicted_cluster_count": initial_k,
                "final_predicted_speaker_count": final_k,
                "estimator": ahc.get("estimator"),
                "selection": ahc.get("selection"),
                "selection_reason": ahc.get("selection_reason"),
                "selected_gap": ahc.get("selected_gap"),
                "best_gap": ahc.get("best_gap"),
                "runner_up_gap": ahc.get("runner_up_gap"),
                "gap_margin": ahc.get("gap_margin"),
                "outlier_cutoff": ahc.get("outlier_cutoff"),
                "fallback_n_speakers": ahc.get("fallback_n_speakers"),
                "cuts": cuts,
            }
            hierarchy_summaries.append(hierarchy_summary)
            for merge in merges:
                serialized = {**hierarchy_base, **merge,
                              "reference_composition": json.dumps(
                                  merge["reference_composition"], sort_keys=True)}
                merge_rows.append(serialized)
                if merge["reference_crossing"]:
                    crossing_rows.append(serialized)

    summary_rows = _aggregate_partition(partition_rows, hierarchy_summaries)
    partition_dir, hierarchy_dir = output / "partition_quality", output / "hierarchy"
    _write_csv(partition_dir / "per_recording.csv", partition_rows)
    _write_csv(partition_dir / "per_cluster.csv", cluster_rows)
    _write_csv(partition_dir / "per_speaker.csv", speaker_rows)
    _write_csv(hierarchy_dir / "merge_analysis.csv", merge_rows)
    _write_csv(hierarchy_dir / "cross_speaker_merges.csv", crossing_rows)
    selected_cuts = {
        "schema_version": 1, "oracle_cut_range": [1, max_speakers],
        "reference_provenance": "silver_reference", "recordings": hierarchy_summaries,
    }
    hierarchy_dir.mkdir(parents=True, exist_ok=True)
    (hierarchy_dir / "selected_cuts.json").write_text(
        json.dumps(selected_cuts, indent=2, allow_nan=False), encoding="utf-8")
    figures = [_plot_critical(recording_id, output, cluster_rows, hierarchy_summaries,
                              crossing_rows, benchmark_rows) for recording_id in critical_ids]
    figures.append(_plot_timeline(
        "vi_000004", output, references["vi_000004"], benchmark_dir))
    critical = []
    by_partition = {(row["recording_id"], row["mode"], row["configuration"]): row
                    for row in partition_rows}
    by_hierarchy = {(row["recording_id"], row["mode"], row["configuration"]): row
                    for row in hierarchy_summaries}
    for regression in regressions:
        recording_id = regression["recording_id"]
        critical.append({
            **regression,
            "fixed_partition": by_partition[(recording_id, "full_pipeline", "fixed_ahc_075")],
            "robust_partition": by_partition[(recording_id, "full_pipeline", "robust_adaptive_ahc_075")],
            "fixed_hierarchy": by_hierarchy[(recording_id, "full_pipeline", "fixed_ahc_075")],
            "robust_hierarchy": by_hierarchy[(recording_id, "full_pipeline", "robust_adaptive_ahc_075")],
        })
    summary = {
        "schema_version": 1, "reference_provenance": "silver_reference",
        "reference_labels_used_for_clustering": False,
        "cache_manifest": str(cache_manifest),
        "cache_source_benchmark_git_commit": cache_metadata["source_benchmark_git_commit"],
        "method_summaries": summary_rows, "largest_robust_regressions": critical,
        "critical_recording_ids": critical_ids, "figures": [path for path in figures if path],
    }
    partition_dir.mkdir(parents=True, exist_ok=True)
    (partition_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return {"per_recording_rows": len(partition_rows), "merge_rows": len(merge_rows),
            "cross_speaker_merge_rows": len(crossing_rows), "figures": len(summary["figures"]),
            "critical_recording_ids": critical_ids}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", default=str(DEFAULT_CACHE_MANIFEST))
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--annotation-manifest", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_real_analysis(
        args.cache_manifest, args.benchmark_dir, args.annotation_manifest, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

