"""Measure AHC count, partition, and hierarchy-gap stability on cached embeddings."""

import argparse
import contextlib
import csv
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.cluster import cluster_embeddings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results/partition_quality_stability/embedding_cache_verified/manifest.json"
DEFAULT_OUTPUT = ROOT / "results/partition_quality_stability/stability"
METHODS = {
    "fixed_0.65": (None, 0.65),
    "legacy_adaptive_0.75": ("adaptive_ahc", 0.75),
    "robust_adaptive_0.75": ("adaptive_ahc_robust", 0.75),
}
NOISE_SEEDS = (11, 23, 47)
NOISE_RADIANS = (0.005, 0.02, 0.05)
SUBSAMPLE_FRACTIONS = (0.75, 0.5, 0.25)
PREFIX_WINDOWS = (3, 4, 6, 8, 12, 24)


def angular_noise(embeddings, radians, seed):
    """Rotate every unit embedding by a fixed bounded angle in a random plane."""
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be a finite two-dimensional array")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if (norms == 0).any() or not 0 <= radians < math.pi:
        raise ValueError("embeddings must be nonzero and radians must be in [0, pi)")
    unit = embeddings / norms
    rng = np.random.default_rng(seed)
    tangent = rng.normal(size=unit.shape)
    tangent -= (tangent * unit).sum(axis=1, keepdims=True) * unit
    tangent_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    # A second draw is deterministic and avoids the measure-zero parallel vector case.
    while (tangent_norms == 0).any():
        mask = (tangent_norms[:, 0] == 0)
        tangent[mask] = rng.normal(size=(mask.sum(), unit.shape[1]))
        tangent[mask] -= (tangent[mask] * unit[mask]).sum(axis=1, keepdims=True) * unit[mask]
        tangent_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= tangent_norms
    return np.cos(radians) * unit + np.sin(radians) * tangent


def ordered_subsample_indices(n_windows, fraction):
    """Return deterministic, approximately uniform indices in temporal order."""
    if n_windows < 1 or not 0 < fraction <= 1:
        raise ValueError("n_windows must be positive and fraction must be in (0, 1]")
    count = max(1, int(math.ceil(n_windows * fraction)))
    return np.unique(np.linspace(0, n_windows - 1, count, dtype=int))


def changed_assignment_fraction(reference, candidate):
    """Return label-permutation-invariant assignment disagreement."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("partitions must have equal shape")
    if reference.ndim != 1:
        raise ValueError("partitions must be one-dimensional")
    if len(reference) == 0:
        return 0.0
    left, left_inverse = np.unique(reference, return_inverse=True)
    right, right_inverse = np.unique(candidate, return_inverse=True)
    contingency = np.zeros((len(left), len(right)), dtype=int)
    np.add.at(contingency, (left_inverse, right_inverse), 1)
    rows, columns = linear_sum_assignment(-contingency)
    return float(1 - contingency[rows, columns].sum() / len(reference))


def cluster_method(embeddings, method):
    diagnostics = {}
    count_estimator, threshold = METHODS[method]
    with contextlib.redirect_stdout(io.StringIO()):
        labels = cluster_embeddings(
            embeddings, method="ahc", threshold=threshold, max_speakers=10,
            count_estimator=count_estimator, diagnostics=diagnostics)
    ahc = diagnostics["ahc"]
    count = len(set(labels.tolist()))
    selected_gap_index = None
    if ahc.get("selection") == "outlying_gap":
        selected_gap_index = len(labels) - count - 1
    return np.asarray(labels), {
        "estimated_k": count,
        "selected_gap_index": selected_gap_index,
        "selected_gap_magnitude": ahc.get("selected_gap"),
        "best_gap": ahc.get("best_gap"),
        "runner_up_gap": ahc.get("runner_up_gap"),
        "gap_margin": ahc.get("gap_margin"),
        "outlier_cutoff": ahc.get("outlier_cutoff"),
        "selection": ahc.get("selection", "fixed_threshold"),
        "selection_reason": ahc.get("selection_reason", "fixed_threshold"),
        "fallback_n_speakers": ahc.get("fallback_n_speakers"),
        "eligible_gap_count": ahc.get("eligible_gap_count"),
    }


def _result_row(recording_id, mode, method, family, variant, indices, labels,
                trace, baseline_labels, baseline_trace, seed=None, magnitude=None):
    baseline_common = baseline_labels[indices]
    baseline_k = baseline_trace["estimated_k"]
    return {
        "recording_id": recording_id,
        "mode": mode,
        "method": method,
        "perturbation_family": family,
        "variant": variant,
        "seed": seed,
        "magnitude": magnitude,
        "retained_windows": len(indices),
        "retained_fraction": len(indices) / len(baseline_labels),
        "baseline_k": baseline_k,
        "baseline_selected_gap_index": baseline_trace["selected_gap_index"],
        "baseline_selected_gap_magnitude": baseline_trace["selected_gap_magnitude"],
        **trace,
        "selected_gap_changed": (
            trace["selected_gap_index"] != baseline_trace["selected_gap_index"]),
        "partition_ari_vs_unperturbed": float(adjusted_rand_score(baseline_common, labels)),
        "changed_assignment_fraction": changed_assignment_fraction(baseline_common, labels),
    }


def analyze_embeddings(embeddings, recording_id="unknown", mode="unknown"):
    """Run predefined, label-free stability experiments for one embedding sequence."""
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2 or len(embeddings) < 1:
        raise ValueError("embeddings must contain at least one two-dimensional row")
    baselines = {method: cluster_method(embeddings, method) for method in METHODS}
    rows = []
    full_indices = np.arange(len(embeddings))
    for method, (baseline_labels, baseline_trace) in baselines.items():
        for seed in NOISE_SEEDS:
            for radians in NOISE_RADIANS:
                labels, trace = cluster_method(angular_noise(embeddings, radians, seed), method)
                rows.append(_result_row(
                    recording_id, mode, method, "angular_noise", f"rad_{radians:g}",
                    full_indices, labels, trace, baseline_labels, baseline_trace, seed, radians))
        for fraction in SUBSAMPLE_FRACTIONS:
            indices = ordered_subsample_indices(len(embeddings), fraction)
            labels, trace = cluster_method(embeddings[indices], method)
            rows.append(_result_row(
                recording_id, mode, method, "ordered_subsample", f"fraction_{fraction:g}",
                indices, labels, trace, baseline_labels, baseline_trace, magnitude=fraction))
        for count in PREFIX_WINDOWS:
            if count >= len(embeddings):
                continue
            indices = np.arange(count)
            labels, trace = cluster_method(embeddings[indices], method)
            rows.append(_result_row(
                recording_id, mode, method, "short_prefix", f"first_{count}", indices,
                labels, trace, baseline_labels, baseline_trace, magnitude=count))
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["recording_id"], row["mode"], row["method"],
                row["perturbation_family"])].append(row)
    count_rows, partition_rows = [], []
    for key, values in sorted(groups.items()):
        recording_id, mode, method, family = key
        counts = [row["estimated_k"] for row in values]
        aris = [row["partition_ari_vs_unperturbed"] for row in values]
        changes = [row["changed_assignment_fraction"] for row in values]
        common = {
            "recording_id": recording_id, "mode": mode, "method": method,
            "perturbation_family": family, "runs": len(values),
            "baseline_k": values[0]["baseline_k"],
        }
        count_rows.append({
            **common, "unique_estimated_k": len(set(counts)), "min_estimated_k": min(counts),
            "max_estimated_k": max(counts),
            "count_change_fraction": sum(k != values[0]["baseline_k"] for k in counts) / len(counts),
            "selected_gap_change_fraction": sum(row["selected_gap_changed"]
                                                for row in values) / len(values),
            "unique_selected_gap_indices": len({row["selected_gap_index"] for row in values}),
            "fallback_fraction": sum(row["selection"] == "fixed_threshold_fallback"
                                     for row in values) / len(values),
            "selection_reasons": json.dumps(dict(sorted(
                (reason, sum(row["selection_reason"] == reason for row in values))
                for reason in {row["selection_reason"] for row in values}))),
        })
        partition_rows.append({
            **common, "mean_ari_vs_unperturbed": float(np.mean(aris)),
            "min_ari_vs_unperturbed": min(aris),
            "mean_changed_assignment_fraction": float(np.mean(changes)),
            "max_changed_assignment_fraction": max(changes),
        })
    return count_rows, partition_rows


def synthetic_cases():
    """Return deterministic controls; labels are for evaluation, never tuning."""
    def unit(rows):
        rows = np.asarray(rows, dtype=float)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    clear = unit([[1, .02, 0], [1, -.02, 0], [.99, .04, 0],
                  [.02, 1, 0], [-.02, 1, 0], [.04, .99, 0]])
    fragmented = unit([[1, .02, 0], [.98, .05, 0], [.75, .65, 0], [.72, .69, 0],
                       [0, 1, .02], [.02, .99, 0]])
    similar = unit([[1, .15, 0], [.95, .3, 0], [.9, .4, 0],
                    [.92, .35, 0], [.87, .48, 0], [.82, .57, 0]])
    outlier = unit([[1, .02, 0], [.99, -.02, 0], [1, .04, 0],
                    [.02, 1, 0], [-.02, .99, 0], [0, 0, 1]])
    unequal = unit([[1, angle, 0] for angle in (-.08, -.04, 0, .03, .06, .09)]
                   + [[0, 1, .02], [.03, .99, 0]])
    short = unit([[1, 0, 0], [.98, .2, 0], [0, 1, 0]])
    wrong = unit([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    return {
        "clear_separation": (clear, [0, 0, 0, 1, 1, 1]),
        "fragmented_speaker": (fragmented, [0, 0, 0, 0, 1, 1]),
        "similar_speakers": (similar, [0, 0, 0, 1, 1, 1]),
        "outlier": (outlier, [0, 0, 0, 1, 1, 0]),
        "unequal_sizes": (unequal, [0, 0, 0, 0, 0, 0, 1, 1]),
        "short_hierarchy": (short, [0, 0, 1]),
        "correct_k_wrong_partition": (wrong, [0, 0, 1, 1]),
    }


def analyze_synthetic_controls():
    results = {"schema_version": 1, "reference_metrics_are_separate_from_stability": True,
               "cases": {}}
    for name, (embeddings, reference) in synthetic_cases().items():
        case = {"n_windows": len(embeddings), "reference_k": len(set(reference)), "methods": {}}
        for method in METHODS:
            labels, trace = cluster_method(embeddings, method)
            case["methods"][method] = {
                **trace, "reference_ari": float(adjusted_rand_score(reference, labels)),
                "labels": labels.tolist(),
            }
        if name == "correct_k_wrong_partition":
            diagnostics = {}
            with contextlib.redirect_stdout(io.StringIO()):
                labels = cluster_embeddings(
                    embeddings, method="ahc", n_speakers=2, diagnostics=diagnostics)
            case["known_k_control"] = {
                "estimated_k": 2, "reference_ari": float(adjusted_rand_score(reference, labels)),
                "labels": labels.tolist(),
                "demonstrates_correct_k_wrong_partition": (
                    len(set(labels.tolist())) == 2
                    and adjusted_rand_score(reference, labels) < 1),
            }
        results["cases"][name] = case
    return results


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty result file: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cache_manifest=DEFAULT_CACHE, output_dir=DEFAULT_OUTPUT):
    manifest_path = Path(cache_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("caches"):
        raise ValueError("embedding cache manifest must be complete and nonempty")
    rows = []
    for cache in manifest["caches"]:
        cache_path = Path(cache["cache_path"])
        if not cache_path.is_absolute():
            cache_path = manifest_path.parent / cache_path
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path) as arrays:
            embeddings = arrays["embeddings"]
        if len(embeddings) != cache["embedding_windows"]:
            raise ValueError(f"window count mismatch in {cache_path}")
        rows.extend(analyze_embeddings(embeddings, cache["recording_id"], cache["mode"]))
    count_rows, partition_rows = summarize(rows)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "perturbation_results.csv", rows)
    _write_csv(output / "count_stability.csv", count_rows)
    _write_csv(output / "partition_stability.csv", partition_rows)
    (output / "synthetic_controls.json").write_text(
        json.dumps(analyze_synthetic_controls(), indent=2, allow_nan=False), encoding="utf-8")
    return {"cache_count": len(manifest["caches"]), "perturbation_runs": len(rows),
            "summary_groups": len(count_rows), "output_dir": str(output)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run(args.cache_manifest, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
