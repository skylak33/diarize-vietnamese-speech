"""Recompute reference-similarity statistics from saved CPU embedding artifacts."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.cluster import cluster_ahc
from diarize.embedding_diagnostics import classify_reference_windows, reference_similarity_stats
from eval.der import load_rttm
from experiments.audit_ecapa_checkpoint import representation_stats


def _aggregate(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return None if not values else {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def center_and_normalize(embeddings, centroid):
    """Subtract a fixed centroid and L2-normalize nonzero residuals."""
    embeddings, centroid = np.asarray(embeddings, dtype=float), np.asarray(centroid, dtype=float)
    if embeddings.ndim != 2 or centroid.shape != (embeddings.shape[1],):
        raise ValueError("centroid must match a 2-D embedding matrix")
    residuals = embeddings - centroid
    norms = np.linalg.norm(residuals, axis=1, keepdims=True)
    if not np.isfinite(residuals).all() or (norms <= np.finfo(float).eps).any():
        raise ValueError("centering produced a nonfinite or zero residual")
    return residuals / norms


def audit(artifacts, reference_rttm, output_dir, min_coverage=.8):
    """Write one row per recording plus aggregate descriptive ranges."""
    artifacts, output_dir = Path(artifacts), Path(output_dir)
    references = load_rttm(reference_rttm)
    directories = [path for path in sorted(artifacts.iterdir())
                   if path.is_dir() and (path / "embeddings.npz").is_file() and path.name in references]
    saved_embeddings = []
    for directory in directories:
        with np.load(directory / "embeddings.npz") as arrays:
            saved_embeddings.append(arrays["embeddings"])
    if not saved_embeddings:
        raise ValueError("no saved embeddings match the reference RTTM")
    all_embeddings = np.concatenate(saved_embeddings)
    test_estimated_centroid = all_embeddings.mean(axis=0)
    all_centered = center_and_normalize(all_embeddings, test_estimated_centroid)
    rows = []
    for directory in directories:
        embeddings_path = directory / "embeddings.npz"
        previous_path = directory / "similarity_stats.json"
        with np.load(embeddings_path) as arrays:
            classifications = classify_reference_windows(
                arrays["window_starts"], arrays["window_ends"], references[directory.name], min_coverage)
            embeddings = arrays["embeddings"]
            centered = center_and_normalize(embeddings, test_estimated_centroid)
            stats = reference_similarity_stats(embeddings, classifications)
            centered_stats = reference_similarity_stats(centered, classifications)
        previous = (json.loads(previous_path.read_text(encoding="utf-8"))
                    .get("reference_speaker_similarity", {}) if previous_path.is_file() else {})
        counts = {}
        for item in classifications:
            classification = item["classification"]
            counts[classification] = counts.get(classification, 0) + 1
        row = {
            "recording_id": directory.name,
            "same_pairs": stats["same_speaker_similarity"]["count"],
            "different_pairs": stats["different_speaker_similarity"]["count"],
            "old_fixed_range_overlap": previous.get("distribution_overlap"),
            "adaptive_overlap": stats["distribution_overlap"],
            "ks_statistic": stats["ks_statistic"],
            "wasserstein_distance": stats["wasserstein_distance"],
            "median_separation": stats["median_separation"],
            **{f"quantile_separation_{key}": value
               for key, value in stats["quantile_separation"].items()},
            "old_diagnostic_auc": previous.get("diagnostic_auc"),
            "diagnostic_auc": stats["diagnostic_auc"],
            "same_similarity_mean": stats["same_speaker_similarity"]["mean"],
            "different_similarity_mean": stats["different_speaker_similarity"]["mean"],
            "original_ahc_speakers": int(len(np.unique(cluster_ahc(embeddings, threshold=.75)))),
            "centered_same_similarity_mean": centered_stats["same_speaker_similarity"]["mean"],
            "centered_different_similarity_mean": centered_stats["different_speaker_similarity"]["mean"],
            "centered_adaptive_overlap": centered_stats["distribution_overlap"],
            "centered_ks_statistic": centered_stats["ks_statistic"],
            "centered_wasserstein_distance": centered_stats["wasserstein_distance"],
            "centered_diagnostic_auc": centered_stats["diagnostic_auc"],
            "centered_ahc_speakers": int(len(np.unique(cluster_ahc(centered, threshold=.75)))),
            "window_classifications": counts,
        }
        rows.append(row)
    fields = [
        "recording_id", "same_pairs", "different_pairs", "old_fixed_range_overlap",
        "adaptive_overlap", "ks_statistic", "wasserstein_distance", "median_separation",
        "quantile_separation_q05", "quantile_separation_q25", "quantile_separation_q50",
        "quantile_separation_q75", "quantile_separation_q95", "old_diagnostic_auc",
        "diagnostic_auc", "same_similarity_mean", "different_similarity_mean",
        "original_ahc_speakers", "centered_same_similarity_mean",
        "centered_different_similarity_mean", "centered_adaptive_overlap",
        "centered_ks_statistic", "centered_wasserstein_distance",
        "centered_diagnostic_auc", "centered_ahc_speakers", "window_classifications",
    ]
    metrics = [field for field in fields if field not in {
        "recording_id", "same_pairs", "different_pairs", "window_classifications"}]
    payload = {
        "schema_version": 1,
        "recording_count": len(rows),
        "estimator_notes": {
            "adaptive_overlap": "Pooled-data bounds, shared equal-width Freedman-Diaconis bins capped at 50.",
            "diagnostic_auc": "P(same-speaker similarity > different-speaker similarity) plus half of ties.",
            "dependence": "Statistics are descriptive; overlapping windows and repeated pairs are not independent.",
        },
        "exploratory_test_estimated_centering": {
            "warning": "The centroid is estimated from the same reused ViYT-Diar test artifacts. Results are exploratory geometry diagnostics, not unbiased validation or a production recommendation.",
            "centroid_norm": float(np.linalg.norm(test_estimated_centroid)),
            "original_geometry": representation_stats(all_embeddings),
            "centered_geometry": representation_stats(all_centered),
            "ahc_threshold": .75,
        },
        "aggregate": {field: _aggregate(rows, field) for field in metrics},
        "recordings": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "statistical_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "statistical_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "window_classifications": json.dumps(
                row["window_classifications"], sort_keys=True)})
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--reference_rttm", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_coverage", type=float, default=.8)
    args = parser.parse_args(argv)
    audit(args.artifacts, args.reference_rttm, args.output_dir, args.min_coverage)


if __name__ == "__main__":
    main()
