"""Reusable, CPU-only diagnostics for diarization embedding windows."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def validate_embeddings(embeddings, window_starts=None, window_ends=None):
    """Return validated embedding arrays and optional window timestamps."""
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be a finite 2-D array")
    if len(embeddings) and (np.linalg.norm(embeddings, axis=1) == 0).any():
        raise ValueError("embeddings must be nonzero vectors")
    if window_starts is None:
        return embeddings, None, None
    starts, ends = np.asarray(window_starts, dtype=float), np.asarray(window_ends, dtype=float)
    if starts.ndim != 1 or ends.ndim != 1 or len(starts) != len(embeddings) or len(ends) != len(embeddings):
        raise ValueError("window timestamps must match embeddings")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or (ends <= starts).any():
        raise ValueError("window timestamps must be finite increasing intervals")
    return embeddings, starts, ends


def cosine_similarity_matrix(embeddings, max_windows=None):
    """Compute pairwise cosine similarity, with an explicit quadratic-size guard."""
    embeddings, _, _ = validate_embeddings(embeddings)
    if max_windows is not None and len(embeddings) > max_windows:
        raise ValueError("embedding count exceeds max_windows")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms) @ (embeddings / norms).T


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "std": None, "quantiles": []}
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()), "quantiles": [float(x) for x in np.quantile(values, [.05, .25, .75, .95])]}


def _distribution_overlap(left, right):
    """Estimate overlap from shared equal-width bins spanning the pooled data range."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if not len(left) or not len(right):
        return None
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("distributions must contain only finite values")
    combined = np.concatenate((left, right))
    lower, upper = combined.min(), combined.max()
    if lower == upper:
        return 1.
    edges = np.histogram_bin_edges(combined, bins="fd", range=(lower, upper))
    bin_count = min(50, max(2, len(edges) - 1))
    edges = np.linspace(np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf), bin_count + 1)
    counts_left, _ = np.histogram(left, bins=edges)
    counts_right, _ = np.histogram(right, bins=edges)
    return float(np.minimum(counts_left / len(left), counts_right / len(right)).sum())


def _pairwise_ranking_auc(left, right):
    """Return P(left > right) + 0.5 P(left == right), without independence claims."""
    if not len(left) or not len(right):
        return None
    ordered = np.sort(right)
    lower = np.searchsorted(ordered, left, side="left")
    upper = np.searchsorted(ordered, left, side="right")
    return float(np.mean((lower + .5 * (upper - lower)) / len(ordered)))


def _distribution_comparison(left, right):
    """Compare empirical samples descriptively; dependent pairs receive no p-values."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if not len(left) or not len(right):
        return {"distribution_overlap": None, "ks_statistic": None,
                "wasserstein_distance": None, "median_separation": None,
                "quantile_separation": {}, "pairwise_ranking_auc": None}
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("distributions must contain only finite values")
    quantiles = [.05, .25, .5, .75, .95]
    separation = np.quantile(left, quantiles) - np.quantile(right, quantiles)
    return {
        "distribution_overlap": _distribution_overlap(left, right),
        "ks_statistic": float(ks_2samp(left, right).statistic),
        "wasserstein_distance": float(wasserstein_distance(left, right)),
        "median_separation": float(np.median(left) - np.median(right)),
        "quantile_separation": {
            f"q{int(quantile * 100):02d}": float(value)
            for quantile, value in zip(quantiles, separation)
        },
        "pairwise_ranking_auc": _pairwise_ranking_auc(left, right),
    }


def cluster_similarity_stats(embeddings, labels, max_windows=None):
    """Summarize predicted-cluster cohesion and temporal-neighbor similarity."""
    embeddings, _, _ = validate_embeddings(embeddings)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("labels must be a 1-D array matching embeddings")
    matrix = cosine_similarity_matrix(embeddings, max_windows)
    upper = np.triu(np.ones((len(labels), len(labels)), dtype=bool), k=1)
    same = labels[:, None] == labels[None, :]
    unique, counts = np.unique(labels, return_counts=True)
    centroids, compactness, nearest = [], {}, {}
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    for label in unique:
        members = normalized[labels == label]
        centroid = members.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if not centroid_norm:
            raise ValueError("predicted cluster has a zero centroid")
        centroid /= centroid_norm
        centroids.append(centroid)
        compactness[str(label)] = _summary(members @ centroid)
    centroid_matrix = cosine_similarity_matrix(np.asarray(centroids))
    for i, label in enumerate(unique):
        choices = [(float(centroid_matrix[i, j]), unique[j].item() if hasattr(unique[j], "item") else unique[j])
                   for j in range(len(unique)) if j != i]
        nearest[str(label)] = None if not choices else {"label": max(choices)[1], "similarity": max(choices)[0]}
    within, between = matrix[same & upper], matrix[~same & upper]
    return {"within_predicted_cluster": _summary(within),
            "between_predicted_cluster": _summary(between),
            "predicted_distribution_overlap": _distribution_overlap(within, between),
            "adjacent_windows": _summary(np.diag(matrix, 1)),
            "cluster_sizes": {str(label): int(count) for label, count in zip(unique, counts)},
            "centroid_similarity": centroid_matrix.tolist(), "compactness": compactness,
            "nearest_competing_cluster": nearest}


def classify_reference_windows(window_starts, window_ends, reference_segments, min_coverage=.8):
    """Classify windows by reference temporal overlap, never by their midpoint."""
    _, starts, ends = validate_embeddings(np.ones((len(window_starts), 1)), window_starts, window_ends)
    reference_segments = list(reference_segments)
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if any(not np.isfinite(start + end) or start < 0 or end <= start
           for start, end, _ in reference_segments):
        raise ValueError("reference segments must be finite positive intervals")
    result = []
    for start, end in zip(starts, ends):
        intervals = {}
        for ref_start, ref_end, speaker in reference_segments:
            clipped = max(start, ref_start), min(end, ref_end)
            if clipped[1] > clipped[0]:
                intervals.setdefault(speaker, []).append(clipped)
        active = {}
        for speaker, speaker_intervals in intervals.items():
            merged = []
            for interval_start, interval_end in sorted(speaker_intervals):
                if merged and interval_start <= merged[-1][1]:
                    merged[-1] = merged[-1][0], max(merged[-1][1], interval_end)
                else:
                    merged.append((interval_start, interval_end))
            active[speaker] = merged
        overlaps = {speaker: sum(right - left for left, right in speaker_intervals)
                    for speaker, speaker_intervals in active.items()}
        duration = end - start
        total, best = sum(overlaps.values()), max(overlaps, key=overlaps.get, default=None)
        active_intervals = list(active.values())
        has_overlap = any(
            max(left_start, right_start) < min(left_end, right_end)
            for index, left_intervals in enumerate(active_intervals)
            for right_intervals in active_intervals[index + 1:]
            for left_start, left_end in left_intervals
            for right_start, right_end in right_intervals)
        if not overlaps:
            kind = "unannotated_window"
        elif has_overlap:
            kind = "overlapping_speech_window"
        elif overlaps[best] / duration >= min_coverage and len(overlaps) == 1:
            kind = "unambiguous_speaker_window"
        elif total / duration >= min_coverage:
            kind = "mixed_speaker_window"
        else:
            kind = "uncertain_window"
        result.append({"classification": kind, "speaker": best if kind == "unambiguous_speaker_window" else None,
                       "coverage": 0. if best is None else overlaps[best] / duration})
    return result


def reference_similarity_stats(embeddings, classifications, max_windows=None):
    """Compare validated windows; AUC is the tie-aware same-vs-different ranking probability."""
    if len(embeddings) != len(classifications):
        raise ValueError("classifications must match embeddings")
    matrix = cosine_similarity_matrix(embeddings, max_windows)
    reliable = [(i, item["speaker"]) for i, item in enumerate(classifications)
                if item["classification"] == "unambiguous_speaker_window"]
    same, different = [], []
    for position, (i, speaker) in enumerate(reliable):
        for j, other in reliable[position + 1:]:
            (same if speaker == other else different).append(matrix[i, j])
    comparison = _distribution_comparison(same, different)
    return {"same_speaker_similarity": _summary(same), "different_speaker_similarity": _summary(different),
            **comparison, "diagnostic_auc": comparison["pairwise_ranking_auc"]}


def write_artifacts(output_dir, embeddings, segments, trace, config):
    """Write the stable diagnostic artifact schema without rerunning inference."""
    embeddings, starts, ends = validate_embeddings(embeddings, [s for s, _ in segments], [e for _, e in segments])
    if len(embeddings) and not np.allclose(np.linalg.norm(embeddings, axis=1), 1., atol=1e-5):
        raise ValueError("serialized embeddings must be L2-normalized")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "embeddings.npz", embeddings=embeddings, window_starts=starts, window_ends=ends)
    (output_dir / "pipeline_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
