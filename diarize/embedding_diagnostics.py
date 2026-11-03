"""Reusable, CPU-only diagnostics for diarization embedding windows."""

import json
from pathlib import Path

import numpy as np


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
    return {"within_predicted_cluster": _summary(matrix[same & upper]),
            "between_predicted_cluster": _summary(matrix[~same & upper]),
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
        overlaps = {}
        for ref_start, ref_end, speaker in reference_segments:
            overlap = max(0., min(end, ref_end) - max(start, ref_start))
            if overlap:
                overlaps[speaker] = overlaps.get(speaker, 0.) + overlap
        duration = end - start
        total, best = sum(overlaps.values()), max(overlaps, key=overlaps.get, default=None)
        if not overlaps:
            kind = "unannotated_window"
        elif total > duration + 1e-9:
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
    """Compare only windows with unambiguous reference-speaker coverage."""
    if len(embeddings) != len(classifications):
        raise ValueError("classifications must match embeddings")
    matrix = cosine_similarity_matrix(embeddings, max_windows)
    reliable = [(i, item["speaker"]) for i, item in enumerate(classifications)
                if item["classification"] == "unambiguous_speaker_window"]
    same, different = [], []
    for position, (i, speaker) in enumerate(reliable):
        for j, other in reliable[position + 1:]:
            (same if speaker == other else different).append(matrix[i, j])
    auc = None
    if same and different:
        comparisons = np.asarray(same)[:, None] - np.asarray(different)[None, :]
        auc = float((comparisons > 0).mean() + .5 * (comparisons == 0).mean())
    return {"same_speaker_similarity": _summary(same), "different_speaker_similarity": _summary(different),
            "diagnostic_auc": auc}


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
