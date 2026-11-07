"""Reference-dependent speaker partition diagnostics.

These functions are analysis-only: reference labels never affect inference.
ARI and NMI use reliable windows as equal evaluation units; duration weights
are used for composition, purity, entropy, and fragmentation summaries.
"""

from collections import Counter, defaultdict
import math

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def associate_reference_windows(windows, reference_segments):
    """Associate windows with a reference only when exactly one speaker overlaps.

    A window touching multiple reference speakers is marked ``ambiguous`` and
    excluded from reference-dependent partition metrics, irrespective of which
    speaker occupies more of the window. It remains available to official DER.
    """
    reference_segments = list(reference_segments)
    for index, (start, end, _) in enumerate(reference_segments):
        if not math.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError(f"reference segment {index} has invalid timestamps")
    rows = []
    for index, (start, end) in enumerate(windows):
        if not math.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError(f"window {index} has invalid timestamps")
        overlap = Counter()
        for ref_start, ref_end, speaker in reference_segments:
            amount = min(end, ref_end) - max(start, ref_start)
            if amount > 0:
                overlap[speaker] += amount
        status = "unlabeled" if not overlap else ("single_speaker" if len(overlap) == 1
                                                   else "ambiguous")
        rows.append({
            "window_index": index,
            "start": float(start),
            "end": float(end),
            "status": status,
            "reference_label": next(iter(overlap)) if len(overlap) == 1 else None,
            "reference_overlap_seconds": dict(sorted(overlap.items(), key=lambda item: str(item[0]))),
        })
    return rows


def _entropy(distribution):
    total = sum(distribution.values())
    return 0.0 if not total else -sum(
        (value / total) * math.log2(value / total)
        for value in distribution.values() if value)


def partition_quality(reference_labels, predicted_labels, durations=None,
                      initial_predicted_count=None, final_predicted_count=None):
    """Summarize count and partition quality on reliable labeled windows.

    ``None`` reference labels denote ambiguous or unlabeled windows. They are
    reported and excluded from every oracle metric, but predicted speaker counts
    are calculated on all supplied predictions.
    """
    reference_labels = list(reference_labels)
    predicted_labels = list(predicted_labels)
    if len(reference_labels) != len(predicted_labels):
        raise ValueError("reference_labels and predicted_labels must have equal length")
    weights = np.ones(len(reference_labels), dtype=float) if durations is None else np.asarray(durations, dtype=float)
    if weights.shape != (len(reference_labels),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("durations must be finite positive values, one per window")
    reliable = [index for index, label in enumerate(reference_labels) if label is not None]
    final_count = len(set(predicted_labels)) if final_predicted_count is None else int(final_predicted_count)
    initial_count = final_count if initial_predicted_count is None else int(initial_predicted_count)
    if final_count < 0 or initial_count < final_count:
        raise ValueError("predicted counts must satisfy initial >= final >= 0")
    ref_count = len({reference_labels[index] for index in reliable})
    summary = {
        "reference_speaker_count": ref_count,
        "initial_predicted_cluster_count": initial_count,
        "final_predicted_speaker_count": final_count,
        "absolute_count_error": abs(final_count - ref_count),
        "predicted_identities_eliminated": initial_count - final_count,
        "total_windows": len(reference_labels),
        "reliable_windows": len(reliable),
        "excluded_ambiguous_or_unlabeled_windows": len(reference_labels) - len(reliable),
        "reference_dependent": True,
    }
    if not reliable:
        summary.update(ari=None, nmi=None, duration_weighted_cluster_purity=None,
                       speaker_confusion=None, duration_weighted_cluster_entropy=None,
                       duration_weighted_cluster_fragmentation=None,
                       duration_weighted_speaker_fragmentation=None)
        return summary, [], []

    refs = [reference_labels[index] for index in reliable]
    preds = [predicted_labels[index] for index in reliable]
    reliable_weights = weights[reliable]
    cluster_composition = defaultdict(Counter)
    speaker_composition = defaultdict(Counter)
    for ref, pred, weight in zip(refs, preds, reliable_weights):
        cluster_composition[pred][ref] += float(weight)
        speaker_composition[ref][pred] += float(weight)

    cluster_rows = []
    for cluster, composition in sorted(cluster_composition.items(), key=lambda item: str(item[0])):
        total = sum(composition.values())
        cluster_rows.append({
            "predicted_cluster": cluster,
            "duration": total,
            "reference_speaker_count": len(composition),
            "purity": max(composition.values()) / total,
            "entropy_bits": _entropy(composition),
            "reference_distribution": {key: value / total for key, value in composition.items()},
        })
    speaker_rows = []
    for speaker, composition in sorted(speaker_composition.items(), key=lambda item: str(item[0])):
        total = sum(composition.values())
        speaker_rows.append({
            "reference_speaker": speaker,
            "duration": total,
            "predicted_cluster_count": len(composition),
            "dominant_cluster_share": max(composition.values()) / total,
            "cluster_distribution": {key: value / total for key, value in composition.items()},
        })

    total_weight = float(reliable_weights.sum())
    purity = sum(row["duration"] * row["purity"] for row in cluster_rows) / total_weight
    summary.update(
        ari=float(adjusted_rand_score(refs, preds)),
        nmi=float(normalized_mutual_info_score(refs, preds)),
        duration_weighted_cluster_purity=purity,
        speaker_confusion=1.0 - purity,
        duration_weighted_cluster_entropy=(
            sum(row["duration"] * row["entropy_bits"] for row in cluster_rows) / total_weight),
        duration_weighted_cluster_fragmentation=(
            sum(row["duration"] * row["reference_speaker_count"] for row in cluster_rows) / total_weight),
        duration_weighted_speaker_fragmentation=(
            sum(row["duration"] * row["predicted_cluster_count"] for row in speaker_rows) / total_weight),
    )
    return summary, cluster_rows, speaker_rows

