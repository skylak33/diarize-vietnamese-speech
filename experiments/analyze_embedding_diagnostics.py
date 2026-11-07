"""Summarize saved embedding diagnostics without rerunning inference."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.embedding_diagnostics import (classify_reference_windows,
                                            cluster_similarity_stats,
                                            reference_similarity_stats)
from eval.der import load_rttm


CORRECT_COUNT_IDS = {"98EV6H0ZRlU", "B5rLHPUJ8qc", "jxTaydBnwIg", "XSC94UbpoRQ"}
UNDERCOUNT_IDS = {
    "Hx0j-Oc18do", "9lm02QThHjo", "sVPA0POzHFQ", "IZxOT44FKsU",
    "HTwksOmzTsc", "kmP3mlKeGRY", "_5qHkZ0miYY", "XJG-o9BK4To",
    "lDvnabH6iyY", "kkPMThxTwq0", "hRYWiO-zi5s", "kBbOVOaU6cc",
    "LNdIO_4fRuI", "sxOvp1cgE14",
}
OVERCOUNT_IDS = {"jE9ZB-zqEgs"}
TARGET_IDS = tuple(sorted(CORRECT_COUNT_IDS | UNDERCOUNT_IDS | OVERCOUNT_IDS))


def _labels(stage):
    return np.asarray(stage.get("labels", stage), dtype=int)


def _summary(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else {
        "min": float(values.min()), "median": float(np.median(values)),
        "max": float(values.max()),
    }


def reference_distinct_merges(ahc, classifications):
    """Find hierarchy steps joining disjoint, reliably labeled speaker sets."""
    speakers = {
        index: ({item["speaker"]}
                if item["classification"] == "unambiguous_speaker_window" else set())
        for index, item in enumerate(classifications)
    }
    mixed = {index: False for index in range(len(classifications))}
    result = []
    for merge in ahc.get("merges", []):
        left, right = merge["left_cluster_id"], merge["right_cluster_id"]
        left_speakers, right_speakers = speakers.get(left, set()), speakers.get(right, set())
        if left_speakers and right_speakers and left_speakers.isdisjoint(right_speakers):
            result.append({
                "merge_step": merge["merge_step"],
                "merge_distance": merge["merge_distance"],
                "left_reference_speakers": sorted(left_speakers),
                "right_reference_speakers": sorted(right_speakers),
                "mixed_cluster_existed_before_merge": mixed.get(left, False) or mixed.get(right, False),
            })
        cluster_id = len(classifications) + merge["merge_step"]
        speakers[cluster_id] = left_speakers | right_speakers
        mixed[cluster_id] = (mixed.get(left, False) or mixed.get(right, False)
                             or len(speakers[cluster_id]) > 1)
    return result


def analyze_recording(artifact_dir, reference_segments=None, min_reference_coverage=.8,
                      max_windows=5000, count_relation=None, speaker_error_seconds=None):
    """Create objective evidence and conservative candidate causes for one recording."""
    artifact_dir = Path(artifact_dir)
    trace = json.loads((artifact_dir / "pipeline_trace.json").read_text(encoding="utf-8"))
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "embeddings.npz") as arrays:
        embeddings = arrays["embeddings"]
        starts, ends = arrays["window_starts"], arrays["window_ends"]
    stages = trace["stages"]
    initial = _labels(stages["initial_ahc"])
    centroid = _labels(stages["after_centroid_merge"])
    smoothed = _labels(stages["after_smoothing"])
    final = _labels(stages["final_rttm_segments"])
    if not (len(initial) == len(centroid) == len(smoothed) == len(embeddings)):
        raise ValueError(f"{artifact_dir}: embedding-stage label lengths do not match")

    similarity = cluster_similarity_stats(embeddings, initial, max_windows=max_windows)
    classifications = reference_stats = None
    reference_speakers = None
    if reference_segments is not None:
        classifications = classify_reference_windows(
            starts, ends, reference_segments, min_coverage=min_reference_coverage)
        reference_stats = reference_similarity_stats(
            embeddings, classifications, max_windows=max_windows)
        reference_speakers = len({speaker for _, _, speaker in reference_segments})
    ahc_payload = {}
    ahc_path = artifact_dir / "ahc_merges.json"
    if ahc_path.is_file():
        ahc_payload = json.loads(ahc_path.read_text(encoding="utf-8"))
    ahc = ahc_payload.get("ahc", ahc_payload)
    distinct_merges = (reference_distinct_merges(ahc, classifications)
                       if classifications is not None else [])

    classification_counts = {}
    for item in classifications or []:
        name = item["classification"]
        classification_counts[name] = classification_counts.get(name, 0) + 1
    smoothing_changes = int(np.count_nonzero(smoothed != centroid))
    initial_count, centroid_count = len(set(initial)), len(set(centroid))
    smoothed_count, final_count = len(set(smoothed)), len(set(final))
    candidate_causes = []
    if reference_speakers is not None and initial_count < reference_speakers:
        candidate_causes.append("ahc_overmerging")
    if reference_speakers is not None and initial_count > reference_speakers:
        candidate_causes.append("ahc_fragmentation")
    if smoothing_changes:
        candidate_causes.append("temporal_smoothing")
    if final_count != smoothed_count:
        candidate_causes.append("segment_postprocessing")
    if classification_counts.get("mixed_speaker_window", 0):
        candidate_causes.append("mixed_speaker_windows")
    auc = None if reference_stats is None else reference_stats["diagnostic_auc"]
    overlap = None if reference_stats is None else reference_stats["distribution_overlap"]
    if reference_speakers is not None and final_count < reference_speakers and initial_count < reference_speakers:
        conclusion = ("ahc_overmerging_despite_embedding_signal"
                      if auc is not None and auc > .5 else
                      "embedding_overlap_and_ahc_overmerging")
    elif reference_speakers is not None and final_count < reference_speakers and smoothed_count < initial_count:
        conclusion = "temporal_smoothing_overmerging"
    elif reference_speakers is not None and final_count > reference_speakers:
        conclusion = ("ahc_fragmentation_with_embedding_overlap"
                      if overlap is not None and overlap >= .5 else "ahc_fragmentation")
    elif reference_speakers is not None and initial_count > reference_speakers and smoothing_changes:
        conclusion = "ahc_fragmentation_resolved_by_smoothing_with_residual_error"
    elif smoothing_changes:
        conclusion = "temporal_smoothing_changed_assignments"
    elif classification_counts.get("mixed_speaker_window", 0):
        conclusion = "mixed_windows_with_residual_assignment_error"
    else:
        conclusion = "residual_temporal_assignment_error"

    evidence = {
        "embedding_discriminability": {
            "predicted_cluster_similarity": similarity,
            "reference_speaker_similarity": reference_stats,
            "window_classifications": classification_counts,
        },
        "ahc_merge_behavior": {
            "merge_count": len(ahc.get("merges", [])),
            "merge_distance_range": _summary(ahc.get("distances", [])),
            "reference_distinct_merges": distinct_merges,
        },
        "postprocessing_changes": {
            "initial_speakers": initial_count,
            "after_centroid_merge_speakers": centroid_count,
            "after_smoothing_speakers": smoothed_count,
            "final_speakers": final_count,
            "labels_changed_by_smoothing": smoothing_changes,
        },
        "reference_reliability": {
            "source": config.get("reference_source"),
            "coverage_threshold": min_reference_coverage if reference_segments is not None else None,
            "window_classifications": classification_counts,
        },
    }
    payload = {
        "recording_id": trace["recording_id"],
        "count_relation": count_relation,
        "speaker_error_seconds": speaker_error_seconds,
        "evidence": evidence,
        "candidate_causes": candidate_causes,
        "evidence_based_conclusion": conclusion,
        "confirmed_causes": [],
        "confidence": "low" if candidate_causes else "unknown",
        "recommended_next_action": ("Review reference-distinct AHC merges and temporal changes"
                                    if candidate_causes else "Collect or load reliable reference evidence"),
    }
    similarity_payload = {
        "predicted_cluster_similarity": similarity,
        "reference_window_classifications": classifications,
        "reference_speaker_similarity": reference_stats,
    }
    (artifact_dir / "similarity_stats.json").write_text(
        json.dumps(similarity_payload, indent=2, sort_keys=True), encoding="utf-8")
    compactness = similarity["compactness"]
    with (artifact_dir / "ahc_cluster_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cluster_label", "size", "compactness_mean"])
        writer.writeheader()
        for label, size in similarity["cluster_sizes"].items():
            writer.writerow({"cluster_label": label, "size": size,
                             "compactness_mean": compactness[label]["mean"]})
    return payload


def _load_current_metrics(path):
    """Load current evaluation counts; historical target groups are not outcomes."""
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        recording_id = row.get("file", row.get("recording_id"))
        if not recording_id or not row.get("N_ref") or not row.get("N_pred"):
            continue
        n_ref, n_pred = int(row["N_ref"]), int(row["N_pred"])
        relation = ("correct_count" if n_ref == n_pred else
                    "overcount" if n_pred > n_ref else "undercount")
        result[recording_id] = {
            "count_relation": relation,
            "speaker_error_seconds": float(row["speaker_error_seconds"]),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--reference_rttm")
    parser.add_argument("--output_dir")
    parser.add_argument("--recording_id", action="append")
    parser.add_argument("--min_reference_coverage", type=float, default=.8)
    parser.add_argument("--max_windows", type=int, default=5000)
    parser.add_argument("--metrics_csv", help="Current per-recording evaluation CSV")
    args = parser.parse_args(argv)
    root = Path(args.artifacts)
    selected = set(args.recording_id or [])
    directories = ([root] if (root / "pipeline_trace.json").is_file() else
                   [path for path in sorted(root.iterdir()) if (path / "pipeline_trace.json").is_file()])
    directories = [path for path in directories if not selected or path.name in selected]
    if not directories:
        parser.error("No matching diagnostic artifact directories")
    references = load_rttm(args.reference_rttm) if args.reference_rttm else {}
    metrics = _load_current_metrics(args.metrics_csv)
    rows = [analyze_recording(
        path, references.get(path.name), args.min_reference_coverage, args.max_windows,
        count_relation=metrics.get(path.name, {}).get("count_relation"),
        speaker_error_seconds=metrics.get(path.name, {}).get("speaker_error_seconds"))
        for path in directories]
    output_dir = Path(args.output_dir or root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({"recordings": rows}, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["recording_id", "count_relation", "speaker_error_seconds",
                  "evidence_based_conclusion", "candidate_causes",
                  "confirmed_causes", "confidence", "recommended_next_action"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (";".join(row[key]) if isinstance(row[key], list) else row[key])
                             for key in fields})
    return rows


if __name__ == "__main__":
    main()
