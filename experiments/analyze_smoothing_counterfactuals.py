"""Recompute smoothing counterfactuals from saved diagnostic embeddings."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.cluster import (cluster_embeddings, merge_short_segments,
                             merge_similar_clusters, smooth_labels)
from diarize.pipeline import merge_consecutive_segments, partition_overlapping_windows
from eval.der import evaluate_annotations, load_rttm


def _speaker_count(labels):
    return len(set(np.asarray(labels).tolist()))


def _postprocess(segments, labels, use_smoothing):
    labels = smooth_labels(segments, labels, window=3) if use_smoothing else labels.copy()
    after_smoothing = _speaker_count(labels)
    final_segments = partition_overlapping_windows(segments)
    final_segments, labels = merge_short_segments(final_segments, labels, min_duration=.3)
    final_segments, labels = merge_consecutive_segments(
        final_segments, labels, gap_threshold=0.)
    return final_segments, labels, after_smoothing


def _condition_variants(embeddings, segments, config, n_speakers):
    count_estimator = config.get("count_estimator")
    count_estimator = None if count_estimator in {None, "backend_default"} else count_estimator
    labels = cluster_embeddings(
        embeddings, method=config.get("method", "ahc"), n_speakers=n_speakers,
        threshold=config.get("threshold", .65),
        max_speakers=config.get("max_speakers", 10), seed=config.get("seed", 42),
        count_estimator=count_estimator if n_speakers is None else None)
    initial_count = _speaker_count(labels)
    if config.get("centroid_merge", False) and n_speakers is None:
        labels = merge_similar_clusters(
            embeddings, labels,
            similarity_threshold=config.get("centroid_similarity_threshold", .8),
            max_small_cluster_size=config.get("max_small_cluster_size", 4),
            min_similarity_margin=config.get("centroid_similarity_margin", .05))
    centroid_count = _speaker_count(labels)
    result = {}
    for name, use_smoothing in (("current", True), ("without_smoothing", False)):
        final_segments, final_labels, smoothed_count = _postprocess(
            segments, labels, use_smoothing)
        result[name] = {
            "segments": final_segments,
            "labels": final_labels,
            "stages": {
                "initial": initial_count,
                "after_centroid_merge": centroid_count,
                "after_smoothing": smoothed_count,
                "final": _speaker_count(final_labels),
            },
        }
    return result


def recompute_variants(embeddings, segments, config, oracle_speaker_count):
    """Recluster once per condition; reference information is limited to oracle count."""
    if oracle_speaker_count < 1:
        raise ValueError("oracle_speaker_count must be positive")
    automatic = _condition_variants(embeddings, segments, config, None)
    oracle = _condition_variants(embeddings, segments, config, oracle_speaker_count)
    return {
        "automatic_current": automatic["current"],
        "automatic_without_smoothing": automatic["without_smoothing"],
        "oracle_count_current": oracle["current"],
        "oracle_count_without_smoothing": oracle["without_smoothing"],
    }


def summarize_variants(variants):
    return {name: {"stages": value["stages"]} for name, value in variants.items()}


def _load_artifact(artifact_dir):
    artifact_dir = Path(artifact_dir)
    trace = json.loads((artifact_dir / "pipeline_trace.json").read_text(encoding="utf-8"))
    if trace.get("schema_version") != 1 or not trace.get("recording_id"):
        raise ValueError("Unsupported pipeline_trace.json schema")
    config_payload = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    config = dict(config_payload.get("inference_configuration", {}))
    config["seed"] = config_payload.get("seed", config.get("seed", 42))
    with np.load(artifact_dir / "embeddings.npz") as archive:
        required = {"embeddings", "window_starts", "window_ends"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"embeddings.npz is missing: {', '.join(sorted(missing))}")
        embeddings = np.asarray(archive["embeddings"])
        starts = np.asarray(archive["window_starts"], dtype=float)
        ends = np.asarray(archive["window_ends"], dtype=float)
    if embeddings.ndim != 2 or starts.ndim != 1 or ends.ndim != 1 or not (
            len(embeddings) == len(starts) == len(ends)):
        raise ValueError("Embedding windows have inconsistent shapes")
    return trace["recording_id"], embeddings, list(zip(starts.tolist(), ends.tolist())), config


def _hypothesis(segments, labels):
    return [(start, end, f"SPEAKER_{int(label):02d}")
            for (start, end), label in zip(segments, labels)]


def _comparison(variants, prefix):
    current = variants[f"{prefix}_current"]["metrics"]
    no_smoothing = variants[f"{prefix}_without_smoothing"]["metrics"]
    return {
        "DER_delta_current_minus_without_smoothing": current["DER"] - no_smoothing["DER"],
        "speaker_error_seconds_delta_current_minus_without_smoothing":
            current["speaker_error_seconds"] - no_smoothing["speaker_error_seconds"],
    }


def analyze_recording(artifact_dir, reference_segments, duration):
    recording_id, embeddings, segments, config = _load_artifact(artifact_dir)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a finite positive number")
    oracle_count = len({speaker for _, _, speaker in reference_segments})
    variants = recompute_variants(embeddings, segments, config, oracle_count)
    payload_variants = summarize_variants(variants)
    for name, variant in variants.items():
        metrics = evaluate_annotations(
            {recording_id: reference_segments},
            {recording_id: _hypothesis(variant["segments"], variant["labels"])},
            durations={recording_id: duration})[recording_id]
        payload_variants[name]["metrics"] = metrics
    return {
        "recording_id": recording_id,
        "duration": float(duration),
        "reference_speaker_count": oracle_count,
        "reference_use": {
            "automatic_clustering": "none",
            "oracle_count_clustering": "speaker_count_only",
            "evaluation": "full_reference_rttm",
        },
        "variants": payload_variants,
        "comparisons": {
            "automatic": _comparison(payload_variants, "automatic"),
            "oracle_count": _comparison(payload_variants, "oracle_count"),
        },
    }


def _wav_duration(path):
    import soundfile
    info = soundfile.info(path)
    if not info.samplerate:
        raise ValueError("WAV has no sample rate")
    return info.frames / info.samplerate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True,
                        help="One directory containing embeddings.npz and pipeline_trace.json")
    parser.add_argument("--reference-rttm", required=True)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument("--duration", type=float, help="Recording duration in seconds")
    duration.add_argument("--wav", help="Audio file used only to obtain its duration")
    parser.add_argument("--output", help="Write JSON here instead of stdout")
    args = parser.parse_args(argv)

    recording_id, _, _, _ = _load_artifact(args.artifacts)
    references = load_rttm(args.reference_rttm)
    if recording_id not in references:
        parser.error(f"Reference RTTM has no recording {recording_id}")
    result = analyze_recording(
        args.artifacts, references[recording_id],
        args.duration if args.duration is not None else _wav_duration(args.wav))
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return result


if __name__ == "__main__":
    main()
