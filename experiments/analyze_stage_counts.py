"""Analyze saved speaker counts without rerunning diarization or audio inference."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.der import load_rttm, speaker_count


STAGE_NAMES = (
    "initial_ahc",
    "after_centroid_merge",
    "after_smoothing",
    "final_rttm_segments",
)
OUTPUT_FIELDS = (
    "recording",
    "reference_K",
    "initial_K",
    "after_centroid_K",
    "after_smoothing_K",
    "final_K",
    "DER",
    "SpeakerError",
)


def _json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _stage_count(path, name, stage):
    if not isinstance(stage, dict) or not isinstance(stage.get("labels"), list):
        raise ValueError(f"{path}: stage {name!r} must contain a labels list")
    labels = stage["labels"]
    if any(isinstance(label, bool) or not isinstance(label, int) for label in labels):
        raise ValueError(f"{path}: stage {name!r} labels must be integers")
    return len(set(labels))


def _load_artifacts(root, *, oracle=False):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Artifact directory does not exist: {root}")
    if (root / "diagnostics.json").is_file():
        return _load_runner_artifacts(root, oracle=oracle)
    directories = sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "pipeline_trace.json").is_file())
    if not directories:
        raise ValueError(f"No pipeline_trace.json artifacts under {root}")
    result = {}
    for directory in directories:
        trace_path, config_path = directory / "pipeline_trace.json", directory / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Missing config.json for {directory.name}")
        trace, config = _json(trace_path), _json(config_path)
        for path, payload in ((trace_path, trace), (config_path, config)):
            if payload.get("schema_version") != 1:
                raise ValueError(f"{path}: unsupported schema_version")
            if payload.get("recording_id") != directory.name:
                raise ValueError(
                    f"{path}: recording_id {payload.get('recording_id')!r} "
                    f"does not match directory {directory.name!r}")
        stages = trace.get("stages")
        if not isinstance(stages, dict):
            raise ValueError(f"{trace_path}: stages must be an object")
        counts = []
        for name in STAGE_NAMES:
            if name not in stages:
                raise ValueError(f"{trace_path}: missing stage {name!r}")
            counts.append(_stage_count(trace_path, name, stages[name]))
        inference = config.get("inference_configuration")
        if not isinstance(inference, dict) or "n_speakers" not in inference:
            raise ValueError(f"{config_path}: invalid inference_configuration")
        if not oracle and inference["n_speakers"] is not None:
            raise ValueError(
                f"Automatic artifact uses n_speakers for {directory.name}; "
                "reference-informed clustering is not automatic")
        if directory.name in result:
            raise ValueError(f"Duplicate recording ID: {directory.name}")
        result[directory.name] = {"counts": tuple(counts), "inference": inference}
    return result


def _load_runner_artifacts(root, *, oracle=False):
    diagnostics_path, config_path = root / "diagnostics.json", root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing config.json under {root}")
    try:
        rows = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {diagnostics_path}: {error}") from error
    config = _json(config_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{diagnostics_path}: expected a non-empty row list")
    mode = config.get("speaker_mode")
    if mode not in {"automatic", "known"}:
        raise ValueError(f"{config_path}: invalid speaker_mode")
    if oracle != (mode == "known"):
        raise ValueError(f"{config_path}: unexpected speaker_mode {mode!r}")

    result = {}
    count_fields = (
        "n_clusters_before_postmerge",
        "n_clusters_after_postmerge",
        "n_clusters_after_smoothing",
        "n_speakers_after_postprocess",
    )
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("recording_id"), str):
            raise ValueError(f"{diagnostics_path}: invalid recording row")
        recording_id = row["recording_id"]
        counts = []
        for field in count_fields:
            value = row.get(field)
            if value is None and field == "n_clusters_after_smoothing":
                counts.append(None)
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{diagnostics_path}: {recording_id} has invalid {field}")
            else:
                counts.append(value)
        if recording_id in result:
            raise ValueError(f"{diagnostics_path}: duplicate recording ID {recording_id!r}")
        result[recording_id] = {"counts": tuple(counts), "inference": config}
    return result


def _number(value, path, field, recording_id):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {recording_id} has invalid {field}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{path}: {recording_id} has invalid {field}")
    return result


def _metric_row(row, path, fallback_id=None):
    if not isinstance(row, dict):
        raise ValueError(f"{path}: evaluation rows must be objects")
    recording_id = row.get("file", row.get("recording_id", fallback_id))
    if not isinstance(recording_id, str) or not recording_id or recording_id == "__total__":
        raise ValueError(f"{path}: evaluation row has no recording ID")
    if fallback_id is not None and recording_id != fallback_id:
        raise ValueError(f"{path}: evaluation recording ID {recording_id!r} != {fallback_id!r}")
    if "DER" not in row:
        raise ValueError(f"{path}: {recording_id} is missing DER")
    speaker_error_key = "SpeakerError" if "SpeakerError" in row else "speaker_error"
    if speaker_error_key not in row:
        raise ValueError(f"{path}: {recording_id} is missing SpeakerError")
    return recording_id, {
        "DER": _number(row["DER"], path, "DER", recording_id),
        "SpeakerError": _number(row[speaker_error_key], path, "SpeakerError", recording_id),
    }


def _load_evaluation(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Evaluation artifact does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("file") != "__total__"]
        parsed = [_metric_row(row, path) for row in rows]
    elif path.suffix.lower() == ".json":
        payload = _json(path)
        if isinstance(payload.get("recordings"), list):
            parsed = [_metric_row(row, path) for row in payload["recordings"]]
        else:
            parsed = [_metric_row(row, path, recording_id)
                      for recording_id, row in payload.items() if recording_id != "__total__"]
    else:
        raise ValueError(f"{path}: evaluation must be CSV or JSON")
    result = {}
    for recording_id, metrics in parsed:
        if recording_id in result:
            raise ValueError(f"{path}: duplicate evaluation ID {recording_id!r}")
        result[recording_id] = metrics
    return result


def _require_same_ids(named_sets):
    names = list(named_sets)
    expected = named_sets[names[0]]
    differences = []
    for name in names[1:]:
        current = named_sets[name]
        if current != expected:
            differences.append(
                f"{name}: missing={sorted(expected - current)}, extra={sorted(current - expected)}")
    if differences:
        raise ValueError("Artifact IDs differ; " + "; ".join(differences))


def _origin(reference_count, counts):
    initial, centroid, smoothing, final = counts
    if initial != reference_count:
        return "initial_clustering"
    if centroid != reference_count:
        return "centroid_merge"
    if smoothing is None:
        return "unresolved_postprocessing" if final != reference_count else "none"
    if smoothing != reference_count:
        return "temporal_smoothing"
    return "final_postprocessing" if final != reference_count else "none"


def _comparison_configuration(entry, reference_count, *, oracle):
    inference = dict(entry["inference"])
    if "speaker_mode" in inference:
        mode = inference.pop("speaker_mode")
        if mode != ("known" if oracle else "automatic"):
            raise ValueError(f"Unexpected speaker_mode {mode!r}")
        requested = reference_count if oracle else None
    else:
        requested = inference.pop("n_speakers", None)
    return requested, inference


def _oracle_identity_elimination(automatic, oracle, reference_counts):
    _require_same_ids({"automatic": set(automatic), "oracle": set(oracle)})
    eliminated = []
    for recording_id in sorted(automatic):
        reference_count = reference_counts[recording_id]
        _, automatic_inference = _comparison_configuration(
            automatic[recording_id], reference_count, oracle=False)
        requested, oracle_inference = _comparison_configuration(
            oracle[recording_id], reference_count, oracle=True)
        if automatic_inference != oracle_inference:
            raise ValueError(f"{recording_id}: automatic and oracle traces are not comparable")
        if requested != reference_count:
            raise ValueError(
                f"{recording_id}: oracle n_speakers {requested!r} != reference count {reference_count}")
        initial, _, _, final = oracle[recording_id]["counts"]
        if initial != reference_count:
            raise ValueError(
                f"{recording_id}: oracle initial count {initial} != requested count {reference_count}")
        if final < initial:
            eliminated.append(recording_id)
    return {
        "comparable_recordings": len(oracle),
        "count": len(eliminated),
        "recordings": eliminated,
    }


def build_report(artifacts, reference_rttm, evaluation, expected_recordings=100,
                 oracle_artifacts=None):
    """Load existing artifacts and return per-recording counts plus aggregate diagnostics."""
    if expected_recordings <= 0:
        raise ValueError("expected_recordings must be positive")
    automatic = _load_artifacts(artifacts)
    references = load_rttm(str(reference_rttm))
    metrics = _load_evaluation(evaluation)
    ids = set(automatic)
    _require_same_ids({"artifacts": ids, "reference": set(references), "evaluation": set(metrics)})
    if len(ids) != expected_recordings:
        raise ValueError(f"Expected {expected_recordings} recordings, found {len(ids)}")
    reference_counts = {recording_id: speaker_count(references[recording_id])
                        for recording_id in ids}
    rows = []
    origins = {name: 0 for name in (
        "initial_clustering", "centroid_merge", "temporal_smoothing",
        "final_postprocessing", "unresolved_postprocessing", "none")}
    effects = {name: 0 for name in ("improved", "worsened", "unchanged")}
    residual = []
    for recording_id in sorted(ids):
        reference_count = reference_counts[recording_id]
        initial, centroid, smoothing, final = automatic[recording_id]["counts"]
        metric = metrics[recording_id]
        rows.append({
            "recording": recording_id,
            "reference_K": reference_count,
            "initial_K": initial,
            "after_centroid_K": centroid,
            "after_smoothing_K": smoothing,
            "final_K": final,
            **metric,
        })
        origins[_origin(reference_count, (initial, centroid, smoothing, final))] += 1
        initial_error, final_error = abs(initial - reference_count), abs(final - reference_count)
        effect = ("improved" if final_error < initial_error else
                  "worsened" if final_error > initial_error else "unchanged")
        effects[effect] += 1
        if final == reference_count and metric["SpeakerError"] > 1e-9:
            residual.append(recording_id)
    oracle_summary = None
    if oracle_artifacts is not None:
        oracle_summary = _oracle_identity_elimination(
            automatic, _load_artifacts(oracle_artifacts, oracle=True), reference_counts)
    return {
        "recordings": rows,
        "summary": {
            "recordings": len(rows),
            "count_error_origins": origins,
            "postprocessing_count_effect": effects,
            "correct_final_count_with_speaker_error": {
                "count": len(residual), "recordings": residual},
            "oracle_count_identity_elimination": oracle_summary,
        },
    }


def _write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stage_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(report["recordings"])
    (output_dir / "stage_count_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True,
                        help="Automatic diagnostic root containing one directory per recording")
    parser.add_argument("--reference-rttm", required=True)
    parser.add_argument("--evaluation", required=True,
                        help="Per-recording eval.der CSV or JSON output")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-artifacts",
                        help="Optional comparable reference-count diagnostic root")
    parser.add_argument("--expected-recordings", type=int, default=100)
    args = parser.parse_args(argv)
    report = build_report(
        args.artifacts, args.reference_rttm, args.evaluation,
        expected_recordings=args.expected_recordings,
        oracle_artifacts=args.oracle_artifacts)
    _write_report(report, args.output_dir)
    return report


if __name__ == "__main__":
    main()
