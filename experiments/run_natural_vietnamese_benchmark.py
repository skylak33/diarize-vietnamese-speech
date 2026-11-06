"""Run frozen diarization configurations on prepared natural Vietnamese data."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.cluster import cluster_embeddings
from diarize.pipeline import extract_embeddings_for_segments, segments_to_rttm
from diarize.vad import get_speech_segments, load_audio
from eval.der import evaluate_annotations, parse_rttm
from experiments.run_synthetic_diarization_benchmark import (
    _embedding_digest,
    _write_csv,
    partition_metrics,
    postprocess,
    reference_speech_regions,
)
from utils.model import load_model
from utils.provenance import environment, sha256
from utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/natural_vietnamese_benchmark.json"
DEFAULT_ANNOTATIONS = ROOT / "results/natural_vietnamese_benchmark/datahive_references/annotation_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "results/natural_vietnamese_benchmark/benchmark"
MODE_NAMES = {"clustering_only", "full_pipeline"}
OVERLAP_NAMES = {"included": False, "excluded": True}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path):
    config = _json(path)
    missing = [field for field in ("seed", "checkpoint", "vad", "embedding",
                                    "evaluation", "configurations")
               if field not in config]
    if missing:
        raise ValueError(f"Natural benchmark config is missing {missing}")
    contracts = {
        "checkpoint": {"path", "sha256"},
        "vad": {"threshold", "min_speech_ms", "min_silence_ms", "speech_pad_ms"},
        "embedding": {"window_sec", "step_sec", "min_segment_sec"},
        "evaluation": {"collar_seconds", "overlap_conditions", "modes"},
    }
    for section, fields in contracts.items():
        section_missing = sorted(fields - config[section].keys())
        if section_missing:
            raise ValueError(f"Natural benchmark {section} is missing {section_missing}")
    modes = config["evaluation"].get("modes")
    if not modes or set(modes) - MODE_NAMES:
        raise ValueError(f"Unsupported evaluation modes: {modes}")
    overlaps = config["evaluation"].get("overlap_conditions")
    if not overlaps or set(overlaps) - OVERLAP_NAMES.keys():
        raise ValueError(f"Unsupported overlap conditions: {overlaps}")
    if len(modes) != len(set(modes)) or len(overlaps) != len(set(overlaps)):
        raise ValueError("Evaluation modes and overlap conditions must be unique")
    configurations = config["configurations"]
    required = {"name", "method", "count_estimator", "threshold"}
    if (not configurations
            or any(required - row.keys() for row in configurations)):
        raise ValueError("Every frozen configuration needs name/method/count_estimator/threshold")
    names = [row["name"] for row in configurations]
    if len(names) != len(set(names)):
        raise ValueError("Frozen configuration names must be unique")
    if any(row["method"] not in {"ahc", "spectral", "nme_sc"}
           or row["count_estimator"] not in {None, "adaptive_ahc", "adaptive_ahc_robust"}
           or not isinstance(row["threshold"], (int, float))
           or not math.isfinite(row["threshold"])
           for row in configurations):
        raise ValueError("Invalid frozen clustering configuration")
    embedding = config["embedding"]
    if not 0 < embedding["step_sec"] <= embedding["window_sec"]:
        raise ValueError("Require 0 < embedding step_sec <= window_sec")
    if config["evaluation"]["collar_seconds"] < 0:
        raise ValueError("Evaluation collar must be non-negative")
    return config


def load_recordings(path, dataset=None):
    path = Path(path).resolve()
    records, exclusions = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if dataset and row.get("dataset") != dataset:
            continue
        if not row.get("eligible"):
            exclusions.append({
                "dataset": row.get("dataset"),
                "recording_id": row.get("recording_id"),
                "reason": row.get("exclusion_reason") or "not_eligible",
            })
            continue
        required = ("dataset", "dataset_revision", "recording_id", "audio_path",
                    "audio_sha256", "rttm_path", "rttm_sha256", "speaker_count",
                    "reference_provenance")
        missing = [field for field in required if row.get(field) is None]
        if missing:
            raise ValueError(f"{path}:{number}: eligible record is missing {missing}")
        if not isinstance(row["speaker_count"], int) or row["speaker_count"] < 1:
            raise ValueError(f"{path}:{number}: speaker_count must be a positive integer")
        audio_path, rttm_path = Path(row["audio_path"]), Path(row["rttm_path"])
        if not audio_path.is_file() or not rttm_path.is_file():
            raise FileNotFoundError(f"{path}:{number}: audio or RTTM is missing")
        if sha256(audio_path) != row["audio_sha256"]:
            raise ValueError(f"{row['recording_id']}: audio SHA256 mismatch")
        if sha256(rttm_path) != row["rttm_sha256"]:
            raise ValueError(f"{row['recording_id']}: RTTM SHA256 mismatch")
        records.append({**row, "audio_path": audio_path.resolve(),
                        "rttm_path": rttm_path.resolve()})
    if not records:
        raise ValueError(f"No eligible recordings in {path}")
    ids = [row["recording_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate eligible recording IDs")
    return sorted(records, key=lambda row: row["recording_id"]), exclusions


def select_recordings(records, recording_ids=None, smoke=False):
    by_id = {row["recording_id"]: row for row in records}
    if recording_ids:
        missing = sorted(set(recording_ids) - by_id.keys())
        if missing:
            raise ValueError(f"Unknown recording IDs: {missing}")
        records = [by_id[recording_id] for recording_id in sorted(set(recording_ids))]
    return records[:1] if smoke else records


def _reference(record):
    parsed = parse_rttm(str(record["rttm_path"]))
    recording_id = record["recording_id"]
    if recording_id not in parsed:
        raise ValueError(f"{record['rttm_path']}: no RTTM rows for {recording_id}")
    reference = parsed[recording_id]
    speakers = {speaker for _, _, speaker in reference}
    if len(speakers) != record["speaker_count"]:
        raise ValueError(
            f"{recording_id}: manifest speaker_count={record['speaker_count']} "
            f"but RTTM has {len(speakers)}")
    return reference


def _weighted(rows, key):
    available = [row for row in rows if row[key] is not None]
    denominator = sum(row["compatible_windows"] for row in available)
    return (sum(row[key] * row["compatible_windows"] for row in available) / denominator
            if denominator else None)


def _recording_row(mode, overlap_condition, config, record, metric, partition,
                   frontend, clustering_runtime, hypothesis_sha256, duration, trace):
    return {
        "mode": mode, "overlap_condition": overlap_condition,
        "configuration": config["name"], "dataset": record["dataset"],
        "dataset_revision": record["dataset_revision"],
        "reference_provenance": record["reference_provenance"],
        "recording_id": record["recording_id"], "duration_seconds": duration,
        "reference_speaker_count": record["speaker_count"],
        "DER": metric["DER"], "Miss": metric["miss"],
        "FalseAlarm": metric["false_alarm"], "SpeakerError": metric["speaker_error"],
        "scored_duration_seconds": metric["duration"],
        "miss_seconds": metric["miss_seconds"],
        "false_alarm_seconds": metric["false_alarm_seconds"],
        "speaker_error_seconds": metric["speaker_error_seconds"],
        "reference_speakers": metric["N_ref"],
        "predicted_speakers": metric["N_pred"],
        "count_correct": metric["exact_match"],
        "count_absolute_error": metric["abs_error"],
        "overclustered": metric["N_pred"] > metric["N_ref"],
        "underclustered": metric["N_pred"] < metric["N_ref"],
        "shared_frontend_runtime_seconds": frontend["runtime_seconds"],
        "clustering_postprocessing_runtime_seconds": clustering_runtime,
        "runtime_seconds": frontend["runtime_seconds"] + clustering_runtime,
        "embedding_windows": len(frontend["windows"]),
        "embedding_sha256": frontend["embedding_sha256"],
        "hypothesis_sha256": hypothesis_sha256,
        "clustering_diagnostics": trace,
        **partition,
    }


def _summary(mode, overlap_condition, config, rows, total):
    frontend_runtime = sum(row["shared_frontend_runtime_seconds"] for row in rows)
    clustering_runtime = sum(row["clustering_postprocessing_runtime_seconds"] for row in rows)
    return {
        "mode": mode, "overlap_condition": overlap_condition,
        "configuration": config["name"],
        "recording_count": len(rows), "DER": total["DER"],
        "Miss": total["miss"], "FalseAlarm": total["false_alarm"],
        "SpeakerError": total["speaker_error"],
        "speaker_count_accuracy": total["speaker_count_accuracy"],
        "speaker_count_MAE": total["speaker_count_mae"],
        "overclustering_rate": total["overclustering_rate"],
        "underclustering_rate": total["underclustering_rate"],
        "shared_frontend_runtime_seconds": frontend_runtime,
        "clustering_postprocessing_runtime_seconds": clustering_runtime,
        "runtime_seconds": frontend_runtime + clustering_runtime,
        "compatible_windows": sum(row["compatible_windows"] for row in rows),
        "ARI": _weighted(rows, "ARI"), "NMI": _weighted(rows, "NMI"),
        "purity": _weighted(rows, "purity"),
        "fragmentation": _weighted(rows, "fragmentation"),
    }


def run(args):
    config_path = Path(args.config).resolve()
    annotation_path = Path(args.annotation_manifest).resolve()
    config = load_config(config_path)
    records, exclusions = load_recordings(annotation_path, args.dataset)
    records = select_recordings(records, args.recording_ids, args.smoke)
    modes = tuple(args.modes or config["evaluation"]["modes"])
    if len(modes) != len(set(modes)) or set(modes) - set(config["evaluation"]["modes"]):
        raise ValueError("Requested modes are not enabled by the frozen config")
    plan = {
        "recording_ids": [row["recording_id"] for row in records],
        "datasets": sorted({row["dataset"] for row in records}),
        "modes": list(modes),
        "overlap_conditions": config["evaluation"]["overlap_conditions"],
        "configurations": config["configurations"],
        "automatic_speaker_count_only": True,
        "seed": config["seed"], "max_speakers": args.max_speakers,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return {"plan": plan, "exclusions": exclusions}

    checkpoint = Path(config["checkpoint"]["path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint SHA256 does not match the frozen config")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Output already contains a benchmark: {output}")

    code_paths = [Path(__file__), ROOT / "experiments/run_synthetic_diarization_benchmark.py",
                  ROOT / "diarize/cluster.py", ROOT / "diarize/pipeline.py",
                  ROOT / "diarize/vad.py", ROOT / "models/ecapa_tdnn.py", ROOT / "eval/der.py"]
    manifest = {
        "schema_version": 1, "status": "running", "plan": plan,
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "annotation_manifest": str(annotation_path),
        "annotation_manifest_sha256": sha256(annotation_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "audio_sha256": {row["recording_id"]: row["audio_sha256"] for row in records},
        "reference_sha256": {row["recording_id"]: row["rttm_sha256"] for row in records},
        "dataset_revisions": {row["dataset"]: row["dataset_revision"] for row in records},
        "reference_provenance": sorted({row["reference_provenance"] for row in records}),
        "excluded_recordings": exclusions,
        "evaluation": config["evaluation"], "vad": config["vad"],
        "embedding": config["embedding"],
        "postprocessing": {"smooth_window": 3, "short_segment_seconds": .3,
                           "consecutive_gap_seconds": 0., "centroid_merge": False},
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
        "environment": environment(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

    set_seed(config["seed"])
    encoder = load_model(str(checkpoint), args.device)
    references, durations, frontends = {}, {}, {}
    embedding = config["embedding"]
    vad = config["vad"]
    for record in records:
        recording_id = record["recording_id"]
        references[recording_id] = _reference(record)
        audio, sample_rate = load_audio(str(record["audio_path"]))
        durations[recording_id] = len(audio) / sample_rate
        for mode in modes:
            started = time.perf_counter()
            regions = (reference_speech_regions(references[recording_id])
                       if mode == "clustering_only" else get_speech_segments(
                           str(record["audio_path"]), threshold=vad["threshold"],
                           min_speech_ms=vad["min_speech_ms"],
                           min_silence_ms=vad["min_silence_ms"],
                           speech_pad_ms=vad["speech_pad_ms"], return_seconds=True))
            windows, embeddings = extract_embeddings_for_segments(
                encoder, audio, regions, embedding["window_sec"],
                embedding["step_sec"], embedding["min_segment_sec"])
            if not len(embeddings):
                raise ValueError(f"{recording_id}/{mode}: no embedding windows")
            frontends[(mode, recording_id)] = {
                "windows": windows, "embeddings": embeddings,
                "embedding_sha256": _embedding_digest(windows, embeddings),
                "runtime_seconds": time.perf_counter() - started,
            }

    rows, summaries = [], []
    for mode in modes:
        for frozen in config["configurations"]:
            hypotheses, prepared = {}, []
            hypothesis_dir = output / "hypothesis_rttm" / mode / frozen["name"]
            hypothesis_dir.mkdir(parents=True, exist_ok=True)
            for record in records:
                recording_id = record["recording_id"]
                frontend = frontends[(mode, recording_id)]
                started = time.perf_counter()
                trace = {}
                labels = cluster_embeddings(
                    frontend["embeddings"], method=frozen["method"], n_speakers=None,
                    threshold=frozen["threshold"], max_speakers=args.max_speakers,
                    seed=config["seed"], diagnostics=trace,
                    count_estimator=frozen["count_estimator"])
                segments, final_labels = postprocess(frontend["windows"], labels)
                clustering_runtime = time.perf_counter() - started
                rttm = segments_to_rttm(segments, final_labels, recording_id)
                rttm_path = hypothesis_dir / f"{recording_id}.rttm"
                rttm_path.write_text(rttm + ("\n" if rttm else ""), encoding="utf-8")
                hypotheses[recording_id] = [
                    (start, end, f"SPEAKER_{int(label):02d}")
                    for (start, end), label in zip(segments, final_labels)
                ]
                prepared.append((record, frontend, clustering_runtime, trace,
                                 partition_metrics(frontend["windows"], labels,
                                                   references[recording_id]),
                                 sha256(rttm_path)))
            for overlap_condition in config["evaluation"]["overlap_conditions"]:
                metrics = evaluate_annotations(
                    references, hypotheses,
                    collar=config["evaluation"]["collar_seconds"],
                    skip_overlap=OVERLAP_NAMES[overlap_condition], durations=durations)
                method_rows = [
                    _recording_row(mode, overlap_condition, frozen, record,
                                   metrics[record["recording_id"]], partition, frontend,
                                   clustering_runtime, hypothesis_hash,
                                   durations[record["recording_id"]], trace)
                    for record, frontend, clustering_runtime, trace, partition, hypothesis_hash
                    in prepared
                ]
                rows.extend(method_rows)
                summaries.append(_summary(
                    mode, overlap_condition, frozen, method_rows, metrics["__total__"]))

    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during benchmark")
    (output / "recording_results.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(output / "recording_results.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    manifest.update(status="complete", recording_count=len(records),
                    result_rows=len(rows), summary_rows=len(summaries))
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"manifest": manifest, "recordings": rows, "summary": summaries}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--annotation-manifest", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dataset")
    parser.add_argument("--recording-ids", nargs="+")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_NAMES))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-speakers", type=int, default=10)
    args = parser.parse_args(argv)
    if args.max_speakers < 1:
        parser.error("max-speakers must be positive")
    return args


def main(argv=None):
    result = run(parse_args(argv))
    if "manifest" in result:
        print(json.dumps({
            "recordings": result["manifest"]["recording_count"],
            "result_rows": result["manifest"]["result_rows"],
            "summary_rows": result["manifest"]["summary_rows"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
