"""Reproducible synthetic Vietnamese diarization benchmark.

The two evaluation modes intentionally share no speech-region decisions:
``clustering_only`` uses reference speech regions, while ``full_pipeline``
uses Silero VAD.  Within each mode embeddings are extracted once and reused by
all five automatic-count clustering configurations.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.cluster import cluster_embeddings, merge_short_segments, smooth_labels
from diarize.pipeline import (extract_embeddings_for_segments, merge_consecutive_segments,
                              partition_overlapping_windows, segments_to_rttm)
from eval.der import evaluate_annotations, parse_rttm
from utils.provenance import environment, sha256


OUTPUT_ROOT = Path("results/synthetic_vietnamese_benchmark")
MODES = ("clustering_only", "full_pipeline")
CONFIGURATIONS = (
    {"name": "fixed_ahc_075", "method": "ahc", "count_estimator": None, "threshold": .75},
    {"name": "legacy_adaptive_ahc_075", "method": "ahc", "count_estimator": "adaptive_ahc", "threshold": .75},
    {"name": "robust_adaptive_ahc_075", "method": "ahc", "count_estimator": "adaptive_ahc_robust", "threshold": .75},
    {"name": "spectral", "method": "spectral", "count_estimator": None, "threshold": .75},
    {"name": "nme_sc", "method": "nme_sc", "count_estimator": None, "threshold": .75},
)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(record, field, metadata_path, suffix):
    value = record.get(field)
    path = Path(value) if value else Path(str(record["conversation_id"]) + suffix)
    if not path.is_absolute():
        path = metadata_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{metadata_path}: missing {field}: {path}")
    return path


def load_conversations(path):
    """Load generated conversation objects, ignoring unrelated JSON manifests."""
    path = Path(path)
    candidates = [path] if path.is_file() else sorted(path.rglob("*.json"))
    conversations = []
    for metadata_path in candidates:
        payload = _json(metadata_path)
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict) or "conversation_id" not in record:
                continue
            missing = [field for field in ("split", "speaker_count", "speaker_ids", "turns")
                       if field not in record]
            if missing:
                raise ValueError(f"{metadata_path}: conversation is missing {missing}")
            item = dict(record)
            item["conversation_id"] = str(item["conversation_id"])
            item["metadata_path"] = metadata_path.resolve()
            item["audio_path"] = _resolve(item, "audio_path", metadata_path, ".wav")
            item["rttm_path"] = _resolve(item, "rttm_path", metadata_path, ".rttm")
            conversations.append(item)
    if not conversations:
        raise ValueError(f"No conversation JSON objects found under {path}")
    ids = [record["conversation_id"] for record in conversations]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate conversation IDs: {duplicates}")
    return sorted(conversations, key=lambda record: record["conversation_id"])


def select_conversations(conversations, recording_ids=None, smoke=False):
    by_id = {record["conversation_id"]: record for record in conversations}
    if recording_ids:
        missing = sorted(set(recording_ids) - by_id.keys())
        if missing:
            raise ValueError(f"Unknown recording IDs: {missing}")
        selected = [by_id[value] for value in sorted(set(recording_ids))]
    else:
        selected = list(conversations)
    return selected[:1] if smoke else selected


def _reference(record):
    parsed = parse_rttm(str(record["rttm_path"]))
    recording_id = record["conversation_id"]
    if recording_id in parsed:
        return parsed[recording_id]
    if len(parsed) == 1:
        return next(iter(parsed.values()))
    raise ValueError(f"{record['rttm_path']}: no RTTM rows for {recording_id}")


def reference_speech_regions(reference):
    """Return the union of reference speech, preserving actual silence gaps."""
    intervals = sorted((float(start), float(end)) for start, end, _ in reference)
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(interval) for interval in merged]


def assign_reference_labels(windows, reference):
    """Assign windows touched by exactly one reference speaker; drop ambiguous ones."""
    labels, indexes = [], []
    for index, (start, end) in enumerate(windows):
        speakers = {speaker for ref_start, ref_end, speaker in reference
                    if min(end, ref_end) > max(start, ref_start)}
        if len(speakers) == 1:
            indexes.append(index)
            labels.append(next(iter(speakers)))
    return np.asarray(indexes, dtype=int), labels


def partition_metrics(windows, predicted, reference):
    indexes, expected = assign_reference_labels(windows, reference)
    if not len(indexes):
        return {"compatible_windows": 0, "ARI": None, "NMI": None,
                "purity": None, "fragmentation": None}
    predicted = np.asarray(predicted)[indexes]
    expected = np.asarray(expected)
    correct = 0
    for label in np.unique(predicted):
        _, counts = np.unique(expected[predicted == label], return_counts=True)
        correct += int(counts.max())
    fragments = [len(set(predicted[expected == speaker].tolist()))
                 for speaker in np.unique(expected)]
    return {"compatible_windows": len(indexes),
            "ARI": float(adjusted_rand_score(expected, predicted)),
            "NMI": float(normalized_mutual_info_score(expected, predicted)),
            "purity": correct / len(expected),
            "fragmentation": float(np.mean(fragments))}


def postprocess(windows, labels):
    """Apply the current production postprocessing sequence and constants."""
    labels = smooth_labels(windows, np.asarray(labels), window=3)
    segments = partition_overlapping_windows(windows)
    segments, labels = merge_short_segments(segments, labels, min_duration=.3)
    return merge_consecutive_segments(segments, labels, gap_threshold=0.)


def _embedding_digest(windows, embeddings):
    digest = hashlib.sha256()
    digest.update(json.dumps(windows, separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(embeddings).tobytes())
    return digest.hexdigest()


def _metric_row(mode, config, record, metric, runtime, frontend_runtime,
                partition, trace, embedding_digest, hypothesis_digest, window_count,
                duration_seconds):
    turns = record["turns"]
    mean_turn_duration = (sum(float(turn["end"]) - float(turn["start"]) for turn in turns)
                          / len(turns) if turns else 0.)
    declared_sources = record.get("source_datasets", [])
    if isinstance(declared_sources, str):
        declared_sources = [declared_sources]
    source_datasets = sorted(set(declared_sources) or
                             {turn["source_dataset"] for turn in turns
                              if turn.get("source_dataset")})
    return {
        "mode": mode, "configuration": config["name"],
        "recording_id": record["conversation_id"], "split": record["split"],
        "scenario": record.get("scenario"),
        "reference_speaker_count": int(record["speaker_count"]),
        "duration_seconds": duration_seconds, "mean_turn_duration_seconds": mean_turn_duration,
        "source_datasets": source_datasets,
        "DER": metric["DER"], "Miss": metric["miss"],
        "FalseAlarm": metric["false_alarm"], "SpeakerError": metric["speaker_error"],
        "scored_duration_seconds": metric["duration"],
        "miss_seconds": metric["miss_seconds"],
        "false_alarm_seconds": metric["false_alarm_seconds"],
        "speaker_error_seconds": metric["speaker_error_seconds"],
        "reference_speakers": metric["N_ref"], "predicted_speakers": metric["N_pred"],
        "count_correct": metric["exact_match"], "count_absolute_error": metric["abs_error"],
        "overclustered": metric["N_pred"] > metric["N_ref"],
        "underclustered": metric["N_pred"] < metric["N_ref"],
        "runtime_seconds": frontend_runtime + runtime,
        "shared_frontend_runtime_seconds": frontend_runtime,
        "clustering_postprocessing_runtime_seconds": runtime,
        "embedding_windows": window_count,
        "embedding_sha256": embedding_digest, "hypothesis_sha256": hypothesis_digest,
        **partition,
        "ahc_diagnostics": trace if config["method"] == "ahc" else {},
    }


def _summary(mode, config, rows, total):
    compatible = sum(row["compatible_windows"] for row in rows)
    def weighted(key):
        available = [row for row in rows if row[key] is not None]
        denominator = sum(row["compatible_windows"] for row in available)
        return (sum(row[key] * row["compatible_windows"] for row in available) / denominator
                if denominator else None)
    return {
        "mode": mode, "configuration": config["name"], "split": rows[0]["split"],
        "DER": total["DER"], "Miss": total["miss"],
        "FalseAlarm": total["false_alarm"], "SpeakerError": total["speaker_error"],
        "speaker_count_accuracy": total["speaker_count_accuracy"],
        "speaker_count_MAE": total["speaker_count_mae"],
        "overclustering_rate": total["overclustering_rate"],
        "underclustering_rate": total["underclustering_rate"],
        "runtime_seconds": sum(row["runtime_seconds"] for row in rows),
        "compatible_windows": compatible,
        "ARI": weighted("ARI"), "NMI": weighted("NMI"),
        "purity": weighted("purity"), "fragmentation": weighted("fragmentation"),
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def run(args):
    from diarize.vad import get_speech_segments, load_audio
    from utils.model import load_model
    from utils.seed import set_seed

    split_conversations = [record for record in load_conversations(args.conversation_json)
                           if record["split"] == args.split]
    if not split_conversations:
        raise ValueError(f"No conversations found for split {args.split!r}")
    conversations = select_conversations(split_conversations, args.recording_ids, args.smoke)
    plan = {"modes": list(MODES), "configurations": list(CONFIGURATIONS),
            "recording_ids": [record["conversation_id"] for record in conversations],
            "split": args.split, "automatic_speaker_count_only": True, "seed": args.seed}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Output already contains a benchmark: {output}")
    model_hash = sha256(args.model)
    code_paths = [Path(__file__), Path(__file__).parents[1] / "diarize/cluster.py",
                  Path(__file__).parents[1] / "diarize/pipeline.py",
                  Path(__file__).parents[1] / "diarize/vad.py",
                  Path(__file__).parents[1] / "models/ecapa_tdnn.py",
                  Path(__file__).parents[1] / "eval/der.py"]
    manifest = {"schema_version": 1, "status": "running", "plan": plan,
                "evaluation": {"collar": .25, "overlap_included": True},
                "postprocessing": {"smooth_window": 3, "short_segment_seconds": .3,
                                   "consecutive_gap_seconds": 0., "centroid_merge": False},
                "windowing": {"window_sec": args.window_sec, "step_sec": args.step_sec,
                              "min_segment_sec": args.min_segment_sec},
                "checkpoint": str(Path(args.model).resolve()), "checkpoint_sha256": model_hash,
                "code_sha256": {str(path.relative_to(Path(__file__).parents[1])): sha256(path)
                                for path in code_paths},
                "conversation_sha256": {record["conversation_id"]: sha256(record["metadata_path"])
                                        for record in conversations},
                "audio_sha256": {record["conversation_id"]: sha256(record["audio_path"])
                                 for record in conversations},
                "reference_sha256": {record["conversation_id"]: sha256(record["rttm_path"])
                                     for record in conversations},
                "generator_provenance": {record["conversation_id"]: {
                    key: record.get(key) for key in ("generator_version", "seed", "split", "scenario",
                                                     "source_utterance_ids", "source_hashes",
                                                     "source_datasets", "speech_annotation")}
                    for record in conversations}, "environment": environment()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    set_seed(args.seed)
    encoder = load_model(args.model, args.device)
    references, durations, cache = {}, {}, {}
    for record in conversations:
        recording_id = record["conversation_id"]
        references[recording_id] = _reference(record)
        reference_count = len({speaker for _, _, speaker in references[recording_id]})
        if reference_count != record["speaker_count"]:
            raise ValueError(f"{recording_id}: metadata speaker_count={record['speaker_count']} "
                             f"but RTTM has {reference_count}")
        audio, sample_rate = load_audio(str(record["audio_path"]))
        durations[recording_id] = len(audio) / sample_rate
        for mode in MODES:
            frontend_started = time.perf_counter()
            regions = (reference_speech_regions(references[recording_id])
                       if mode == "clustering_only" else get_speech_segments(
                           str(record["audio_path"]), threshold=args.vad_threshold,
                           min_speech_ms=args.vad_min_speech_ms,
                           min_silence_ms=args.vad_min_silence_ms,
                           speech_pad_ms=args.vad_speech_pad_ms, return_seconds=True))
            windows, embeddings = extract_embeddings_for_segments(
                encoder, audio, regions, args.window_sec, args.step_sec, args.min_segment_sec)
            cache[(mode, recording_id)] = (windows, embeddings,
                _embedding_digest(windows, embeddings) if len(embeddings) else None,
                time.perf_counter() - frontend_started)

    rows, summaries = [], []
    for mode in MODES:
        for config in CONFIGURATIONS:
            hypotheses, config_rows = {}, []
            hypothesis_dir = output / "hypothesis_rttm" / mode / config["name"]
            hypothesis_dir.mkdir(parents=True, exist_ok=True)
            for record in conversations:
                recording_id = record["conversation_id"]
                windows, embeddings, embedding_digest, frontend_runtime = cache[(mode, recording_id)]
                started = time.perf_counter()
                trace = {}
                labels = cluster_embeddings(
                    embeddings, method=config["method"], n_speakers=None,
                    threshold=config["threshold"], max_speakers=args.max_speakers,
                    seed=args.seed, diagnostics=trace, count_estimator=config["count_estimator"])
                segments, final_labels = postprocess(windows, labels)
                runtime = time.perf_counter() - started
                rttm = segments_to_rttm(segments, final_labels, recording_id)
                rttm_path = hypothesis_dir / f"{recording_id}.rttm"
                rttm_path.write_text(rttm + ("\n" if rttm else ""), encoding="utf-8")
                hypotheses[recording_id] = [(start, end, f"SPEAKER_{label:02d}")
                                             for (start, end), label in zip(segments, final_labels)]
                config_rows.append((record, runtime,
                    partition_metrics(windows, labels, references[recording_id]), trace,
                    embedding_digest, sha256(rttm_path), len(windows), frontend_runtime))
            metrics = evaluate_annotations(references, hypotheses, collar=.25,
                                           skip_overlap=False, durations=durations)
            method_rows = [_metric_row(mode, config, record, metrics[record["conversation_id"]],
                                       runtime, frontend_runtime, partition, trace,
                                       embedding_digest, hypothesis_digest, window_count,
                                       durations[record["conversation_id"]])
                           for record, runtime, partition, trace, embedding_digest,
                               hypothesis_digest, window_count, frontend_runtime
                           in config_rows]
            rows.extend(method_rows)
            summaries.append(_summary(mode, config, method_rows, metrics["__total__"]))

    if sha256(args.model) != model_hash:
        raise RuntimeError("Checkpoint changed during benchmark")
    (output / "recording_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _write_csv(output / "recording_results.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    manifest.update(status="complete", recording_count=len(conversations), result_rows=len(rows))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "recordings": rows, "summary": summaries}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation-json", required=True,
                        help="Generated conversation JSON file or directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("development", "validation", "holdout"),
                        required=True, help="Evaluate exactly one leakage-safe split")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--recording-ids", nargs="+")
    parser.add_argument("--smoke", action="store_true", help="Run the first selected recording only")
    parser.add_argument("--dry-run", action="store_true", help="Print the deterministic plan only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-speakers", type=int, default=10)
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--step-sec", type=float, default=.75)
    parser.add_argument("--min-segment-sec", type=float, default=.5)
    parser.add_argument("--vad-threshold", type=float, default=.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=100)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=30)
    args = parser.parse_args(argv)
    if not 0 < args.step_sec <= args.window_sec or args.min_segment_sec <= 0:
        parser.error("require 0 < step-sec <= window-sec and min-segment-sec > 0")
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
