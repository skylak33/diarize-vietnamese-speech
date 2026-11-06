"""Build validated natural Vietnamese diarization references and QC manifests."""

import argparse
import hashlib
import io
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.pipeline import segments_to_rttm
from diarize.vad import get_speech_segments_from_audio
from utils.provenance import sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/natural_vietnamese_benchmark.json"
DEFAULT_ACQUISITION = ROOT / "results/natural_vietnamese_benchmark/acquisition_manifest.json"
DEFAULT_OUTPUT = ROOT / "results/natural_vietnamese_benchmark/references"
TIMESTAMP = re.compile(r"^(\d{2,}):([0-5]\d):([0-5]\d)(?:\.(\d{1,9}))?$")


def parse_timestamp(value):
    if not isinstance(value, str) or not (match := TIMESTAMP.fullmatch(value)):
        raise ValueError(f"invalid timestamp {value!r}")
    hours, minutes, seconds = map(int, match.group(1, 2, 3))
    fraction = float(f"0.{match.group(4)}") if match.group(4) else 0.0
    return hours * 3600 + minutes * 60 + seconds + fraction


def parse_humyn_annotation(transcript_json, duration_seconds):
    """Parse strict Humyn timestamped speaker turns without repairing them."""
    try:
        rows = json.loads(transcript_json)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("unsupported annotation format: expected JSON") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError("unsupported annotation format: expected a non-empty JSON list")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("invalid audio duration")

    segments, labels, speaker_map, seen = [], [], {}, set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"segment {index}: expected an object")
        speaker = row.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"segment {index}: invalid speaker")
        speaker = speaker.strip()
        start, end = parse_timestamp(row.get("start")), parse_timestamp(row.get("end"))
        if start >= end:
            raise ValueError(f"segment {index}: start must be before end")
        if end > duration_seconds + 1e-6:
            raise ValueError(f"segment {index}: end exceeds audio duration")
        duplicate = (start, end, speaker)
        if duplicate in seen:
            raise ValueError(f"segment {index}: duplicate segment")
        seen.add(duplicate)
        speaker_map.setdefault(speaker, len(speaker_map))
        segments.append((start, end))
        labels.append(speaker_map[speaker])
    return segments, np.asarray(labels), speaker_map


def _overlaps(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def _mask(segments, frames, sample_rate):
    mask = np.zeros(frames, dtype=bool)
    for start, end in segments:
        mask[max(0, int(start * sample_rate)):min(frames, math.ceil(end * sample_rate))] = True
    return mask


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) if len(values) else 0.0


def _channel_stats(audio, silence_amplitude, clipping_amplitude):
    return {
        "rms": _rms(audio),
        "peak": float(np.max(np.abs(audio))),
        "silence_fraction": float(np.mean(np.abs(audio) <= silence_amplitude)),
        "clipped_fraction": float(np.mean(np.abs(audio) >= clipping_amplitude)),
    }


def _validate_segments(segments, duration):
    for index, (start, end) in enumerate(segments):
        if not all(math.isfinite(value) for value in (start, end)):
            raise ValueError(f"VAD segment {index} is non-finite")
        if start < 0 or start >= end or end > duration + 1 / 16000:
            raise ValueError(f"VAD segment {index} is outside audio duration")


def _rttm(segments, labels, recording_id):
    if len(segments) != len(labels):
        raise ValueError("segments and labels must have equal length")
    ordered = sorted(zip(segments, labels), key=lambda item: (*item[0], int(item[1])))
    return segments_to_rttm(
        [segment for segment, _ in ordered],
        np.asarray([label for _, label in ordered]), recording_id)


def prepare_datahive_record(record, spec, vad, qc, reference_dir):
    reference_dir.mkdir(parents=True, exist_ok=True)
    path = Path(record.get("local_path", ""))
    if not path.is_file():
        raise ValueError("audio_not_cached")
    digest = sha256(path)
    if record.get("expected_sha256") and digest != record["expected_sha256"]:
        raise ValueError("audio_checksum_mismatch")
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("unreadable_audio") from error
    if audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("expected_finite_nonempty_stereo_audio")

    duration = len(audio) / sample_rate
    channel_segments = [
        get_speech_segments_from_audio(audio[:, channel], sample_rate, **vad)
        for channel in range(2)
    ]
    for segments in channel_segments:
        _validate_segments(segments, duration)
    speech = [sum(end - start for start, end in segments)
              for segments in channel_segments]
    if any(value == 0 for value in speech):
        raise ValueError("channel_without_detected_speech")

    overlaps = _overlaps(*channel_segments)
    overlap_seconds = sum(end - start for start, end in overlaps)
    union_seconds = sum(speech) - overlap_seconds
    masks = [_mask(segments, len(audio), sample_rate) for segments in channel_segments]
    exclusive = [masks[0] & ~masks[1], masks[1] & ~masks[0]]
    leakage = []
    for source, other in ((0, 1), (1, 0)):
        source_rms = _rms(audio[exclusive[source], source])
        other_rms = _rms(audio[exclusive[source], other])
        leakage.append(other_rms / source_rms if source_rms else 0.0)
    if np.std(audio[:, 0]) and np.std(audio[:, 1]):
        correlation = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
    else:
        correlation = 0.0
    overlap_ratio = overlap_seconds / union_seconds if union_seconds else 0.0
    imbalance = min(speech) / max(speech)
    flags = []
    if abs(correlation) >= qc["high_channel_correlation"]:
        flags.append("high_cross_channel_correlation")
    if overlap_ratio >= qc["high_overlap_ratio"]:
        flags.append("high_overlap_ratio")
    if max(leakage) >= qc["leakage_rms_ratio"]:
        flags.append("possible_cross_channel_leakage")
    if imbalance <= qc["low_speech_balance_ratio"]:
        flags.append("severe_channel_imbalance")

    segments = channel_segments[0] + channel_segments[1]
    labels = np.asarray([0] * len(channel_segments[0]) + [1] * len(channel_segments[1]))
    rttm = _rttm(segments, labels, record["recording_id"])
    rttm_path = reference_dir / f"{record['recording_id']}.rttm"
    rttm_path.write_text(rttm + "\n", encoding="utf-8")
    diagnostics = {
        "duration_seconds": duration,
        "channels": [
            {**_channel_stats(audio[:, channel], qc["silence_amplitude"],
                              qc["clipping_amplitude"]),
             "vad_segment_count": len(channel_segments[channel]),
             "vad_speech_seconds": speech[channel]}
            for channel in range(2)
        ],
        "speech_balance_ratio": imbalance,
        "overlap_seconds": overlap_seconds,
        "overlap_ratio": overlap_ratio,
        "zero_lag_correlation": correlation,
        "exclusive_speech_leakage_rms_ratio": leakage,
        "review_flags": flags,
    }
    review = {
        "suspicious_intervals": [
            {"start": start, "end": end,
             "reason": "simultaneous_channel_activity"}
            for start, end in overlaps
            if end - start >= qc["suspicious_overlap_min_seconds"]
        ],
        "ordinary_candidates": [
            {"start": start, "end": end, "speaker": f"SPEAKER_{channel:02d}"}
            for channel, own in enumerate(channel_segments)
            for start, end in own
            if not any(start < other_end and end > other_start
                       for other_start, other_end in channel_segments[1 - channel])
        ],
    }
    return {
        "eligible": True, "exclusion_reason": None,
        "audio_path": str(path.resolve()), "audio_sha256": digest,
        "rttm_path": str(rttm_path.resolve()), "rttm_sha256": sha256(rttm_path),
        "speaker_count": 2, "reference_provenance": spec["reference_provenance"],
        "speaker_mapping": {"left": "SPEAKER_00", "right": "SPEAKER_01"},
        "diagnostics": diagnostics,
    }, review


def _read_humyn_rows(path, indices):
    import pyarrow.parquet as pq

    wanted, found = set(indices), {}
    for index, batch in enumerate(pq.ParquetFile(path).iter_batches(
            batch_size=1, columns=["file_name", "audio", "transcript_json", "type"])):
        if index in wanted:
            found[index] = batch.to_pylist()[0]
        if len(found) == len(wanted):
            break
    return found


def prepare_humyn_record(record, row, spec, reference_dir, audio_dir):
    reference_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    if row.get("file_name") != record["file_name"]:
        raise ValueError("parquet_row_identity_mismatch")
    audio_value = row.get("audio") or {}
    audio_bytes = audio_value.get("bytes")
    if not audio_bytes:
        raise ValueError("missing_embedded_audio")
    audio_bytes = bytes(audio_bytes)
    try:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("unreadable_audio") from error
    if not len(audio) or not np.isfinite(audio).all() or sample_rate <= 0:
        raise ValueError("invalid_audio")
    duration = len(audio) / sample_rate
    segments, labels, speaker_map = parse_humyn_annotation(
        row.get("transcript_json"), duration)
    rttm = _rttm(segments, labels, record["recording_id"])
    rttm_path = reference_dir / f"{record['recording_id']}.rttm"
    rttm_path.write_text(rttm + "\n", encoding="utf-8")
    suffix = Path(record["file_name"]).suffix or ".wav"
    audio_path = audio_dir / f"{record['recording_id']}{suffix}"
    audio_path.write_bytes(audio_bytes)
    overlaps = []
    for index, ((start, end), label) in enumerate(zip(segments, labels)):
        for (other_start, other_end), other_label in zip(
                segments[index + 1:], labels[index + 1:]):
            if label != other_label and max(start, other_start) < min(end, other_end):
                overlaps.append((max(start, other_start), min(end, other_end)))
    prepared = {
        "eligible": True, "exclusion_reason": None,
        "audio_path": str(audio_path.resolve()),
        "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "rttm_path": str(rttm_path.resolve()), "rttm_sha256": sha256(rttm_path),
        "speaker_count": len(speaker_map),
        "reference_provenance": spec["reference_provenance"],
        "speaker_mapping": {name: f"SPEAKER_{label:02d}"
                            for name, label in speaker_map.items()},
        "diagnostics": {"duration_seconds": duration,
                        "segment_count": len(segments),
                        "sample_rate_hz": sample_rate},
    }
    review = {
        "suspicious_intervals": [
            {"start": start, "end": end, "reason": "transcript_overlap"}
            for start, end in overlaps
        ],
        "ordinary_candidates": [
            {"start": start, "end": end, "speaker": f"SPEAKER_{int(label):02d}"}
            for (start, end), label in zip(segments, labels)
            if not any(start < other_end and end > other_start
                       for other_start, other_end in overlaps)
        ],
    }
    return prepared, review


def _sample_review(review, rng, count):
    candidates = review.pop("ordinary_candidates")
    review["ordinary_intervals"] = sorted(
        rng.sample(candidates, min(count, len(candidates))),
        key=lambda row: (row["start"], row["end"], row["speaker"]))
    return review


def prepare(config_path=DEFAULT_CONFIG, acquisition_path=DEFAULT_ACQUISITION,
            output_dir=DEFAULT_OUTPUT, dataset="all", seed=None):
    config_path, acquisition_path = Path(config_path), Path(acquisition_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition.get("config_sha256") != sha256(config_path):
        raise ValueError("Acquisition manifest was created with a different config")
    seed = config["seed"] if seed is None else seed
    rng = random.Random(seed)
    reference_dir, audio_dir = output_dir / "reference_rttm", output_dir / "audio"
    reference_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    names = tuple(acquisition["datasets"]) if dataset == "all" else (dataset,)
    rows, reviews = [], []

    for name in names:
        if name not in acquisition["datasets"]:
            raise ValueError(f"Acquisition manifest does not contain {name}")
        source = acquisition["datasets"][name]
        spec = config["datasets"][name]
        humyn_rows = {}
        if name == "humyn":
            shard_paths = {item["remote_path"]: item.get("path")
                           for item in source.get("downloaded_files", [])}
            grouped = {}
            for record in source["records"]:
                grouped.setdefault(record["shard"], []).append(record["row_index_in_shard"])
            for shard, indices in grouped.items():
                if path := shard_paths.get(shard):
                    humyn_rows.update({(shard, index): row for index, row in
                                      _read_humyn_rows(path, indices).items()})

        for record in sorted(source["records"], key=lambda item: item["recording_id"]):
            base = {
                "dataset": name, "recording_id": record["recording_id"],
                "dataset_revision": spec["revision"],
            }
            try:
                if name == "datahive":
                    prepared, review = prepare_datahive_record(
                        record, spec, config["vad"], config["quality_control"], reference_dir)
                    reviews.append({**base, "recording_flags":
                                    prepared["diagnostics"]["review_flags"],
                                    **_sample_review(review, rng,
                                        config["quality_control"]["ordinary_review_intervals_per_recording"])})
                else:
                    if record.get("transcript_format") != "json_list":
                        raise ValueError("unsupported_annotation_format")
                    key = (record["shard"], record["row_index_in_shard"])
                    if key not in humyn_rows:
                        raise ValueError("audio_shard_not_cached")
                    prepared, review = prepare_humyn_record(
                        record, humyn_rows[key], spec, reference_dir, audio_dir)
                    reviews.append({**base, "recording_flags": [],
                                    **_sample_review(review, rng,
                                        config["quality_control"]["ordinary_review_intervals_per_recording"])})
            except ValueError as error:
                prepared = {
                    "eligible": False, "exclusion_reason": str(error),
                    "reference_provenance": spec["reference_provenance"],
                }
            rows.append({**base, **prepared})

    manifest_path = output_dir / "annotation_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    review_path = output_dir / "review_manifest.json"
    review_path.write_text(json.dumps({
        "schema_version": 1, "seed": seed,
        "human_verified": False, "recordings": reviews,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--acquisition-manifest", default=str(DEFAULT_ACQUISITION))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dataset", choices=("datahive", "humyn", "all"), default="all")
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = prepare(args.config, args.acquisition_manifest, args.output_dir,
                   args.dataset, args.seed)
    print(json.dumps({
        "recordings": len(rows),
        "eligible": sum(row["eligible"] for row in rows),
        "excluded": sum(not row["eligible"] for row in rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
