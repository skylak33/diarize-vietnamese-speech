"""Audit stereo channel-derived silver RTTM without changing the reference."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.der import load_rttm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "results/natural_vietnamese_benchmark/datahive_references/annotation_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "results/partition_quality_stability/silver_reference_audit"
DEFAULT_HYPOTHESES = ROOT / "results/natural_vietnamese_benchmark/full_evaluation/hypothesis_rttm/full_pipeline"
SPEAKERS = ("SPEAKER_00", "SPEAKER_01")


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) if len(values) else 0.0


def _correlation(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def validate_stereo(audio, sample_rate):
    """Fail closed: an interleaved stereo file guarantees equal channel lengths."""
    if sample_rate <= 0 or audio.ndim != 2 or audio.shape[1] != 2 or not len(audio):
        raise ValueError("expected_nonempty_stereo_audio")
    if not np.isfinite(audio).all():
        raise ValueError("expected_finite_audio")
    return {"channels": 2, "frames_per_channel": len(audio), "sample_rate_hz": sample_rate,
            "sample_alignment": "shared_interleaved_frame_clock"}


def validate_reference(segments, duration):
    unknown = {speaker for _, _, speaker in segments} - set(SPEAKERS)
    if unknown:
        raise ValueError(f"unexpected_reference_speakers: {sorted(unknown)}")
    for start, end, _ in segments:
        if not math.isfinite(start + end) or start < 0 or start >= end or end > duration + 0.002:
            raise ValueError("invalid_reference_timestamp")


def _mask(segments, frames, sample_rate, speaker=None):
    mask = np.zeros(frames, dtype=bool)
    for start, end, label in segments:
        if speaker is None or label == speaker:
            mask[int(start * sample_rate):min(frames, math.ceil(end * sample_rate))] = True
    return mask


def simultaneous_intervals(segments, minimum_seconds=0.5):
    left = sorted((start, end) for start, end, speaker in segments if speaker == SPEAKERS[0])
    right = sorted((start, end) for start, end, speaker in segments if speaker == SPEAKERS[1])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end - start >= minimum_seconds:
            result.append({"start": start, "end": end, "reason": "simultaneous_reference_activity",
                           "score": end - start})
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def bounded_lag_correlation(audio, sample_rate, max_lag_seconds=0.5, envelope_hz=50):
    """Correlate fixed-rate RMS envelopes; this is an echo indicator, not proof."""
    frame = max(1, sample_rate // envelope_hz)
    usable = len(audio) // frame * frame
    envelope = np.sqrt(np.mean(np.square(audio[:usable].reshape(-1, frame, 2), dtype=np.float64), axis=1))
    max_lag = max(0, int(round(max_lag_seconds * envelope_hz)))
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            value = _correlation(envelope[-lag:, 0], envelope[:lag, 1])
        elif lag > 0:
            value = _correlation(envelope[:-lag, 0], envelope[lag:, 1])
        else:
            value = _correlation(envelope[:, 0], envelope[:, 1])
        candidates.append((abs(value), value, lag))
    _, value, lag = max(candidates, default=(0.0, 0.0, 0))
    return {"max_abs_lagged_envelope_correlation": abs(value),
            "lagged_envelope_correlation": value,
            "lag_seconds": lag / envelope_hz,
            "envelope_rate_hz": envelope_hz}


def interval_diagnostics(audio, sample_rate, segments, window_seconds=2.0,
                         leakage_ratio=0.35, high_confidence_ratio=0.15,
                         minimum_source_rms=0.005):
    masks = [_mask(segments, len(audio), sample_rate, speaker) for speaker in SPEAKERS]
    exclusive = [masks[0] & ~masks[1], masks[1] & ~masks[0]]
    leakage = []
    for source in range(2):
        own = _rms(audio[exclusive[source], source])
        other = _rms(audio[exclusive[source], 1 - source])
        leakage.append(other / own if own else None)

    suspicious, confident = [], []
    window = max(1, round(window_seconds * sample_rate))
    for start_frame in range(0, len(audio), window):
        end_frame = min(len(audio), start_frame + window)
        active = [float(np.mean(mask[start_frame:end_frame])) for mask in masks]
        if max(active) < 0.8 or min(active) > 0.05:
            continue
        source = int(active[1] > active[0])
        source_rms = _rms(audio[start_frame:end_frame, source])
        other_rms = _rms(audio[start_frame:end_frame, 1 - source])
        ratio = other_rms / source_rms if source_rms else math.inf
        row = {"start": start_frame / sample_rate, "end": end_frame / sample_rate,
               "source_channel": source, "source_rms": source_rms,
               "other_rms": other_rms, "other_to_source_rms_ratio": ratio}
        if source_rms >= minimum_source_rms and ratio >= leakage_ratio:
            suspicious.append({**row, "reason": "exclusive_speech_energy_leakage", "score": ratio})
        if source_rms >= minimum_source_rms and ratio <= high_confidence_ratio:
            confident.append(row)
    return leakage, suspicious, confident


def _stream_diagnostics(audio_path, segments, window_seconds=2.0,
                        leakage_ratio=0.35, high_confidence_ratio=0.15,
                        minimum_source_rms=0.005, envelope_hz=50):
    info = sf.info(audio_path)
    validate_stereo(np.zeros((1, info.channels), dtype=np.float32), info.samplerate)
    sample_rate, frames = info.samplerate, info.frames
    by_speaker = [[(start, end) for start, end, label in segments if label == speaker]
                  for speaker in SPEAKERS]
    exclusive_squares = np.zeros((2, 2), dtype=np.float64)
    exclusive_counts = np.zeros(2, dtype=np.int64)
    sums = np.zeros(5, dtype=np.float64)  # x, y, x*x, y*y, x*y
    envelopes, suspicious, confident = [], [], []
    blocksize = max(1, round(window_seconds * sample_rate))
    envelope_frame = max(1, sample_rate // envelope_hz)
    with sf.SoundFile(audio_path) as handle:
        for start_frame in range(0, frames, blocksize):
            block = handle.read(blocksize, dtype="float32", always_2d=True)
            if not len(block):
                break
            validate_stereo(block, sample_rate)
            end_frame = start_frame + len(block)
            x, y = block[:, 0].astype(np.float64), block[:, 1].astype(np.float64)
            sums += (x.sum(), y.sum(), x @ x, y @ y, x @ y)
            usable = len(block) // envelope_frame * envelope_frame
            if usable:
                envelopes.append(np.sqrt(np.mean(np.square(
                    block[:usable].reshape(-1, envelope_frame, 2), dtype=np.float64), axis=1)))
            masks = []
            for own_segments in by_speaker:
                mask = np.zeros(len(block), dtype=bool)
                for start, end in own_segments:
                    left = max(0, int(start * sample_rate) - start_frame)
                    right = min(len(block), math.ceil(end * sample_rate) - start_frame)
                    if left < right:
                        mask[left:right] = True
                masks.append(mask)
            active = [float(np.mean(mask)) for mask in masks]
            for source in range(2):
                selected = masks[source] & ~masks[1 - source]
                exclusive_counts[source] += selected.sum()
                if selected.any():
                    exclusive_squares[source] += np.square(block[selected], dtype=np.float64).sum(axis=0)
            if max(active) < 0.8 or min(active) > 0.05:
                continue
            source = int(active[1] > active[0])
            source_rms, other_rms = _rms(block[:, source]), _rms(block[:, 1 - source])
            ratio = other_rms / source_rms if source_rms else math.inf
            row = {"start": start_frame / sample_rate, "end": end_frame / sample_rate,
                   "source_channel": source, "source_rms": source_rms, "other_rms": other_rms,
                   "other_to_source_rms_ratio": ratio}
            if source_rms >= minimum_source_rms and ratio >= leakage_ratio:
                suspicious.append({**row, "reason": "exclusive_speech_energy_leakage", "score": ratio})
            if source_rms >= minimum_source_rms and ratio <= high_confidence_ratio:
                confident.append(row)
    n = frames
    denominator = math.sqrt(max(0.0, n * sums[2] - sums[0] ** 2) *
                            max(0.0, n * sums[3] - sums[1] ** 2))
    zero_lag = (n * sums[4] - sums[0] * sums[1]) / denominator if denominator else 0.0
    envelope = np.concatenate(envelopes) if envelopes else np.empty((0, 2))
    max_lag = max(0, int(round(0.5 * envelope_hz)))
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            value = _correlation(envelope[-lag:, 0], envelope[:lag, 1])
        elif lag > 0:
            value = _correlation(envelope[:-lag, 0], envelope[lag:, 1])
        else:
            value = _correlation(envelope[:, 0], envelope[:, 1])
        candidates.append((abs(value), value, lag))
    _, lagged_value, lag = max(candidates, default=(0.0, 0.0, 0))
    leakage = []
    for source in range(2):
        count = exclusive_counts[source]
        own = math.sqrt(exclusive_squares[source, source] / count) if count else 0.0
        other = math.sqrt(exclusive_squares[source, 1 - source] / count) if count else 0.0
        leakage.append(other / own if own else None)
    return zero_lag, {"max_abs_lagged_envelope_correlation": abs(lagged_value),
                      "lagged_envelope_correlation": lagged_value,
                      "lag_seconds": lag / envelope_hz, "envelope_rate_hz": envelope_hz}, \
        leakage, suspicious, confident


def audit_recording(recording_id, audio_path, rttm_path):
    info = sf.info(audio_path)
    sample_rate, frames = info.samplerate, info.frames
    alignment = validate_stereo(np.zeros((1, info.channels), dtype=np.float32), sample_rate)
    alignment["frames_per_channel"] = frames
    references = load_rttm(rttm_path)
    if set(references) != {recording_id}:
        raise ValueError("reference_recording_id_mismatch")
    segments = references[recording_id]
    duration = frames / sample_rate
    validate_reference(segments, duration)
    overlap = simultaneous_intervals(segments)
    zero_lag, lagged, leakage, leakage_intervals, confident = _stream_diagnostics(
        audio_path, segments)
    lagged["lag_at_search_boundary"] = abs(lagged["lag_seconds"]) >= 0.5 - 1e-12
    suspicious = overlap + leakage_intervals
    return ({"recording_id": recording_id, "duration_seconds": duration, **alignment,
             "zero_lag_waveform_correlation": zero_lag, **lagged,
             "simultaneous_activity_seconds": sum(row["end"] - row["start"] for row in overlap),
             "simultaneous_activity_interval_count": len(overlap),
             "exclusive_speech_leakage_rms_ratio_0_to_1": leakage[0],
             "exclusive_speech_leakage_rms_ratio_1_to_0": leakage[1],
             "suspicious_interval_count": len(suspicious),
             "high_confidence_interval_count": len(confident)}, suspicious, confident)


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _interval_labels(segments, start, end):
    return sorted({speaker for left, right, speaker in segments if left < end and right > start})


def run(annotation_manifest=DEFAULT_ANNOTATIONS, output_dir=DEFAULT_OUTPUT,
        write_excerpts=False, hypothesis_root=DEFAULT_HYPOTHESES):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(annotation_manifest).read_text(encoding="utf-8").splitlines()]
    diagnostics, suspicious_rows, confidence_rows, review = [], [], [], []
    excerpts = output_dir / "review_excerpts"
    if write_excerpts:
        excerpts.mkdir(exist_ok=True)
    hypothesis_root = Path(hypothesis_root) if hypothesis_root else None
    hypotheses = ({path.name: load_rttm(path, allow_empty=True)
                   for path in sorted(hypothesis_root.iterdir()) if path.is_dir()}
                  if hypothesis_root and hypothesis_root.is_dir() else {})
    for record in records:
        if not record.get("eligible") or record.get("dataset") != "datahive":
            continue
        recording_id = record["recording_id"]
        summary, suspicious, confident = audit_recording(
            recording_id, record["audio_path"], record["rttm_path"])
        reference_segments = load_rttm(record["rttm_path"])[recording_id]
        diagnostics.append(summary)
        suspicious_rows.extend({"recording_id": recording_id, **row} for row in suspicious)
        confidence_rows.extend({"recording_id": recording_id, **row} for row in confident)
        for index, row in enumerate(sorted(suspicious, key=lambda item: -item["score"])[:3]):
            start, end = max(0.0, row["start"] - 1.0), min(summary["duration_seconds"], row["end"] + 1.0)
            excerpt_path = excerpts / f"{recording_id}_{index + 1:02d}.wav"
            if write_excerpts:
                with sf.SoundFile(record["audio_path"]) as handle:
                    handle.seek(int(start * handle.samplerate))
                    audio = handle.read(math.ceil((end - start) * handle.samplerate),
                                        dtype="float32", always_2d=True)
                    sf.write(excerpt_path, audio, handle.samplerate)
            review.append({"recording_id": recording_id, "start": start, "end": end,
                           "reason": row["reason"],
                           "reference_labels": _interval_labels(reference_segments, start, end),
                           "predicted_labels": {
                               method: _interval_labels(data.get(recording_id, []), start, end)
                               for method, data in hypotheses.items()},
                           "channel_activity": row,
                           "audio_excerpt_path": str(excerpt_path.resolve()),
                           "excerpt_written": write_excerpts})
    _write_csv(output_dir / "channel_diagnostics.csv", diagnostics)
    _write_csv(output_dir / "suspicious_intervals.csv", suspicious_rows)
    _write_csv(output_dir / "high_confidence_intervals.csv", confidence_rows)
    (output_dir / "human_review_manifest.json").write_text(json.dumps({
        "schema_version": 1, "human_verified": False,
        "rule": {"window_seconds": 2.0, "single_channel_activity_min_fraction": 0.8,
                 "other_channel_activity_max_fraction": 0.05,
                 "minimum_source_rms": 0.005, "other_to_source_rms_ratio_max": 0.15},
        "items": review,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-manifest", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--write-excerpts", action="store_true")
    parser.add_argument("--hypothesis-root", default=str(DEFAULT_HYPOTHESES))
    args = parser.parse_args(argv)
    rows = run(args.annotation_manifest, args.output_dir, args.write_excerpts,
               args.hypothesis_root)
    print(json.dumps({"recordings": len(rows), "output_dir": str(Path(args.output_dir).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
