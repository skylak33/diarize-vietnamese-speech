"""Validate and fingerprint an independent diarization development corpus."""

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path


SPLITS = {"development", "validation"}
REQUIRED_DATASET_FIELDS = {
    "name",
    "source",
    "license",
    "revision",
    "language",
    "domain",
    "annotation_format",
    "project_usage",
    "global_speaker_ids",
}
REQUIRED_RECORDING_FIELDS = {
    "recording_id",
    "split",
    "audio_path",
    "rttm_path",
    "source_id",
    "conversation_id",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_strings(row, fields, label):
    missing = sorted(field for field in fields if field not in row)
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")
    string_fields = fields - {"global_speaker_ids"}
    empty = sorted(field for field in string_fields if not isinstance(row[field], str) or not row[field].strip())
    if empty:
        raise ValueError(f"{label} has empty/non-string fields: {empty}")


def _resolve(root, relative, label):
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative to dataset root: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes dataset root: {relative}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _wav_duration(path):
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getframerate() <= 0:
                raise ValueError(f"Invalid WAV sample rate: {path}")
            return audio.getnframes() / audio.getframerate()
    except wave.Error as error:
        raise ValueError(f"Audio must be a readable WAV file: {path}") from error


def _read_rttm(path, recording_id, audio_duration):
    speakers = set()
    turns = 0
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"{path}:{number}: expected a 10-field SPEAKER RTTM row")
        if fields[1] != recording_id:
            raise ValueError(f"{path}:{number}: RTTM recording ID {fields[1]!r} != {recording_id!r}")
        try:
            start, duration = float(fields[3]), float(fields[4])
        except ValueError as error:
            raise ValueError(f"{path}:{number}: invalid RTTM time") from error
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
            raise ValueError(f"{path}:{number}: RTTM times must be finite, start >= 0, duration > 0")
        if start + duration > audio_duration + 0.02:
            raise ValueError(f"{path}:{number}: RTTM turn exceeds WAV duration")
        if not fields[7] or fields[7] == "<NA>":
            raise ValueError(f"{path}:{number}: RTTM speaker ID is missing")
        speakers.add(fields[7])
        turns += 1
    if not turns:
        raise ValueError(f"RTTM has no speaker turns: {path}")
    return speakers, turns


def _cross_split_overlap(records, field):
    values = {split: {record[field] for record in records if record["split"] == split} for split in SPLITS}
    return values["development"] & values["validation"]


def prepare_manifest(spec_path):
    """Return a validated, hash-addressed manifest without copying source data."""
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("dataset"), dict):
        raise ValueError("Spec requires a dataset object")
    dataset = spec["dataset"]
    _required_strings(dataset, REQUIRED_DATASET_FIELDS, "dataset")
    if not isinstance(dataset["global_speaker_ids"], bool):
        raise ValueError("dataset.global_speaker_ids must be true or false")
    if dataset["annotation_format"] != "RTTM":
        raise ValueError("Only official or provenance-tracked RTTM annotations are supported")
    if dataset["project_usage"] != "independent_development":
        raise ValueError("Dataset is not declared independent; reused test and verification data are refused")

    root_value = spec.get("root", ".")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError("root must be a non-empty path string")
    root = (spec_path.parent / root_value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = spec.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Spec requires a non-empty recordings list")

    prepared = []
    recording_ids = set()
    audio_hashes = {}
    reference_hashes = {}
    speakers_by_split = {split: set() for split in SPLITS}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"recordings[{index}] must be an object")
        _required_strings(row, REQUIRED_RECORDING_FIELDS, f"recordings[{index}]")
        recording_id = row["recording_id"]
        if row["split"] not in SPLITS:
            raise ValueError(f"{recording_id}: split must be development or validation")
        if recording_id in recording_ids:
            raise ValueError(f"Duplicate recording_id: {recording_id}")
        recording_ids.add(recording_id)
        audio_path = _resolve(root, row["audio_path"], "audio_path")
        rttm_path = _resolve(root, row["rttm_path"], "rttm_path")
        duration = _wav_duration(audio_path)
        speakers, turns = _read_rttm(rttm_path, recording_id, duration)
        audio_digest, reference_digest = sha256(audio_path), sha256(rttm_path)
        if audio_digest in audio_hashes:
            raise ValueError(f"Duplicate audio content: {audio_hashes[audio_digest]}, {recording_id}")
        if reference_digest in reference_hashes:
            raise ValueError(f"Duplicate RTTM content: {reference_hashes[reference_digest]}, {recording_id}")
        audio_hashes[audio_digest] = recording_id
        reference_hashes[reference_digest] = recording_id
        speakers_by_split[row["split"]].update(speakers)
        prepared.append({
            **{field: row[field] for field in sorted(REQUIRED_RECORDING_FIELDS)},
            "audio_sha256": audio_digest,
            "rttm_sha256": reference_digest,
            "duration_seconds": round(duration, 6),
            "speaker_count": len(speakers),
            "turn_count": turns,
        })

    missing_splits = sorted(SPLITS - {row["split"] for row in prepared})
    if missing_splits:
        raise ValueError(f"Both development and validation are required; missing {missing_splits}")
    if max(row["speaker_count"] for row in prepared) < 2:
        raise ValueError("Corpus has no multi-speaker recording")
    for field in ("source_id", "conversation_id"):
        overlap = _cross_split_overlap(prepared, field)
        if overlap:
            raise ValueError(f"Cross-split {field} leakage: {sorted(overlap)}")
    speaker_overlap = speakers_by_split["development"] & speakers_by_split["validation"]
    if dataset["global_speaker_ids"] and speaker_overlap:
        raise ValueError(f"Cross-split speaker leakage: {sorted(speaker_overlap)}")

    by_split = {}
    for split in sorted(SPLITS):
        split_rows = [row for row in prepared if row["split"] == split]
        by_split[split] = {
            "recording_count": len(split_rows),
            "duration_seconds": round(sum(row["duration_seconds"] for row in split_rows), 6),
            "speaker_count": len(speakers_by_split[split]) if dataset["global_speaker_ids"] else None,
            "speaker_instances": sum(row["speaker_count"] for row in split_rows),
        }
    return {
        "schema_version": 1,
        "dataset": {field: dataset[field] for field in sorted(REQUIRED_DATASET_FIELDS)},
        "root": root_value,
        "recordings": sorted(prepared, key=lambda row: (row["split"], row["recording_id"])),
        "summary": {
            "recording_count": len(prepared),
            "duration_seconds": round(sum(row["duration_seconds"] for row in prepared), 6),
            "speaker_count": len(set().union(*speakers_by_split.values())) if dataset["global_speaker_ids"] else None,
            "speaker_instances": sum(row["speaker_count"] for row in prepared),
            "by_split": by_split,
        },
        "checks": {
            "audio_content_unique": True,
            "reference_content_unique": True,
            "source_disjoint": True,
            "conversation_disjoint": True,
            "speaker_disjoint": True if dataset["global_speaker_ids"] else None,
            "speaker_disjoint_note": None if dataset["global_speaker_ids"] else "Not checkable: corpus does not declare global speaker identities.",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", nargs="?", help="JSON inventory for an independent diarization corpus")
    parser.add_argument("--output", help="Write the validated manifest here; stdout if omitted")
    args = parser.parse_args(argv)
    if not args.spec:
        parser.error("no independent corpus spec supplied; refusing to use local ViYT-Diar or verification data")
    manifest = prepare_manifest(args.spec)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
