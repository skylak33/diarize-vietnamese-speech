"""Audit local Vietnamese single-speaker corpora without decoding audio payloads."""
import argparse
import csv
import json
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


PARQUET_INFO = {
    "vietnamese_asr": {
        "name": "Vietnamese ASR (VIVOS)",
        "license": "CC BY-NC-SA 4.0",
        "provenance": "Hugging Face mirror of VIVOS v1.00 (AILAB/VNUHCM University of Science, 2016)",
        "duration": {"reported_hours": 15.0, "kind": "dataset-card approximation"},
        "audio": {"sample_rate_hz": 16000, "container": "embedded in Parquet", "codec": "WAV/PCM per VIVOS source"},
    },
    "viespeaker": {
        "name": "VieSpeaker-Dataset-opus",
        "license": "CC BY-NC 4.0",
        "provenance": "giangndm compressed mirror of hustep-lab/VieSpeaker-Dataset; non-audio columns retained",
        "duration": {"local_hours": None, "reported_hours": 901.96,
                     "kind": "full-dataset card; local partial duration not measured"},
        "audio": {"sample_rate_hz": 16000, "container": "embedded in Parquet", "codec": "Opus, lossy"},
    },
}


def _parquet_shards(files):
    seen = defaultdict(set)
    totals = {}
    for path in files:
        match = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.parquet", path.name)
        if match:
            prefix, index, total = match.groups()
            seen[prefix].add(int(index))
            totals[prefix] = int(total)
    missing = {prefix: sorted(set(range(totals[prefix])) - indices)
               for prefix, indices in seen.items()}
    return missing


def audit_parquet_dataset(root, dataset, expected_shards, expected_rows, official_split,
                          manifest_write):
    """Scan Parquet metadata columns and emit row locators, never audio bytes."""
    root = Path(root).resolve()
    files = sorted((root / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {root / 'data'}")
    missing_by_split = _parquet_shards(files)
    speakers = Counter()
    speakers_by_split = defaultdict(set)
    rows_by_split = Counter()
    source_splits = Counter()
    empty = Counter()
    duplicate_keys = 0
    seen_keys = set()
    transcripts = Counter()
    transcript_splits = defaultdict(set)
    schemas = set()
    utterances = 0

    for path in files:
        parquet = pq.ParquetFile(path)
        schemas.add(tuple((field.name, str(field.type)) for field in parquet.schema_arrow))
        file_split = path.name.split("-", 1)[0] if dataset == "vietnamese_asr" else official_split
        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                row_group, columns=["speaker_id", "split", "path", "audio.path", "sentence"])
            for row_index, row in enumerate(table.to_pylist()):
                speaker = row.get("speaker_id")
                audio = row.get("audio") or {}
                audio_path = row.get("path") or audio.get("path")
                source_split = row.get("split")
                if not speaker:
                    empty["speaker_id"] += 1
                if not audio_path:
                    empty["audio_path"] += 1
                if not speaker or not audio_path:
                    continue
                key = (speaker, audio_path)
                if key in seen_keys:
                    duplicate_keys += 1
                seen_keys.add(key)
                utterances += 1
                speakers[speaker] += 1
                speakers_by_split[file_split].add(speaker)
                rows_by_split[file_split] += 1
                if source_split:
                    source_splits[source_split] += 1
                transcript = row.get("sentence")
                if transcript:
                    transcripts[transcript] += 1
                    transcript_splits[transcript].add(file_split)
                utterance_id = f"{dataset}::{path.name}:{row_group}:{row_index}"
                manifest_write({
                    "dataset": dataset,
                    "utterance_id": utterance_id,
                    "source_recording_id": utterance_id,
                    "split": file_split,
                    "source_split": source_split,
                    "speaker_id": f"{dataset}::{speaker}",
                    "source_speaker_id": speaker,
                    "source_type": "parquet_row",
                    "source_path": str(path.resolve()),
                    "row_group": row_group,
                    "row_index_in_group": row_index,
                    "audio_path": audio_path,
                    "source_locator": f"{path.resolve()}#row_group={row_group}&row_index={row_index}",
                    "content_sha256": None,
                    "content_hash_status": "deferred until the generator reads selected audio bytes",
                    "validation": "speaker/path metadata and Parquet locator",
                })

    missing_flat = sorted(index for values in missing_by_split.values() for index in values)
    incomplete = len(files) != expected_shards or utterances != expected_rows or bool(missing_flat)
    cross_split_transcripts = sum(len(value) > 1 for value in transcript_splits.values())
    issues = []
    if incomplete:
        issues.append(
            f"Local mirror is incomplete: {len(files)}/{expected_shards} shards and "
            f"{utterances}/{expected_rows} rows are present."
        )
    if empty:
        issues.append(f"Rows with missing required metadata: {dict(empty)}")
    if duplicate_keys:
        issues.append(f"Duplicate speaker/audio-path keys: {duplicate_keys}")
    if cross_split_transcripts:
        issues.append(f"Repeated transcript text crosses official splits in {cross_split_transcripts} groups.")
    info = PARQUET_INFO[dataset]
    return {
        "dataset": info["name"],
        "root": str(root),
        "format": "Hugging Face Parquet with embedded audio bytes",
        "local_completeness": {
            "complete": not incomplete,
            "found_shards": len(files),
            "expected_shards": expected_shards,
            "missing_shards": missing_flat,
            "missing_shards_by_split": missing_by_split,
            "found_rows": utterances,
            "expected_rows": expected_rows,
        },
        "reliable_speaker_labels": {
            "value": not empty.get("speaker_id"),
            "basis": "Explicit non-empty speaker_id for every locally present row; identity was not independently verified from audio.",
        },
        "official_split_semantics": {
            "manifest_splits": dict(rows_by_split),
            "parquet_split_column": dict(source_splits),
            "speaker_overlap": {
                f"{first}_{second}": len(speakers_by_split[first] & speakers_by_split[second])
                for index, first in enumerate(sorted(speakers_by_split))
                for second in sorted(speakers_by_split)[index + 1:]
            },
            "note": ("VIVOS train/test filenames are the official ASR splits."
                     if dataset == "vietnamese_asr" else
                     "The Hugging Face dataset exposes one split named 'part'; its retained source split column is 'train'."),
        },
        "license": info["license"],
        "provenance": info["provenance"],
        "counts": {
            "utterances": utterances,
            "speakers": len(speakers),
            "speakers_by_split": {key: len(value) for key, value in speakers_by_split.items()},
            "parquet_files": len(files),
            "storage_bytes": sum(path.stat().st_size for path in files),
        },
        "duration": info["duration"],
        "schema": [dict(columns=[{"name": name, "type": kind} for name, kind in schema])
                   for schema in sorted(schemas)],
        "audio_metadata": info["audio"],
        "duplicates": {
            "speaker_audio_path_extra_rows": duplicate_keys,
            "transcript_extra_rows": sum(value - 1 for value in transcripts.values() if value > 1),
            "transcript_groups_crossing_splits": cross_split_transcripts,
        },
        "issues": issues,
        "usable": {
            "value": not empty and not duplicate_keys,
            "manifest_rows": utterances,
            "scope": "all locally present rows" if incomplete else "complete local snapshot",
        },
        "checks_not_performed": ["audio payload decode", "audio-content hashing", "duration scan"],
        "manifest_hash_policy": "Parquet content_sha256 is null; the generator must hash decoded selected audio bytes.",
    }


def audit_dnu(root, manifest_write):
    """Scan DNU CSV metadata and WAV headers without reading sample frames."""
    root = Path(root).resolve()
    splits = ("train", "validation", "test")
    counts = Counter()
    durations = Counter()
    speakers = defaultdict(set)
    ids, paths, checksums = Counter(), Counter(), Counter()
    text_splits = defaultdict(set)
    text_counts = Counter()
    header_mismatches = Counter()
    missing_audio = invalid_wav = invalid_rows = 0
    fields = None
    storage_bytes = 0

    for split in splits:
        metadata = root / "data" / split / "metadata.csv"
        with metadata.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = fields or reader.fieldnames
            for row in reader:
                try:
                    audio = (metadata.parent / row["file_name"]).resolve()
                    duration = float(row["duration"])
                    sample_rate = int(row["sample_rate"])
                    channels = int(row["num_channels"])
                except (KeyError, TypeError, ValueError):
                    invalid_rows += 1
                    continue
                relative_audio = Path(row.get("file_name", ""))
                checksum = row.get("checksum_sha256", "")
                if (row.get("split") != split or not row.get("speaker_id") or
                        not row.get("audio_id") or row["speaker_id"] not in relative_audio.parts or
                        not re.fullmatch(r"[0-9a-f]{64}", checksum)):
                    invalid_rows += 1
                    continue
                if not audio.is_relative_to(metadata.parent.resolve()):
                    invalid_rows += 1
                    continue
                if not audio.is_file():
                    missing_audio += 1
                    continue
                try:
                    with wave.open(str(audio), "rb") as handle:
                        actual_rate = handle.getframerate()
                        actual_channels = handle.getnchannels()
                        sample_width = handle.getsampwidth()
                        actual_duration = handle.getnframes() / actual_rate
                except (EOFError, OSError, wave.Error):
                    invalid_wav += 1
                    continue
                if actual_rate != sample_rate:
                    header_mismatches["sample_rate"] += 1
                if actual_channels != channels:
                    header_mismatches["channels"] += 1
                if sample_width != 2:
                    header_mismatches["sample_width"] += 1
                if abs(actual_duration - duration) > .002:
                    header_mismatches["duration"] += 1
                counts[split] += 1
                durations[split] += duration
                speakers[split].add(row["speaker_id"])
                ids[row["audio_id"]] += 1
                paths[(split, row["file_name"])] += 1
                checksums[row.get("checksum_sha256")] += 1
                text = row.get("text_normalized", "")
                text_counts[text] += 1
                text_splits[text].add(split)
                storage_bytes += audio.stat().st_size
                utterance_id = f"dnu::{row['audio_id']}"
                manifest_write({
                    "dataset": "dnu_vispeech",
                    "utterance_id": utterance_id,
                    "source_recording_id": utterance_id,
                    "split": split,
                    "speaker_id": f"dnu::{row['speaker_id']}",
                    "source_speaker_id": row["speaker_id"],
                    "source_type": "audio_file",
                    "source_path": str(audio),
                    "source_locator": str(audio),
                    "duration_seconds": duration,
                    "sample_rate_hz": sample_rate,
                    "num_channels": channels,
                    "checksum_sha256": row.get("checksum_sha256"),
                    "content_sha256": row.get("checksum_sha256"),
                    "content_hash_status": "declared release checksum; contents not recomputed by this bounded audit",
                    "validation": "metadata and WAV header",
                })

    duplicate_ids = sum(value - 1 for value in ids.values() if value > 1)
    duplicate_paths = sum(value - 1 for value in paths.values() if value > 1)
    duplicate_checksums = sum(value - 1 for value in checksums.values() if value > 1)
    cross_split_text = sum(len(value) > 1 for value in text_splits.values())
    issues = []
    if invalid_rows:
        issues.append(f"Invalid metadata rows: {invalid_rows}")
    if missing_audio:
        issues.append(f"Missing audio files: {missing_audio}")
    if invalid_wav:
        issues.append(f"Invalid WAV headers: {invalid_wav}")
    if header_mismatches:
        issues.append(f"WAV/metadata mismatches: {dict(header_mismatches)}")
    if duplicate_ids or duplicate_paths or duplicate_checksums:
        issues.append("Duplicate audio identifiers, paths, or declared checksums were found.")
    total = sum(counts.values())
    all_speakers = set().union(*speakers.values())
    return {
        "dataset": "DNU-ViSpeech",
        "root": str(root),
        "format": "AudioFolder CSV metadata plus WAV files",
        "local_completeness": {
            "complete": total == 31070 and not missing_audio and not invalid_wav,
            "found_rows": total,
            "expected_rows": 31070,
            "found_audio_files": total,
            "expected_audio_files": 31070,
        },
        "reliable_speaker_labels": {
            "value": not invalid_rows and len(all_speakers) == 26,
            "basis": "Explicit anonymized speaker_id, consistent path prefix, and speaker-disjoint official splits; identity was not independently verified from audio.",
        },
        "official_split_semantics": {
            "rows": dict(counts),
            "speakers": {key: len(value) for key, value in speakers.items()},
            "speaker_overlap": {
                "train_validation": len(speakers["train"] & speakers["validation"]),
                "train_test": len(speakers["train"] & speakers["test"]),
                "validation_test": len(speakers["validation"] & speakers["test"]),
            },
            "note": "Official speaker-disjoint train/validation/test ASR splits; test also carries a recording-configuration shift.",
        },
        "license": "CC BY-NC 4.0",
        "provenance": "DNU-ViSpeech v1.1 public release; participant read speech with owner-approved release dated 2026-09-05",
        "counts": {
            "utterances": total,
            "speakers": len(all_speakers),
            "wav_files": total,
            "storage_bytes": storage_bytes,
        },
        "duration": {
            "metadata_hours": sum(durations.values()) / 3600,
            "hours_by_split": {key: value / 3600 for key, value in durations.items()},
        },
        "schema": {"metadata_csv_columns": fields},
        "audio_metadata": {
            "container": "WAV", "codec": "PCM signed 16-bit", "sample_rate_hz": 16000,
            "num_channels": 1, "headers_checked": total,
            "header_mismatches": dict(header_mismatches),
        },
        "duplicates": {
            "audio_id_extra_rows": duplicate_ids,
            "path_extra_rows": duplicate_paths,
            "declared_checksum_extra_rows": duplicate_checksums,
            "normalized_text_extra_rows_within_splits": sum(
                value - 1 for text, value in text_counts.items()
                if value > 1 and len(text_splits[text]) == 1),
            "normalized_text_groups_crossing_splits": cross_split_text,
        },
        "issues": issues,
        "usable": {"value": not issues, "manifest_rows": total, "scope": "complete local snapshot"},
        "checks_not_performed": ["full WAV checksum recomputation", "full audio decode"],
        "manifest_hash_policy": "content_sha256 is the release-declared WAV checksum; this audit does not recompute it.",
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vietnamese-asr", type=Path, required=True)
    parser.add_argument("--viespeaker", type=Path, required=True)
    parser.add_argument("--dnu-vispeech", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/synthetic_vietnamese_benchmark/dataset_audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "utterance_manifest.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="\n") as manifest:
        write_manifest = lambda row: manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
        audits = {
            "vietnamese_asr": audit_parquet_dataset(
                args.vietnamese_asr, "vietnamese_asr", 4, 12420, None, write_manifest),
            "viespeaker": audit_parquet_dataset(
                args.viespeaker, "viespeaker", 110, 365844, "part", write_manifest),
            "dnu_vispeech": audit_dnu(args.dnu_vispeech, write_manifest),
        }
    temporary_manifest.replace(manifest_path)
    for name, audit in audits.items():
        _write_json(args.output / f"{name}.json", audit)
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["dataset", "usable", "complete", "utterances", "speakers",
                   "duration_hours", "storage_bytes", "format", "license", "issues"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, audit in audits.items():
            duration = audit["duration"].get(
                "metadata_hours", audit["duration"].get("local_hours", audit["duration"].get("reported_hours")))
            writer.writerow({
                "dataset": name, "usable": audit["usable"]["value"],
                "complete": audit["local_completeness"]["complete"],
                "utterances": audit["counts"]["utterances"],
                "speakers": audit["counts"]["speakers"], "duration_hours": duration,
                "storage_bytes": audit["counts"]["storage_bytes"], "format": audit["format"],
                "license": audit["license"], "issues": " | ".join(audit["issues"]),
            })
    print(json.dumps({"output": str(args.output.resolve()),
                      "manifest_rows": sum(audit["usable"]["manifest_rows"] for audit in audits.values())},
                     indent=2))


if __name__ == "__main__":
    main()
