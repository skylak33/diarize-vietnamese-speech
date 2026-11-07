"""Inspect and cache Vietnamese-only natural diarization benchmark data."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.provenance import sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/natural_vietnamese_benchmark.json"
DEFAULT_OUTPUT = ROOT / "results/natural_vietnamese_benchmark/acquisition_manifest.json"
DATASETS = ("datahive", "humyn")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or any(name not in datasets for name in DATASETS):
        raise ValueError(f"Config must define datasets: {DATASETS}")
    for name in DATASETS:
        missing = [key for key in ("repo_id", "revision", "license",
                                   "language_field", "language_value")
                   if not datasets[name].get(key)]
        if missing:
            raise ValueError(f"datasets.{name} is missing {missing}")
    if not datasets["datahive"].get("metadata_path"):
        raise ValueError("datasets.datahive.metadata_path is required")
    if not datasets["humyn"].get("required_parquet_shards"):
        raise ValueError("datasets.humyn.required_parquet_shards is required")
    return config


def _hub_download(spec, filename, cache_dir, offline):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
        filename=filename, cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=offline,
    )).resolve()


def _hub_files(spec, offline):
    if offline:
        return {}
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(
        spec["repo_id"], revision=spec["revision"], files_metadata=True)
    if info.sha != spec["revision"]:
        raise RuntimeError(
            f"{spec['repo_id']}: resolved revision {info.sha} != {spec['revision']}")
    return {
        item.rfilename: {
            "size": item.size,
            "blob_id": item.blob_id,
            "sha256": item.lfs.sha256 if item.lfs else None,
        }
        for item in info.siblings
    }


def verify_file(path, expected=None):
    path = Path(path)
    expected = expected or {}
    size = path.stat().st_size
    if expected.get("size") is not None and size != expected["size"]:
        raise ValueError(f"{path}: size {size} != expected {expected['size']}")
    digest = sha256(path)
    if expected.get("sha256") and digest != expected["sha256"]:
        raise ValueError(f"{path}: SHA256 {digest} != expected {expected['sha256']}")
    return {"path": str(path.resolve()), "size": size, "sha256": digest}


def parse_datahive_metadata(path, spec, max_recordings=None):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if row.get(spec["language_field"]) != spec["language_value"]:
            continue
        filename = row.get("file_name")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"{path}:{number}: unsafe or missing file_name")
        duration = row.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"{path}:{number}: invalid duration_seconds")
        records.append({
            "recording_id": Path(filename).stem,
            "file_name": filename,
            "remote_path": f"preview/{filename}",
            "duration_seconds": float(duration),
            "language": row[spec["language_field"]],
        })
    records.sort(key=lambda row: row["recording_id"])
    if not records:
        raise ValueError(f"{path}: no {spec['language_value']} records")
    if len({row["recording_id"] for row in records}) != len(records):
        raise ValueError(f"{path}: duplicate Vietnamese recording_id")
    return records[:max_recordings] if max_recordings else records


def validate_datahive_audio(path):
    import soundfile

    try:
        info = soundfile.info(path)
    except RuntimeError as error:
        raise ValueError(f"Unreadable audio: {path}") from error
    if info.channels != 2 or info.samplerate != 48000 or info.frames <= 0:
        raise ValueError(
            f"{path}: expected non-empty 48 kHz stereo, got "
            f"{info.samplerate} Hz/{info.channels} channels/{info.frames} frames")
    return {
        "channels": info.channels, "sample_rate_hz": info.samplerate,
        "frames": info.frames, "duration_seconds": info.duration,
        "format": info.format, "subtype": info.subtype,
    }


def acquire_datahive(spec, cache_dir, offline, download, max_recordings,
                     previous=None):
    metadata_path = _hub_download(spec, spec["metadata_path"], cache_dir, offline)
    records = parse_datahive_metadata(metadata_path, spec, max_recordings)
    files = _hub_files(spec, offline)
    previous_records = {
        row["remote_path"]: row for row in (previous or {}).get("records", [])
    }
    downloaded = []
    for record in records:
        expected = files.get(record["remote_path"])
        if not offline and expected is None:
            raise ValueError(f"Hub revision is missing {record['remote_path']}")
        if expected:
            record["expected_size"] = expected["size"]
            record["expected_sha256"] = expected["sha256"]
        elif record["remote_path"] in previous_records:
            record["expected_size"] = previous_records[record["remote_path"]].get("expected_size")
            record["expected_sha256"] = previous_records[record["remote_path"]].get("expected_sha256")
        if download:
            local = _hub_download(spec, record["remote_path"], cache_dir, offline)
            artifact = verify_file(local, {
                "size": record.get("expected_size"),
                "sha256": record.get("expected_sha256"),
            })
            record["audio"] = validate_datahive_audio(local)
            record["local_path"] = artifact["path"]
            downloaded.append({"remote_path": record["remote_path"], **artifact})
    return {
        "repo_id": spec["repo_id"], "revision": spec["revision"],
        "license": spec["license"], "metadata": verify_file(metadata_path),
        "record_count": len(records),
        "duration_seconds": sum(row["duration_seconds"] for row in records),
        "records": records, "downloaded_files": downloaded,
    }


def scan_humyn_shard(source, shard, spec):
    parquet = pq.ParquetFile(source)
    required = {"language", "file_name", "audio", "transcript_json", "type"}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{shard}: missing Parquet fields {missing}")
    audio_type = parquet.schema_arrow.field("audio").type
    if (not pa.types.is_struct(audio_type)
            or set(audio_type.names) != {"bytes", "path"}):
        raise ValueError(f"{shard}: audio must be a bytes/path struct")
    rows = parquet.read(
        columns=["language", "file_name", "transcript_json", "type"]).to_pylist()
    selected = []
    for index, row in enumerate(rows):
        if row[spec["language_field"]] != spec["language_value"]:
            continue
        filename = row["file_name"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"{shard}: row {index} has unsafe or missing file_name")
        if row["type"] not in {"single-speaker", "multi-speaker"}:
            raise ValueError(f"{shard}: row {index} has unsupported type {row['type']!r}")
        if not isinstance(row["transcript_json"], str) or not row["transcript_json"].strip():
            raise ValueError(f"{shard}: row {index} has an empty transcript")
        try:
            transcript = json.loads(row["transcript_json"])
            transcript_format = "json_list" if isinstance(transcript, list) else "json_object"
        except (json.JSONDecodeError, TypeError):
            transcript_format = "plain_text"
        selected.append({
            "recording_id": Path(filename).stem,
            "file_name": filename, "type": row["type"],
            "transcript_format": transcript_format,
            "shard": shard, "row_index_in_shard": index,
            "language": row[spec["language_field"]],
        })
    return selected


def _scan_humyn_online(spec):
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    records = []
    for shard in spec["required_parquet_shards"]:
        remote = f"datasets/{spec['repo_id']}@{spec['revision']}/{shard}"
        with filesystem.open(remote, "rb") as handle:
            records.extend(scan_humyn_shard(handle, shard, spec))
    return records


def acquire_humyn(spec, cache_dir, offline, download, max_recordings,
                  previous=None):
    files = _hub_files(spec, offline)
    previous_shards = {
        row["remote_path"]: row for row in (previous or {}).get("required_shards", [])
    }
    if offline and previous:
        records = [dict(row) for row in previous.get("records", [])]
    elif offline:
        from huggingface_hub.errors import LocalEntryNotFoundError

        records = []
        for shard in spec["required_parquet_shards"]:
            try:
                local = _hub_download(spec, shard, cache_dir, True)
            except LocalEntryNotFoundError as error:
                raise RuntimeError(
                    "Offline Humyn inspection requires a previous manifest or cached shards") from error
            records.extend(scan_humyn_shard(local, shard, spec))
    else:
        records = _scan_humyn_online(spec)
    records.sort(key=lambda row: (row["shard"], row["row_index_in_shard"]))
    if max_recordings:
        records = records[:max_recordings]
    if not records:
        raise ValueError("No Vietnamese Humyn records found")
    if len({row["recording_id"] for row in records}) != len(records):
        raise ValueError("Duplicate Vietnamese Humyn recording_id")
    selected_shards = sorted({row["shard"] for row in records})
    downloaded = []
    for shard in selected_shards:
        expected = files.get(shard) or previous_shards.get(shard)
        if not offline and expected is None:
            raise ValueError(f"Hub revision is missing {shard}")
        if download:
            local = _hub_download(spec, shard, cache_dir, offline)
            downloaded.append({"remote_path": shard, **verify_file(local, expected)})
    return {
        "repo_id": spec["repo_id"], "revision": spec["revision"],
        "license": spec["license"], "record_count": len(records),
        "records": records, "required_shards": [
            {"remote_path": shard, **(files.get(shard) or previous_shards.get(shard) or {})}
            for shard in selected_shards
        ],
        "downloaded_files": downloaded,
    }


def acquire(config_path=DEFAULT_CONFIG, dataset="all", cache_dir=None,
            offline=False, download=False, max_recordings=None,
            output=DEFAULT_OUTPUT):
    if max_recordings is not None and max_recordings < 1:
        raise ValueError("max_recordings must be positive")
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load_config(config_path)
    config_digest = sha256(config_path)
    previous = None
    if offline and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_digest:
            raise ValueError("Offline manifest was created with a different config")
    names = DATASETS if dataset == "all" else (dataset,)
    results = {}
    for name in names:
        prior = (previous or {}).get("datasets", {}).get(name)
        if name == "datahive":
            results[name] = acquire_datahive(
                config["datasets"][name], cache_dir, offline, download,
                max_recordings, prior)
        else:
            results[name] = acquire_humyn(
                config["datasets"][name], cache_dir, offline, download,
                max_recordings, prior)
    manifest = {
        "schema_version": 1, "config_path": str(config_path),
        "config_sha256": config_digest, "offline": offline,
        "download": download, "max_recordings": max_recordings,
        "datasets": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", choices=("datahive", "humyn", "all"), default="all")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true", help="Cache selected audio/shards")
    action.add_argument("--metadata-only", action="store_true",
                        help="Inspect metadata without caching audio")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = acquire(
        config_path=args.config, dataset=args.dataset,
        cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
        offline=args.offline, download=args.download,
        max_recordings=args.max_recordings, output=args.output,
    )
    print(json.dumps({
        name: {"record_count": value["record_count"],
               "downloaded_files": len(value["downloaded_files"])}
        for name, value in manifest["datasets"].items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
