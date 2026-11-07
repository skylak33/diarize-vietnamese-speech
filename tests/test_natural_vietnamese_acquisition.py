import hashlib
import json
import wave
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.acquire_natural_vietnamese_benchmark import (
    acquire,
    acquire_datahive,
    acquire_humyn,
    parse_args,
    parse_datahive_metadata,
    scan_humyn_shard,
    validate_datahive_audio,
    verify_file,
)


DATAHIVE = {
    "repo_id": "owner/datahive", "revision": "a" * 40,
    "license": "CC-BY-4.0", "language_field": "language",
    "language_value": "vi", "metadata_path": "preview/metadata.jsonl",
}
HUMYN = {
    "repo_id": "owner/humyn", "revision": "b" * 40,
    "license": "CC-BY-4.0", "language_field": "language",
    "language_value": "Vietnamese",
    "required_parquet_shards": ["data/shard.parquet"],
}


def _metadata(path):
    rows = [
        {"file_name": "en_1.mp3", "language": "en", "duration_seconds": 1},
        {"file_name": "vi_2.mp3", "language": "vi", "duration_seconds": 2.5},
        {"file_name": "vi_1.mp3", "language": "vi", "duration_seconds": 1.5},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _wav(path, channels=2, sample_rate=48000):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * channels * 100)


def test_datahive_metadata_is_vietnamese_only_sorted_and_limited(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    _metadata(metadata)

    records = parse_datahive_metadata(metadata, DATAHIVE, max_recordings=1)

    assert records == [{
        "recording_id": "vi_1", "file_name": "vi_1.mp3",
        "remote_path": "preview/vi_1.mp3", "duration_seconds": 1.5,
        "language": "vi",
    }]


def test_integrity_and_datahive_audio_validation_fail_closed(tmp_path):
    audio = tmp_path / "stereo.wav"
    _wav(audio)
    expected_hash = hashlib.sha256(audio.read_bytes()).hexdigest()

    assert verify_file(audio, {"size": audio.stat().st_size,
                               "sha256": expected_hash})["sha256"] == expected_hash
    assert validate_datahive_audio(audio)["channels"] == 2
    with pytest.raises(ValueError, match="size"):
        verify_file(audio, {"size": 1})

    mono = tmp_path / "mono.wav"
    _wav(mono, channels=1)
    with pytest.raises(ValueError, match="expected non-empty 48 kHz stereo"):
        validate_datahive_audio(mono)
    corrupt = tmp_path / "corrupt.mp3"
    corrupt.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Unreadable audio"):
        validate_datahive_audio(corrupt)


def test_humyn_parquet_scan_validates_schema_and_filters_language(tmp_path):
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({
        "language": ["Vietnamese", "Korean", "Vietnamese"],
        "file_name": ["dialog.wav", "other.wav", "single.mp3"],
        "audio": pa.array([
            {"bytes": b"a", "path": "dialog.wav"},
            {"bytes": b"b", "path": "other.wav"},
            {"bytes": b"c", "path": "single.mp3"},
        ], type=audio_type),
        "transcript_json": [
            json.dumps([{"start": "00:00:00", "end": "00:00:01",
                         "speaker": "Speaker 1"}]),
            "ignored", "plain transcript",
        ],
        "type": ["multi-speaker", "single-speaker", "single-speaker"],
    })
    path = tmp_path / "fixture.parquet"
    pq.write_table(table, path)

    rows = scan_humyn_shard(path, "fixture.parquet", HUMYN)

    assert [(row["recording_id"], row["transcript_format"], row["row_index_in_shard"])
            for row in rows] == [("dialog", "json_list", 0),
                                 ("single", "plain_text", 2)]

    bad = tmp_path / "bad.parquet"
    pq.write_table(table.drop(["type"]), bad)
    with pytest.raises(ValueError, match="missing Parquet fields"):
        scan_humyn_shard(bad, "bad.parquet", HUMYN)


def test_datahive_metadata_only_uses_pinned_hub_metadata(tmp_path, monkeypatch):
    import huggingface_hub

    metadata = tmp_path / "metadata.jsonl"
    _metadata(metadata)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(metadata)

    sibling = SimpleNamespace(
        rfilename="preview/vi_1.mp3", size=123, blob_id="blob",
        lfs=SimpleNamespace(sha256="f" * 64),
    )
    info = SimpleNamespace(sha=DATAHIVE["revision"], siblings=[sibling])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "HfApi",
                        lambda: SimpleNamespace(dataset_info=lambda *a, **k: info))

    result = acquire_datahive(
        DATAHIVE, tmp_path / "cache", offline=False, download=False,
        max_recordings=1)

    assert result["record_count"] == 1
    assert result["records"][0]["expected_sha256"] == "f" * 64
    assert result["downloaded_files"] == []
    assert [call["filename"] for call in calls] == ["preview/metadata.jsonl"]
    assert calls[0]["revision"] == DATAHIVE["revision"]


def test_datahive_offline_download_reuses_cache_without_api(tmp_path, monkeypatch):
    import huggingface_hub

    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps({
        "file_name": "vi_1.mp3", "language": "vi", "duration_seconds": 1.0,
    }), encoding="utf-8")
    audio = tmp_path / "vi_1.wav"
    _wav(audio)
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(metadata if kwargs["filename"].endswith("metadata.jsonl") else audio)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        huggingface_hub, "HfApi",
        lambda: pytest.fail("offline mode must not construct HfApi"),
    )
    previous = {"records": [{
        "remote_path": "preview/vi_1.mp3", "expected_size": audio.stat().st_size,
        "expected_sha256": digest,
    }]}

    result = acquire_datahive(
        DATAHIVE, tmp_path / "cache", offline=True, download=True,
        max_recordings=None, previous=previous)

    assert result["downloaded_files"][0]["sha256"] == digest
    assert result["records"][0]["audio"]["sample_rate_hz"] == 48000
    assert all(call["local_files_only"] for call in calls)


def test_humyn_offline_download_reuses_manifest_integrity(tmp_path, monkeypatch):
    import huggingface_hub

    shard = tmp_path / "shard.parquet"
    shard.write_bytes(b"cached shard")
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(shard)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        huggingface_hub, "HfApi",
        lambda: pytest.fail("offline mode must not construct HfApi"),
    )
    previous = {
        "records": [{
            "recording_id": "dialog", "file_name": "dialog.wav",
            "type": "multi-speaker", "transcript_format": "json_list",
            "shard": "data/shard.parquet", "row_index_in_shard": 0,
            "language": "Vietnamese",
        }],
        "required_shards": [{
            "remote_path": "data/shard.parquet", "size": shard.stat().st_size,
            "sha256": digest,
        }],
    }

    result = acquire_humyn(
        HUMYN, tmp_path / "cache", offline=True, download=True,
        max_recordings=None, previous=previous)

    assert result["downloaded_files"][0]["sha256"] == digest
    assert calls[0]["local_files_only"] is True


def test_cli_exposes_required_acquisition_controls():
    args = parse_args([
        "--metadata-only", "--dataset", "datahive", "--cache-dir", "cache",
        "--offline", "--max-recordings", "2",
    ])
    assert args.metadata_only and not args.download
    assert args.dataset == "datahive" and args.offline and args.max_recordings == 2


def test_offline_reuse_rejects_a_manifest_from_another_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "datasets": {
            "datahive": {**DATAHIVE, "metadata_path": "preview/metadata.jsonl"},
            "humyn": {
                **HUMYN,
            },
        },
    }), encoding="utf-8")
    output = tmp_path / "manifest.json"
    output.write_text(json.dumps({"config_sha256": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="different config"):
        acquire(config, dataset="datahive", offline=True, output=output)
