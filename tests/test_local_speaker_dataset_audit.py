import csv
import json
import wave

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.audit_local_speaker_datasets import audit_dnu, audit_parquet_dataset


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "speaker_id": [row[0] for row in rows],
        "audio": pa.array(
            [{"bytes": b"audio", "path": row[1]} for row in rows],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        ),
        "split": ["train"] * len(rows),
    })
    pq.write_table(table, path)


def test_parquet_audit_reports_missing_shards_and_streams_manifest(tmp_path):
    root = tmp_path / "VieSpeaker-Dataset-opus"
    write_parquet(root / "data" / "part-00000-of-00002.parquet", [("speaker1", "a.opus")])
    manifest = []

    audit = audit_parquet_dataset(
        root, "viespeaker", expected_shards=2, expected_rows=2,
        official_split="part", manifest_write=manifest.append,
    )

    assert audit["local_completeness"]["missing_shards"] == [1]
    assert audit["counts"]["utterances"] == 1
    assert manifest[0]["source_type"] == "parquet_row"
    assert manifest[0]["speaker_id"] == "viespeaker::speaker1"
    assert manifest[0]["source_speaker_id"] == "speaker1"
    assert manifest[0]["utterance_id"].startswith("viespeaker::")
    assert "#row_group=0&row_index=0" in manifest[0]["source_locator"]
    assert manifest[0]["content_sha256"] is None
    assert manifest[0]["row_index_in_group"] == 0


def test_dnu_audit_validates_headers_and_manifest_paths(tmp_path):
    root = tmp_path / "DNU-ViSpeech"
    fields = [
        "file_name", "audio_id", "speaker_id", "text_normalized", "duration",
        "sample_rate", "num_channels", "checksum_sha256", "split",
    ]
    for split, speaker in (("train", "speaker001"), ("validation", "speaker019"),
                           ("test", "speaker023")):
        directory = root / "data" / split
        audio = directory / "audio" / speaker / f"{speaker}_0001.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\0\0" * 1600)
        with (directory / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "file_name": f"audio/{speaker}/{speaker}_0001.wav",
                "audio_id": f"{speaker}_0001", "speaker_id": speaker,
                "text_normalized": split, "duration": "0.1", "sample_rate": "16000",
                "num_channels": "1", "checksum_sha256": {
                    "train": "a" * 64, "validation": "b" * 64, "test": "c" * 64,
                }[split], "split": split,
            })
    manifest = []

    audit = audit_dnu(root, manifest.append)

    assert audit["counts"]["utterances"] == 3
    assert audit["audio_metadata"]["header_mismatches"] == {}
    assert audit["issues"] == []
    assert all(json.loads(json.dumps(row))["source_type"] == "audio_file" for row in manifest)
    assert manifest[0]["speaker_id"] == "dnu::speaker001"
    assert manifest[0]["source_recording_id"] == "dnu::speaker001_0001"
    assert manifest[0]["source_locator"].endswith("speaker001_0001.wav")
