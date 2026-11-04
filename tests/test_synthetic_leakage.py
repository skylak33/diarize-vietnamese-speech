import hashlib
import json

import pytest

from experiments.synthetic_benchmark import (
    ASSIGNMENT_ALGORITHM,
    assign_splits,
    create_split_manifest,
    validate_split_leakage,
)


def _record(speaker, source_file, recording):
    return {
        "speaker_id": speaker,
        "source_file": source_file,
        "content_sha256": hashlib.sha256(source_file.encode()).hexdigest(),
        "source_recording_id": recording,
    }


def test_assignment_is_order_independent_and_keeps_connected_sources_together():
    records = [
        _record("speaker-a", "a-1.wav", "recording-a"),
        _record("speaker-a", "a-2.wav", "recording-b"),
        _record("speaker-b", "b.wav", "recording-b"),
        _record("speaker-c", "c.wav", "recording-c"),
    ]
    first = assign_splits(records, seed=123)
    reversed_result = assign_splits(list(reversed(records)), seed=123)
    first_by_file = {record["source_file"]: record["split"] for record in first}
    assert first_by_file == {record["source_file"]: record["split"] for record in reversed_result}
    assert first_by_file["a-1.wav"] == first_by_file["a-2.wav"] == first_by_file["b.wav"]
    assert validate_split_leakage(first) == {
        "speaker_disjoint": True,
        "source_file_disjoint": True,
        "content_hash_disjoint": True,
        "source_recording_disjoint": True,
    }


@pytest.mark.parametrize(
    "field",
    ["speaker_id", "source_file", "content_sha256", "source_recording_id"],
)
def test_cross_split_identity_duplicates_fail(field):
    records = [
        {**_record("speaker-a", "a.wav", "recording-a"), "split": "train"},
        {**_record("speaker-b", "b.wav", "recording-b"), "split": "test"},
    ]
    records[1][field] = records[0][field]
    with pytest.raises(ValueError, match=field):
        validate_split_leakage(records)


def test_manifest_records_seed_ratios_checks_and_provenance():
    records = [_record(f"speaker-{index}", f"{index}.wav", f"recording-{index}") for index in range(20)]
    provenance = {"dataset": "synthetic-vietnamese-v1", "generator_revision": "abc123"}
    manifest = create_split_manifest(
        records,
        provenance=provenance,
        seed=7,
    )
    assert manifest["provenance"] == provenance
    assert manifest["assignment"] == {
        "algorithm": ASSIGNMENT_ALGORITHM,
        "seed": 7,
        "split_ratios": {"development": 0.6, "holdout": 0.2, "validation": 0.2},
    }
    assert manifest["summary"]["record_count"] == 20
    assert sum(manifest["summary"]["by_split"].values()) == 20
    assert all(manifest["summary"]["by_split"].values())
    assert all(manifest["checks"].values())
    json.dumps(manifest, sort_keys=True)


def test_invalid_hash_and_missing_provenance_are_rejected():
    record = _record("speaker-a", "a.wav", "recording-a")
    record["content_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        assign_splits([record])
    with pytest.raises(ValueError, match="provenance"):
        create_split_manifest([_record("speaker-a", "a.wav", "recording-a")], provenance={})


def test_manifest_rejects_empty_production_split():
    records = [_record("speaker-a", "a.wav", "recording-a")]
    with pytest.raises(ValueError, match="empty splits"):
        create_split_manifest(records, provenance={"dataset": "fixture"})
