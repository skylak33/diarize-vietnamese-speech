import json
import wave

import pytest

from experiments.prepare_diarization_devset import main, prepare_manifest


def _wav(path, seconds=1, sample=0):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(sample.to_bytes(2, "little") * 8000 * seconds)


def _rttm(path, recording_id, speakers):
    duration = 0.9 / len(speakers)
    path.write_text("".join(
        f"SPEAKER {recording_id} 1 {index * duration:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        for index, speaker in enumerate(speakers)
    ), encoding="utf-8")


def _spec(tmp_path, global_speakers=True):
    for sample, (recording_id, speakers) in enumerate((("dev", ["A", "B"]), ("val", ["C"]))):
        _wav(tmp_path / f"{recording_id}.wav", sample=sample)
        _rttm(tmp_path / f"{recording_id}.rttm", recording_id, speakers)
    spec = {
        "dataset": {
            "name": "Example conversations",
            "source": "https://example.test/dataset",
            "license": "CC-BY-4.0",
            "revision": "v1",
            "language": "English",
            "domain": "meetings",
            "annotation_format": "RTTM",
            "project_usage": "independent_development",
            "global_speaker_ids": global_speakers,
        },
        "root": ".",
        "recordings": [
            {"recording_id": "dev", "split": "development", "audio_path": "dev.wav", "rttm_path": "dev.rttm", "source_id": "source-1", "conversation_id": "conversation-1"},
            {"recording_id": "val", "split": "validation", "audio_path": "val.wav", "rttm_path": "val.rttm", "source_id": "source-2", "conversation_id": "conversation-2"},
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def test_prepares_hashes_duration_counts_and_provenance(tmp_path):
    path, _ = _spec(tmp_path)
    manifest = prepare_manifest(path)
    assert manifest["summary"] == {
        "recording_count": 2,
        "duration_seconds": 2.0,
        "speaker_count": 3,
        "speaker_instances": 3,
        "by_split": {
            "development": {"recording_count": 1, "duration_seconds": 1.0, "speaker_count": 2, "speaker_instances": 2},
            "validation": {"recording_count": 1, "duration_seconds": 1.0, "speaker_count": 1, "speaker_instances": 1},
        },
    }
    assert manifest["dataset"]["revision"] == "v1"
    assert len(manifest["recordings"][0]["audio_sha256"]) == 64
    assert manifest["checks"]["speaker_disjoint"] is True


@pytest.mark.parametrize("field", ["source_id", "conversation_id"])
def test_rejects_source_and_conversation_leakage(tmp_path, field):
    path, spec = _spec(tmp_path)
    spec["recordings"][1][field] = spec["recordings"][0][field]
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{field} leakage"):
        prepare_manifest(path)


def test_global_speaker_leakage_is_enforced_and_local_ids_are_marked_unknown(tmp_path):
    path, spec = _spec(tmp_path)
    _rttm(tmp_path / "val.rttm", "val", ["A"])
    with pytest.raises(ValueError, match="speaker leakage"):
        prepare_manifest(path)

    spec["dataset"]["global_speaker_ids"] = False
    path.write_text(json.dumps(spec), encoding="utf-8")
    manifest = prepare_manifest(path)
    assert manifest["checks"]["speaker_disjoint"] is None
    assert manifest["summary"]["speaker_count"] is None


def test_rejects_reused_test_data_and_missing_independent_inventory(tmp_path, capsys):
    path, spec = _spec(tmp_path)
    spec["dataset"]["project_usage"] = "reused_test"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="reused test"):
        prepare_manifest(path)

    with pytest.raises(SystemExit):
        main([])
    assert "refusing to use local ViYT-Diar" in capsys.readouterr().err


def test_requires_both_splits_and_multispeaker_audio(tmp_path):
    path, spec = _spec(tmp_path)
    spec["recordings"] = spec["recordings"][:1]
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="Both development and validation"):
        prepare_manifest(path)

    path, spec = _spec(tmp_path)
    _rttm(tmp_path / "dev.rttm", "dev", ["A"])
    with pytest.raises(ValueError, match="no multi-speaker"):
        prepare_manifest(path)
