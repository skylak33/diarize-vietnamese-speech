import json

import numpy as np
import pytest
import soundfile as sf

from experiments import audit_silver_reference as audit


def _rttm(path, recording_id, rows):
    path.write_text("\n".join(
        f"SPEAKER {recording_id} 1 {start:.3f} {end-start:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for start, end, speaker in rows) + "\n", encoding="utf-8")


def test_stereo_alignment_lag_overlap_leakage_and_confidence(tmp_path):
    sample_rate, seconds = 1000, 8
    rng = np.random.default_rng(7)
    source = rng.normal(0, 0.2, sample_rate * seconds).astype(np.float32)
    delayed = np.concatenate((np.zeros(100, dtype=np.float32), source[:-100]))
    audio = np.column_stack((source, 0.7 * delayed))
    path = tmp_path / "sample.wav"
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    rttm = tmp_path / "sample.rttm"
    _rttm(rttm, "sample", [(0, 4, "SPEAKER_00"), (3, 8, "SPEAKER_01")])

    summary, suspicious, confident = audit.audit_recording("sample", path, rttm)

    assert summary["sample_alignment"] == "shared_interleaved_frame_clock"
    assert summary["simultaneous_activity_seconds"] == pytest.approx(1.0)
    assert abs(summary["lag_seconds"]) == pytest.approx(0.1, abs=0.02)
    assert any(row["reason"] == "exclusive_speech_energy_leakage" for row in suspicious)
    assert not confident


def test_no_overlap_high_confidence_and_output_manifest_is_not_human_verified(tmp_path):
    sample_rate = 1000
    audio = np.zeros((4000, 2), dtype=np.float32)
    audio[:2000, 0] = 0.1
    audio[2000:, 1] = 0.1
    path = tmp_path / "clean.wav"
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    rttm = tmp_path / "clean.rttm"
    _rttm(rttm, "clean", [(0, 2, "SPEAKER_00"), (2, 4, "SPEAKER_01")])
    manifest = tmp_path / "annotations.jsonl"
    manifest.write_text(json.dumps({"dataset": "datahive", "eligible": True,
        "recording_id": "clean", "audio_path": str(path), "rttm_path": str(rttm)}) + "\n")

    rows = audit.run(manifest, tmp_path / "out")

    assert rows[0]["simultaneous_activity_seconds"] == 0
    assert rows[0]["high_confidence_interval_count"] == 2
    review = json.loads((tmp_path / "out/human_review_manifest.json").read_text())
    assert review["human_verified"] is False and review["items"] == []


def test_missing_channel_and_invalid_reference_timestamp_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="stereo"):
        audit.validate_stereo(np.zeros((10, 1)), 16000)
    with pytest.raises(ValueError, match="invalid_reference_timestamp"):
        audit.validate_reference([(0, 2, "SPEAKER_00")], 1.0)
