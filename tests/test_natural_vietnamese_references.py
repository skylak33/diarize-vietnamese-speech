import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from experiments import prepare_natural_vietnamese_references as references
from diarize.vad import _load_silero_model, get_speech_segments_from_audio, load_audio
from utils.provenance import sha256


VAD = {
    "threshold": 0.5, "min_speech_ms": 250,
    "min_silence_ms": 100, "speech_pad_ms": 30,
}
QC = {
    "silence_amplitude": 0.0001, "clipping_amplitude": 0.999,
    "suspicious_overlap_min_seconds": 0.1, "high_overlap_ratio": 0.5,
    "high_channel_correlation": 0.8, "leakage_rms_ratio": 0.5,
    "low_speech_balance_ratio": 0.1,
    "ordinary_review_intervals_per_recording": 1,
}
SPEC = {"revision": "a" * 40, "reference_provenance": "silver_reference"}


def _wav_bytes(audio, sample_rate=16000):
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def test_datahive_channel_rttm_preserves_overlap_and_provenance(tmp_path, monkeypatch):
    audio = np.column_stack((np.full(48000, 0.1), np.full(48000, 0.2)))
    path = tmp_path / "dialog.wav"
    sf.write(path, audio, 48000)
    record = {
        "recording_id": "dialog", "local_path": str(path),
        "expected_sha256": sha256(path),
    }

    def fake_vad(channel, sample_rate, **kwargs):
        assert sample_rate == 48000 and kwargs == VAD
        return [(0.0, 0.6)] if np.mean(channel) < 0.15 else [(0.4, 1.0)]

    monkeypatch.setattr(references, "get_speech_segments_from_audio", fake_vad)
    prepared, review = references.prepare_datahive_record(
        record, SPEC, VAD, QC, tmp_path / "rttm")

    lines = (tmp_path / "rttm/dialog.rttm").read_text().splitlines()
    assert lines == [
        "SPEAKER dialog 1 0.000 0.600 <NA> <NA> SPEAKER_00 <NA> <NA>",
        "SPEAKER dialog 1 0.400 0.600 <NA> <NA> SPEAKER_01 <NA> <NA>",
    ]
    assert prepared["reference_provenance"] == "silver_reference"
    assert prepared["diagnostics"]["overlap_seconds"] == pytest.approx(0.2)
    assert review["suspicious_intervals"] == [
        {"start": 0.4, "end": 0.6, "reason": "simultaneous_channel_activity"}
    ]


def test_datahive_reference_rejects_corrupt_or_silent_channel(tmp_path, monkeypatch):
    corrupt = tmp_path / "bad.mp3"
    corrupt.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="unreadable_audio"):
        references.prepare_datahive_record(
            {"recording_id": "bad", "local_path": str(corrupt)},
            SPEC, VAD, QC, tmp_path)

    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.ones((16000, 2), dtype=np.float32) * 0.1, 16000)
    monkeypatch.setattr(
        references, "get_speech_segments_from_audio",
        lambda channel, sample_rate, **kwargs: [] if channel[0] > 0 else [(0, 1)],
    )
    with pytest.raises(ValueError, match="channel_without_detected_speech"):
        references.prepare_datahive_record(
            {"recording_id": "silent", "local_path": str(stereo)},
            SPEC, VAD, QC, tmp_path)


def test_humyn_timestamp_validation_and_speaker_mapping():
    transcript = json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "speaker": "A"},
        {"start": "00:00:00.500", "end": "00:00:02.000", "speaker": "B"},
    ])
    segments, labels, mapping = references.parse_humyn_annotation(transcript, 2.0)
    assert segments == [(0.0, 1.0), (0.5, 2.0)]
    assert labels.tolist() == [0, 1] and mapping == {"A": 0, "B": 1}

    invalid = [
        ([{"start": "bad", "end": "00:00:01", "speaker": "A"}], "invalid timestamp"),
        ([{"start": "00:00:01", "end": "00:00:01", "speaker": "A"}], "start must"),
        ([{"start": "00:00:00", "end": "00:00:03", "speaker": "A"}], "exceeds"),
        ([{"start": "00:00:00", "end": "00:00:01", "speaker": ""}], "invalid speaker"),
    ]
    for rows, message in invalid:
        with pytest.raises(ValueError, match=message):
            references.parse_humyn_annotation(json.dumps(rows), 2.0)
    duplicate = [{"start": "00:00:00", "end": "00:00:01", "speaker": "A"}] * 2
    with pytest.raises(ValueError, match="duplicate"):
        references.parse_humyn_annotation(json.dumps(duplicate), 2.0)
    with pytest.raises(ValueError, match="unsupported annotation format"):
        references.parse_humyn_annotation("plain transcript", 2.0)


def test_humyn_embedded_audio_reference_and_overlap(tmp_path):
    transcript = json.dumps([
        {"start": "00:00:00", "end": "00:00:00.750", "speaker": "Speaker 1"},
        {"start": "00:00:00.500", "end": "00:00:01.000", "speaker": "Speaker 2"},
    ])
    record = {"recording_id": "viet", "file_name": "viet.wav"}
    row = {
        "file_name": "viet.wav", "audio": {"bytes": _wav_bytes(np.zeros(16000))},
        "transcript_json": transcript,
    }
    prepared, review = references.prepare_humyn_record(
        record, row,
        {"reference_provenance": "transcript_derived_reference"},
        tmp_path / "rttm", tmp_path / "audio")
    lines = (tmp_path / "rttm/viet.rttm").read_text().splitlines()
    assert [line.split()[7] for line in lines] == ["SPEAKER_00", "SPEAKER_01"]
    assert lines[1].split()[3:5] == ["0.500", "0.500"]
    assert prepared["reference_provenance"] == "transcript_derived_reference"
    assert (tmp_path / "audio/viet.wav").is_file()
    assert review["suspicious_intervals"] == [
        {"start": 0.5, "end": 0.75, "reason": "transcript_overlap"}
    ]


def test_humyn_parquet_row_extraction_uses_manifest_indices(tmp_path):
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({
        "file_name": ["skip.wav", "viet.wav"],
        "audio": pa.array([
            {"bytes": b"skip", "path": "skip.wav"},
            {"bytes": b"audio", "path": "viet.wav"},
        ], type=audio_type),
        "transcript_json": ["skip", "[]"],
        "type": ["single-speaker", "multi-speaker"],
    })
    path = tmp_path / "fixture.parquet"
    pq.write_table(table, path)

    rows = references._read_humyn_rows(path, [1])

    assert list(rows) == [1]
    assert rows[1]["file_name"] == "viet.wav"
    assert rows[1]["audio"]["bytes"] == b"audio"


def test_humyn_missing_shard_and_plain_text_have_explicit_exclusions(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "seed": 42,
        "datasets": {"humyn": {
            "revision": "b" * 40,
            "reference_provenance": "transcript_derived_reference",
        }},
    }), encoding="utf-8")
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(json.dumps({
        "config_sha256": sha256(config),
        "datasets": {"humyn": {
            "downloaded_files": [],
            "records": [
                {"recording_id": "plain", "file_name": "plain.wav",
                 "shard": "data.parquet", "row_index_in_shard": 0,
                 "transcript_format": "plain_text"},
                {"recording_id": "dialog", "file_name": "dialog.wav",
                 "shard": "data.parquet", "row_index_in_shard": 1,
                 "transcript_format": "json_list"},
            ],
        }},
    }), encoding="utf-8")

    rows = references.prepare(
        config, acquisition, tmp_path / "output", dataset="humyn")

    assert [(row["recording_id"], row["exclusion_reason"]) for row in rows] == [
        ("dialog", "audio_shard_not_cached"),
        ("plain", "unsupported_annotation_format"),
    ]
    assert all(row["reference_provenance"] == "transcript_derived_reference"
               for row in rows)


def test_review_sampling_is_deterministic():
    review = {"ordinary_candidates": [
        {"start": value, "end": value + 1, "speaker": "SPEAKER_00"}
        for value in range(10)
    ], "suspicious_intervals": []}
    first = references._sample_review(dict(review), __import__("random").Random(42), 3)
    second = references._sample_review(dict(review), __import__("random").Random(42), 3)
    assert first == second


def test_vad_waveform_input_resamples_and_file_loading_downmixes(tmp_path, monkeypatch):
    import silero_vad

    received = {}
    loads = []
    _load_silero_model.cache_clear()
    monkeypatch.setattr(
        silero_vad, "load_silero_vad",
        lambda: loads.append(object()) or loads[-1])

    def fake_timestamps(audio, model, **kwargs):
        received.update(kwargs)
        received["samples"] = len(audio)
        return [{"start": 0.0, "end": 1.0}]

    monkeypatch.setattr(silero_vad, "get_speech_timestamps", fake_timestamps)
    assert get_speech_segments_from_audio(np.ones(8000), 8000) == [(0.0, 1.0)]
    assert get_speech_segments_from_audio(np.ones(16000), 16000) == [(0.0, 1.0)]
    assert received["samples"] == 16000 and received["sampling_rate"] == 16000
    assert len(loads) == 1
    with pytest.raises(ValueError, match="mono waveform"):
        get_speech_segments_from_audio(np.ones((10, 2)), 16000)

    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack((np.zeros(8000), np.ones(8000) * 0.5)), 8000)
    mono, sample_rate = load_audio(str(path))
    assert sample_rate == 16000 and len(mono) == 16000
    assert np.mean(mono) == pytest.approx(0.25, abs=1e-3)
    _load_silero_model.cache_clear()
