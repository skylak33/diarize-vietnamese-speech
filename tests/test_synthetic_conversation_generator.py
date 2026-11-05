import hashlib
import json
import random
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from experiments import generate_synthetic_conversations as generator
from experiments.generate_synthetic_conversations import (
    SourceReader,
    assign_source_split,
    generate,
    load_config,
)
from experiments.run_synthetic_diarization_benchmark import load_conversations


CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_vietnamese_benchmark.json"


class AllSpeechVad:
    def detect(self, audio):
        return [(0, len(audio))]


def _wav(path, seconds=8):
    frequency = 220 + sum(path.name.encode()) % 100
    audio = np.sin(np.arange(16000 * seconds) * 2 * np.pi * frequency / 16000).astype(np.float32) * .1
    sf.write(path, audio, 16000, subtype="PCM_16")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path):
    rows = []
    mapping = {"train": "speaker001", "validation": "speaker019", "test": "speaker023"}
    for split, speaker in mapping.items():
        for index in range(6):
            path = tmp_path / f"{split}-{index}.wav"
            digest = _wav(path)
            utterance = f"dnu::{speaker}_{index:04d}"
            rows.append({
                "dataset": "dnu_vispeech", "utterance_id": utterance,
                "source_recording_id": utterance, "split": split,
                "speaker_id": f"dnu::{speaker}", "source_speaker_id": speaker,
                "source_type": "audio_file", "source_path": str(path),
                "source_locator": str(path), "content_sha256": digest,
            })
    path = tmp_path / "utterance_manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_committed_config_is_clean_balanced_and_reproducible():
    config = load_config(CONFIG)
    assert config["seed"] == 42
    assert config["target_counts"] == {
        "development": 300, "validation": 150, "holdout": 150}
    assert config["speaker_counts"] == [1, 2, 3, 4, 5]
    assert config["scenario"] == "clean"
    assert not config["noise"] and not config["overlap"] and config["transformations"] == []


def test_official_and_speaker_hash_split_policies_are_deterministic():
    assert assign_source_split({"dataset": "dnu_vispeech", "split": "train",
                                "speaker_id": "dnu::speaker001"}, 42) == "development"
    assert assign_source_split({"dataset": "dnu_vispeech", "split": "validation",
                                "speaker_id": "dnu::speaker019"}, 42) == "validation"
    assert assign_source_split({"dataset": "dnu_vispeech", "split": "test",
                                "speaker_id": "dnu::speaker023"}, 42) == "holdout"
    assert assign_source_split({"dataset": "vietnamese_asr", "split": "test",
                                "speaker_id": "vietnamese_asr::VIVOSDEV01"}, 42) == "holdout"
    row = {"dataset": "viespeaker", "split": "part", "speaker_id": "viespeaker::id00001"}
    assert assign_source_split(row, 42) == assign_source_split(row, 42)


def test_three_speaker_conversation_balances_available_source_datasets():
    pools = {
        "dnu::a": [{"dataset": "dnu_vispeech"}],
        "vietnamese_asr::a": [{"dataset": "vietnamese_asr"}],
        "viespeaker::a": [{"dataset": "viespeaker"}],
    }
    selected = generator._select_speakers(pools, 3, 1, random.Random(42))
    assert {pools[speaker][0]["dataset"] for speaker in selected} == set(
        ("dnu_vispeech", "vietnamese_asr", "viespeaker"))


def test_source_reader_verifies_declared_dnu_checksum(tmp_path):
    path = tmp_path / "source.wav"
    digest = _wav(path, 3)
    reader = SourceReader()
    audio, actual = reader.read({"source_type": "audio_file", "source_path": str(path),
                                 "content_sha256": digest})
    assert len(audio) == 48000 and actual == digest
    with pytest.raises(ValueError, match="checksum"):
        reader.read({"source_type": "audio_file", "source_path": str(path),
                     "content_sha256": "0" * 64})


def test_tiny_clean_generation_writes_valid_wav_rttm_json(tmp_path):
    config = load_config(CONFIG)
    output = tmp_path / "generated"
    result = generate(
        config, _manifest(tmp_path), output,
        target_counts={"development": 1, "validation": 1, "holdout": 1},
        speaker_counts=[1], vad=AllSpeechVad(),
    )

    assert result["counts"] == {"development": 1, "validation": 1, "holdout": 1}
    conversations = load_conversations(output)
    assert len(conversations) == 3
    assert Counter(row["split"] for row in conversations) == result["counts"]
    for row in conversations:
        assert row["scenario"] == "clean" and row["speaker_count"] == 1
        assert 30 <= row["duration_seconds"] <= 45
        assert len(row["turns"]) >= 4
        assert {turn["speaker_id"] for turn in row["turns"]} == set(row["speaker_ids"])
        assert min(Counter(turn["speaker_id"] for turn in row["turns"]).values()) >= 2
        assert all(3 <= turn["end"] - turn["start"] <= 8 for turn in row["turns"])
        with wave.open(str(row["audio_path"]), "rb") as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        rttm = row["rttm_path"].read_text(encoding="utf-8").splitlines()
        assert rttm and all(line.startswith(f"SPEAKER {row['conversation_id']} 1 ") for line in rttm)
        assert all("<NA> <NA> dnu::" in line for line in rttm)


def test_generation_drops_speaker_without_usable_speech_and_retries(tmp_path, monkeypatch):
    original = generator._conversation
    failed = {"value": False}

    def fail_once(config, split, index, speaker_count, pools, rng, reader, vad):
        if split == "development" and not failed["value"]:
            failed["value"] = True
            raise generator.UnusableSpeakerError("dnu::unusable")
        return original(config, split, index, speaker_count, pools, rng, reader, vad)

    monkeypatch.setattr(generator, "_conversation", fail_once)
    result = generate(
        load_config(CONFIG), _manifest(tmp_path), tmp_path / "generated-retry",
        target_counts={"development": 1, "validation": 1, "holdout": 1},
        speaker_counts=[1], vad=AllSpeechVad())
    assert failed["value"] and result["counts"]["development"] == 1
