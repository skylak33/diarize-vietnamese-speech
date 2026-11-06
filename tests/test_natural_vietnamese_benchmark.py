import json
from argparse import Namespace

import numpy as np
import pytest
import soundfile as sf

from experiments import run_natural_vietnamese_benchmark as benchmark
from utils.provenance import sha256


CONFIGURATIONS = [
    {"name": "fixed_ahc_065", "method": "ahc", "count_estimator": None,
     "threshold": 0.65},
    {"name": "fixed_ahc_075", "method": "ahc", "count_estimator": None,
     "threshold": 0.75},
    {"name": "legacy_adaptive_ahc_075", "method": "ahc",
     "count_estimator": "adaptive_ahc", "threshold": 0.75},
    {"name": "robust_adaptive_ahc_075", "method": "ahc",
     "count_estimator": "adaptive_ahc_robust", "threshold": 0.75},
    {"name": "spectral", "method": "spectral", "count_estimator": None,
     "threshold": 0.75},
    {"name": "nme_sc", "method": "nme_sc", "count_estimator": None,
     "threshold": 0.75},
]


def _fixture(tmp_path):
    audio = tmp_path / "dialog.wav"
    sf.write(audio, np.zeros(4 * 16000, dtype=np.float32), 16000)
    rttm = tmp_path / "dialog.rttm"
    rttm.write_text(
        "SPEAKER dialog 1 0.000 2.000 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER dialog 1 2.000 2.000 <NA> <NA> B <NA> <NA>\n",
        encoding="utf-8")
    checkpoint = tmp_path / "model.model"
    checkpoint.write_bytes(b"frozen checkpoint fixture")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "seed": 42,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "vad": {"threshold": 0.5, "min_speech_ms": 250,
                "min_silence_ms": 100, "speech_pad_ms": 30},
        "embedding": {"window_sec": 2.0, "step_sec": 2.0,
                      "min_segment_sec": 0.5},
        "evaluation": {"collar_seconds": 0.0,
                       "overlap_conditions": ["included", "excluded"],
                       "modes": ["full_pipeline", "clustering_only"]},
        "configurations": CONFIGURATIONS,
    }), encoding="utf-8")
    annotations = tmp_path / "annotations.jsonl"
    eligible = json.dumps({
        "dataset": "datahive", "dataset_revision": "a" * 40,
        "recording_id": "dialog", "eligible": True,
        "audio_path": str(audio), "audio_sha256": sha256(audio),
        "rttm_path": str(rttm), "rttm_sha256": sha256(rttm),
        "speaker_count": 2, "reference_provenance": "silver_reference",
    })
    excluded = json.dumps({
        "dataset": "datahive", "recording_id": "excluded",
        "eligible": False, "exclusion_reason": "channel_without_detected_speech",
    })
    annotations.write_text(eligible + "\n" + excluded + "\n", encoding="utf-8")
    return config, annotations, audio


def _args(config, annotations, output, dry_run=False):
    return Namespace(
        config=str(config), annotation_manifest=str(annotations),
        output_dir=str(output), dataset="datahive", recording_ids=None,
        smoke=False, dry_run=dry_run, modes=None, device="cpu", max_speakers=10,
    )


def test_natural_plan_uses_all_frozen_conditions_without_oracle_counts(tmp_path):
    config, annotations, _ = _fixture(tmp_path)

    result = benchmark.run(_args(config, annotations, tmp_path / "unused", dry_run=True))

    assert result["plan"]["automatic_speaker_count_only"] is True
    assert result["plan"]["modes"] == ["full_pipeline", "clustering_only"]
    assert result["plan"]["overlap_conditions"] == ["included", "excluded"]
    assert [row["name"] for row in result["plan"]["configurations"]] == [
        row["name"] for row in CONFIGURATIONS]


def test_natural_runner_reuses_each_frontend_for_all_configurations(tmp_path, monkeypatch):
    config, annotations, _ = _fixture(tmp_path)
    extract_calls, cluster_calls, vad_calls = [], [], []
    monkeypatch.setattr(benchmark, "load_model", lambda *args: object())
    monkeypatch.setattr(
        benchmark, "load_audio", lambda path: (np.zeros(4 * 16000), 16000))
    monkeypatch.setattr(
        benchmark, "get_speech_segments",
        lambda *args, **kwargs: vad_calls.append(kwargs) or [(0.0, 4.0)])

    def fake_extract(*args):
        extract_calls.append(args)
        return [(0.0, 2.0), (2.0, 4.0)], np.asarray([[1.0, 0.0], [0.0, 1.0]])

    def fake_cluster(embeddings, **kwargs):
        cluster_calls.append((id(embeddings), kwargs))
        return np.asarray([0, 1])

    monkeypatch.setattr(benchmark, "extract_embeddings_for_segments", fake_extract)
    monkeypatch.setattr(benchmark, "cluster_embeddings", fake_cluster)
    monkeypatch.setattr(
        benchmark, "postprocess", lambda windows, labels: (windows, labels))
    output = tmp_path / "results"

    result = benchmark.run(_args(config, annotations, output))

    assert len(extract_calls) == 2 and len(vad_calls) == 1
    assert len(cluster_calls) == 12
    assert len({embedding_id for embedding_id, _ in cluster_calls}) == 2
    assert all(call["n_speakers"] is None for _, call in cluster_calls)
    assert len(result["recordings"]) == 24 and len(result["summary"]) == 24
    assert {row["overlap_condition"] for row in result["summary"]} == {
        "included", "excluded"}
    assert {row["mode"] for row in result["summary"]} == {
        "full_pipeline", "clustering_only"}
    assert all(row["DER"] == 0.0 for row in result["summary"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["recording_count"] == 1
    assert manifest["result_rows"] == manifest["summary_rows"] == 24
    assert manifest["reference_provenance"] == ["silver_reference"]
    assert manifest["excluded_recordings"] == [{
        "dataset": "datahive", "recording_id": "excluded",
        "reason": "channel_without_detected_speech",
    }]


def test_natural_manifest_hash_mismatch_fails_closed(tmp_path):
    _, annotations, audio = _fixture(tmp_path)
    audio.write_bytes(b"changed after annotation preparation")

    with pytest.raises(ValueError, match="audio SHA256 mismatch"):
        benchmark.load_recordings(annotations, "datahive")
