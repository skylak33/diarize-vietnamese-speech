import json

import numpy as np

from diarize import pipeline
from diarize.embedding_diagnostics import (classify_reference_windows, cluster_similarity_stats,
                                           cosine_similarity_matrix, reference_similarity_stats)


def _mock_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline, "load_audio", lambda _: (np.zeros(32000), 16000))
    monkeypatch.setattr(pipeline, "get_speech_segments", lambda *a, **k: [(0., 2.)])
    monkeypatch.setattr(pipeline, "extract_embeddings_for_segments", lambda *a: (
        [(0., 1.), (1., 2.)], np.array([[1., 0.], [0., 1.]])))


def test_diagnostic_artifacts_are_deterministic_and_preserve_rttm(tmp_path, monkeypatch):
    _mock_pipeline(monkeypatch)
    audio, model = tmp_path / "sample.wav", tmp_path / "model.pt"
    audio.write_bytes(b"audio"); model.write_bytes(b"model")
    monkeypatch.setattr(pipeline, "cluster_embeddings", lambda *a, **k: np.array([0, 1]))
    baseline = pipeline.diarize(str(audio), str(model), encoder=object())
    cache, encoder = {}, object()
    first = pipeline.diarize(str(audio), str(model), encoder=encoder, embedding_cache=cache,
                             diagnostic_output_dir=str(tmp_path / "diagnostics"))
    second = pipeline.diarize(str(audio), str(model), encoder=encoder, embedding_cache=cache,
                              diagnostic_output_dir=str(tmp_path / "diagnostics"))
    artifact_dir = tmp_path / "diagnostics" / "sample"
    saved = np.load(artifact_dir / "embeddings.npz")
    trace = json.loads((artifact_dir / "pipeline_trace.json").read_text())
    config = json.loads((artifact_dir / "config.json").read_text())
    assert baseline == first == second
    assert np.allclose(np.linalg.norm(saved["embeddings"], axis=1), 1)
    assert saved["window_starts"].tolist() == [0., 1.]
    assert trace["stages"]["initial_ahc"]["labels"] == [0, 1]
    assert config["audio_sha256"] and config["checkpoint_sha256"]


def test_similarity_statistics_and_reference_overlap():
    embeddings = np.array([[1., 0.], [.99, .01], [0., 1.]])
    labels = np.array([5, 5, 7])
    stats = cluster_similarity_stats(embeddings, labels)
    assert stats["cluster_sizes"] == {"5": 2, "7": 1}
    assert stats["within_predicted_cluster"]["count"] == 1
    try:
        cosine_similarity_matrix(embeddings, max_windows=2)
    except ValueError as error:
        assert "max_windows" in str(error)
    else:
        raise AssertionError("max_windows did not bound pairwise similarity")
    windows = classify_reference_windows([0, 1, 2], [1, 2, 3],
        [(0, 1, "a"), (1, 1.5, "a"), (1.5, 2, "b"), (2, 3, "b")])
    assert [item["classification"] for item in windows] == [
        "unambiguous_speaker_window", "mixed_speaker_window", "unambiguous_speaker_window"]
    reference = reference_similarity_stats(embeddings, windows)
    assert reference["same_speaker_similarity"]["count"] == 0
    assert reference["different_speaker_similarity"]["count"] == 1
