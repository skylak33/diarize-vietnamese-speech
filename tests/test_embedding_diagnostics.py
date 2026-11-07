import json
import csv

import numpy as np

from diarize import pipeline
from diarize.embedding_diagnostics import (_distribution_comparison, _distribution_overlap,
                                           classify_reference_windows, cluster_similarity_stats,
                                           cosine_similarity_matrix, reference_similarity_stats,
                                           validate_embeddings, write_artifacts)
from experiments.analyze_embedding_diagnostics import _load_current_metrics


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
    assert config["audio_sha256"] and config["checkpoint_sha256"] and config["encoder_source_sha256"]


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
    assert reference["diagnostic_auc"] is None
    assert reference["distribution_overlap"] is None
    assert reference["ks_statistic"] is None


def test_embedding_validation_and_reference_classes_cover_edge_cases():
    empty, starts, ends = validate_embeddings(np.empty((0, 2)), [], [])
    assert empty.shape == (0, 2) and starts.size == ends.size == 0
    for bad in (np.array([[np.nan, 0.]]), np.array([[0., 0.]])):
        try:
            validate_embeddings(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid embeddings were accepted")

    classifications = classify_reference_windows(
        [0., 1., 2., 3.], [1., 2., 3., 4.],
        [(1., 1.2, "A"), (2., 3., "A"), (2.5, 3., "B")])
    assert [item["classification"] for item in classifications] == [
        "unannotated_window", "uncertain_window", "overlapping_speech_window",
        "unannotated_window"]
    partial_overlap = classify_reference_windows(
        [0.], [1.], [(0., .7, "A"), (.5, .8, "B")])
    assert partial_overlap[0]["classification"] == "overlapping_speech_window"


def test_distribution_comparison_resolves_narrow_ranges_and_degenerate_samples():
    left = np.array([.9950, .9970, .9990])
    right = np.array([.9951, .9971, .9991])
    assert _distribution_overlap(left, right) < 1.
    assert _distribution_overlap([.997], [.997]) == 1.
    assert _distribution_overlap([.997], [.998]) == 0.

    comparison = _distribution_comparison([1., 2.], [0., 1.])
    assert comparison["ks_statistic"] == .5
    assert comparison["wasserstein_distance"] == 1.
    assert comparison["median_separation"] == 1.
    assert comparison["pairwise_ranking_auc"] == .875


def test_reference_intervals_are_unioned_per_speaker_without_double_counting():
    classifications = classify_reference_windows(
        [0., 1.], [1., 2.],
        [(0., 1., "A"), (0., 1., "A"), (1., 1.5, "A"), (1.5, 2., "B")])
    assert classifications[0] == {
        "classification": "unambiguous_speaker_window", "speaker": "A", "coverage": 1.}
    assert classifications[1]["classification"] == "mixed_speaker_window"


def test_diagnostic_auc_remains_tie_aware_pairwise_ranking_probability():
    embeddings = np.array([[1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    classifications = [
        {"classification": "unambiguous_speaker_window", "speaker": speaker}
        for speaker in ("A", "A", "B", "B")]
    stats = reference_similarity_stats(embeddings, classifications)
    assert stats["diagnostic_auc"] == stats["pairwise_ranking_auc"] == .5


def test_reference_source_is_metadata_and_cannot_change_diagnostic_rttm(tmp_path, monkeypatch):
    _mock_pipeline(monkeypatch)
    audio, model = tmp_path / "sample.wav", tmp_path / "model.pt"
    audio.write_bytes(b"audio")
    model.write_bytes(b"model")
    monkeypatch.setattr(pipeline, "cluster_embeddings", lambda *a, **k: np.array([1, 0]))
    outputs = {}
    for source in ("manual_reference", "published_reference"):
        directory = tmp_path / source
        outputs[source] = pipeline.diarize(
            str(audio), str(model), encoder=object(), diagnostic_output_dir=str(directory),
            reference_source=source)
        config = json.loads((directory / "sample" / "config.json").read_text())
        assert config["reference_source"] == source
    assert outputs["manual_reference"] == outputs["published_reference"]


def test_current_metrics_override_historical_target_groups(tmp_path):
    path = tmp_path / "per_file.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "file", "N_ref", "N_pred", "speaker_error_seconds"])
        writer.writeheader()
        writer.writerow({"file": "Hx0j-Oc18do", "N_ref": 2, "N_pred": 2,
                         "speaker_error_seconds": 1.25})
    assert _load_current_metrics(path)["Hx0j-Oc18do"] == {
        "count_relation": "correct_count", "speaker_error_seconds": 1.25}


def test_compactness_is_scale_invariant_and_artifacts_reject_non_normalized_rows(tmp_path):
    embeddings = np.array([[1., 0.], [.8, .2], [0., 1.]])
    labels = np.array([0, 0, 1])
    scaled = embeddings * np.array([[7.], [.2], [13.]])
    assert cluster_similarity_stats(embeddings, labels)["compactness"] == (
        cluster_similarity_stats(scaled, labels)["compactness"])

    try:
        write_artifacts(tmp_path / "bad", scaled, [(0., 1.), (1., 2.), (2., 3.)], {}, {})
    except ValueError as error:
        assert "L2-normalized" in str(error)
    else:
        raise AssertionError("non-normalized diagnostic embeddings were serialized")
    assert not (tmp_path / "bad").exists()


def test_write_artifacts_preserves_float32_embeddings(tmp_path):
    embeddings = np.eye(2, dtype=np.float32)
    write_artifacts(tmp_path, embeddings, [(0., 1.), (1., 2.)], {}, {})
    with np.load(tmp_path / "embeddings.npz") as arrays:
        assert arrays["embeddings"].dtype == np.float32
