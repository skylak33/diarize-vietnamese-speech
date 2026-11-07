import json

import numpy as np

from experiments import analyze_smoothing_counterfactuals as analysis


def _artifact(tmp_path):
    artifact = tmp_path / "sample"
    artifact.mkdir()
    embeddings = np.eye(5, dtype=float)
    starts = np.arange(5, dtype=float)
    ends = starts + 1
    np.savez(artifact / "embeddings.npz", embeddings=embeddings,
             window_starts=starts, window_ends=ends)
    (artifact / "pipeline_trace.json").write_text(json.dumps({
        "schema_version": 1, "recording_id": "sample", "stages": {}}),
        encoding="utf-8")
    (artifact / "config.json").write_text(json.dumps({
        "inference_configuration": {
            "method": "ahc", "count_estimator": "backend_default",
            "threshold": .75, "max_speakers": 10, "seed": 7,
            "centroid_merge": False,
        }}), encoding="utf-8")
    return artifact


def test_reports_stage_counts_for_current_and_no_smoothing(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)

    def cluster(_embeddings, n_speakers=None, **_kwargs):
        return np.array([0, 0, 1, 2, 0] if n_speakers == 3 else [0, 0, 1, 0, 0])

    monkeypatch.setattr(analysis, "cluster_embeddings", cluster)
    reference = [(0., 2., "a"), (2., 3., "b"), (3., 5., "c")]
    result = analysis.analyze_recording(artifact, reference, duration=5.)

    assert result["variants"]["automatic_current"]["stages"] == {
        "initial": 2, "after_centroid_merge": 2, "after_smoothing": 1, "final": 1}
    assert result["variants"]["automatic_without_smoothing"]["stages"]["final"] == 2
    assert result["variants"]["oracle_count_current"]["stages"]["initial"] == 3
    assert result["variants"]["oracle_count_without_smoothing"]["stages"]["final"] == 3


def test_recomputation_is_deterministic():
    embeddings = np.array([
        [1., 0.], [.99, .01], [0., 1.], [.01, .99], [1., .02], [.02, 1.]])
    segments = [(float(i), float(i + 1)) for i in range(len(embeddings))]
    config = {"method": "ahc", "count_estimator": "backend_default",
              "threshold": .2, "max_speakers": 10, "seed": 42,
              "centroid_merge": False}

    first = analysis.recompute_variants(embeddings, segments, config, 2)
    second = analysis.recompute_variants(embeddings, segments, config, 2)

    assert analysis.summarize_variants(first) == analysis.summarize_variants(second)


def test_reference_content_never_reaches_clustering(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    calls = []

    def cluster(embeddings, **kwargs):
        calls.append((embeddings.copy(), kwargs.copy()))
        return np.array([0, 0, 1, 1, 1])

    monkeypatch.setattr(analysis, "cluster_embeddings", cluster)
    first_reference = [(0., 2., "alice"), (2., 5., "bob")]
    second_reference = [(0., 1., "bob"), (1., 5., "alice")]
    analysis.analyze_recording(artifact, first_reference, duration=5.)
    first_calls = calls[:2]
    calls.clear()
    analysis.analyze_recording(artifact, second_reference, duration=5.)

    assert [call[1]["n_speakers"] for call in first_calls] == [None, 2]
    assert [call[1] for call in first_calls] == [call[1] for call in calls]
    assert all(np.array_equal(left[0], right[0]) for left, right in zip(first_calls, calls))
