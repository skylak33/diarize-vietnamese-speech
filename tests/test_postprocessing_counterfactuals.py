import json

import numpy as np
import pytest

from diarize import pipeline


def _run(tmp_path, monkeypatch, segments, initial_labels):
    audio, model = tmp_path / "sample.wav", tmp_path / "model.pt"
    audio.write_bytes(b"audio")
    model.write_bytes(b"model")
    monkeypatch.setattr(pipeline, "load_audio", lambda _: (np.zeros(32000), 16000))
    monkeypatch.setattr(pipeline, "get_speech_segments", lambda *a, **k: [(0., 2.)])
    embeddings = np.eye(len(segments), dtype=float)
    monkeypatch.setattr(pipeline, "extract_embeddings_for_segments",
                        lambda *a: (segments, embeddings))
    monkeypatch.setattr(pipeline, "cluster_embeddings",
                        lambda *a, **k: np.asarray(initial_labels))
    output = pipeline.diarize(
        str(audio), str(model), encoder=object(),
        diagnostic_output_dir=str(tmp_path / "diagnostics"),
        diagnostic_counterfactuals=True)
    directory = tmp_path / "diagnostics" / "counterfactuals" / "sample"
    trace = json.loads((directory / "pipeline_trace.json").read_text(encoding="utf-8"))
    return output, directory, trace


def test_counterfactuals_reuse_labels_and_do_not_change_baseline(tmp_path, monkeypatch):
    segments = [(0., .5), (.5, 1.), (1., 1.5), (1.5, 2.)]
    output, directory, trace = _run(tmp_path, monkeypatch, segments, [0, 1, 0, 0])
    baseline = (directory / "baseline.rttm").read_text(encoding="utf-8").strip()
    no_smoothing = (directory / "without_temporal_smoothing.rttm").read_text(
        encoding="utf-8")
    assert output == baseline
    assert "SPEAKER_01" not in baseline
    assert "SPEAKER_01" in no_smoothing
    assert set(trace) == {"baseline", "without_temporal_smoothing",
                          "without_short_segment_merging"}


def test_without_short_merge_preserves_a_short_turn(tmp_path, monkeypatch):
    _, directory, _ = _run(tmp_path, monkeypatch, [(0., .2), (.2, .4)], [0, 1])
    baseline = (directory / "baseline.rttm").read_text(encoding="utf-8")
    no_short_merge = (directory / "without_short_segment_merging.rttm").read_text(
        encoding="utf-8")
    assert "SPEAKER_01" not in baseline
    assert "SPEAKER_01" in no_short_merge


def test_counterfactuals_require_an_output_directory():
    with pytest.raises(ValueError, match="requires diagnostic_output_dir"):
        pipeline.diarize("unused.wav", "unused.model", diagnostic_counterfactuals=True)
