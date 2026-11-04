"""Integration boundaries for automatic speaker-count experiments."""
from argparse import Namespace

import numpy as np

from diarize import pipeline
from diarize.config import load_config, pipeline_kwargs
from experiments import run_diarization_experiments as runner


def _mock_embeddings(monkeypatch, embeddings):
    monkeypatch.setattr(
        pipeline, "load_audio", lambda _: (np.zeros(16000 * len(embeddings)), 16000)
    )
    monkeypatch.setattr(
        pipeline,
        "get_speech_segments",
        lambda *args, **kwargs: [(0.0, float(len(embeddings)))],
    )
    segments = [(float(index), float(index + 1)) for index in range(len(embeddings))]
    monkeypatch.setattr(
        pipeline,
        "extract_embeddings_for_segments",
        lambda *args, **kwargs: (segments, np.asarray(embeddings, dtype=float)),
    )


def _runner_args(experiment):
    return Namespace(
        method="ahc",
        threshold=0.65,
        count_estimator=None,
        vad_threshold=0.5,
        window_sec=1.5,
        step_sec=None,
        speaker_mode="automatic",
        experiment=experiment,
    )


def test_default_configuration_keeps_the_frozen_output(monkeypatch):
    _mock_embeddings(monkeypatch, [[1.0, 0.0]] * 4 + [[0.8, 0.6]] * 4)

    kwargs = pipeline_kwargs(load_config())
    assert kwargs["count_estimator"] is None
    assert pipeline.diarize("sample.wav", "unused", encoder=object(), **kwargs) == (
        "SPEAKER sample 1 0.000 8.000 <NA> <NA> SPEAKER_00 <NA> <NA>"
    )


def test_adaptive_ahc_is_opt_in_and_reuses_only_embeddings(tmp_path, monkeypatch):
    embeddings = np.vstack(
        [np.tile([1.0, 0.0, 0.0], (3, 1)),
         np.tile([0.0, 1.0, 0.0], (3, 1)),
         np.tile([0.0, 0.0, 1.0], (3, 1))]
    )
    _mock_embeddings(monkeypatch, embeddings)
    audio = tmp_path / "sample.wav"
    model = tmp_path / "model.model"
    audio.write_bytes(b"synthetic audio identity")
    model.write_bytes(b"synthetic checkpoint identity")
    cache = {}
    fixed_trace, adaptive_trace = {}, {}
    encoder = object()

    pipeline.diarize(str(audio), str(model), encoder=encoder, embedding_cache=cache,
                     diagnostics=fixed_trace)
    pipeline.diarize(str(audio), str(model), encoder=encoder, embedding_cache=cache,
                     diagnostics=adaptive_trace, count_estimator="adaptive_ahc")

    assert fixed_trace["clustering"]["ahc"]["estimator"] == "fixed_threshold"
    assert adaptive_trace["clustering"]["ahc"]["estimator"] == "adaptive_ahc"
    assert fixed_trace["embedding_cache_hit"] is False
    assert adaptive_trace["embedding_cache_hit"] is True


def test_fair_comparison_never_injects_reference_counts_into_automatic_backends():
    comparison = runner.configurations(_runner_args("clustering_count"))
    assert [(row["method"], row["count_estimator"]) for row in comparison] == [
        ("ahc", None),
        ("ahc", "adaptive_ahc"),
        ("spectral", None),
        ("nme_sc", None),
    ]
    assert {row["speaker_mode"] for row in comparison} == {"automatic"}

    oracle_diagnostic = runner.configurations(_runner_args("speaker_count"))
    assert {row["speaker_mode"] for row in oracle_diagnostic} == {"automatic", "known"}
    assert next(row for row in oracle_diagnostic if row["speaker_mode"] == "known")[
        "count_estimator"
    ] is None
