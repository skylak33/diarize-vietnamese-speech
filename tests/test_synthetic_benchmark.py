import json

import numpy as np
import pytest

from experiments import run_synthetic_diarization_benchmark as benchmark
from experiments import analyze_synthetic_benchmark as analysis


def _conversation(root, recording_id, split="holdout"):
    (root / f"{recording_id}.wav").write_bytes(b"wav")
    (root / f"{recording_id}.rttm").write_text(
        f"SPEAKER {recording_id} 1 0 1 <NA> <NA> A <NA> <NA>\n", encoding="utf-8")
    path = root / f"{recording_id}.json"
    path.write_text(json.dumps({"conversation_id": recording_id, "split": split,
        "scenario": "clean", "speaker_count": 1, "speaker_ids": ["A"],
        "source_datasets": ["fixture"],
        "turns": [{"start": 0., "end": 1., "source_dataset": "fixture"}]}),
        encoding="utf-8")
    return path


def test_exact_five_automatic_count_configurations():
    assert [item["name"] for item in benchmark.CONFIGURATIONS] == [
        "fixed_ahc_075", "legacy_adaptive_ahc_075", "robust_adaptive_ahc_075",
        "spectral", "nme_sc"]
    assert all(item["threshold"] == .75 for item in benchmark.CONFIGURATIONS)
    assert all("n_speakers" not in item for item in benchmark.CONFIGURATIONS)


def test_conversation_discovery_selection_and_dry_run(tmp_path):
    _conversation(tmp_path, "b")
    _conversation(tmp_path, "a")
    (tmp_path / "manifest.json").write_text('{"schema_version": 1}')
    records = benchmark.load_conversations(tmp_path)
    assert [record["conversation_id"] for record in records] == ["a", "b"]
    assert benchmark.select_conversations(records, smoke=True)[0]["conversation_id"] == "a"
    with pytest.raises(ValueError, match="Unknown recording"):
        benchmark.select_conversations(records, ["missing"])
    args = benchmark.parse_args(["--conversation-json", str(tmp_path), "--model", "unused",
                                 "--split", "holdout", "--dry-run", "--recording-ids", "b"])
    assert benchmark.run(args)["recording_ids"] == ["b"]


def test_split_filter_prevents_mixed_runs(tmp_path):
    _conversation(tmp_path, "dev", "development")
    _conversation(tmp_path, "test", "holdout")
    args = benchmark.parse_args(["--conversation-json", str(tmp_path), "--model", "unused",
                                 "--split", "development", "--dry-run"])
    assert benchmark.run(args)["recording_ids"] == ["dev"]


def test_reference_regions_and_unambiguous_partition_metrics():
    reference = [(0., 2., "A"), (1.5, 3., "B"), (4., 5., "A")]
    assert benchmark.reference_speech_regions(reference) == [(0., 3.), (4., 5.)]
    windows = [(0., 1.), (1.5, 2.), (2., 3.), (4., 5.)]
    indexes, labels = benchmark.assign_reference_labels(windows, reference)
    assert indexes.tolist() == [0, 2, 3] and labels == ["A", "B", "A"]
    metrics = benchmark.partition_metrics(windows, np.array([0, 9, 1, 0]), reference)
    assert metrics == {"compatible_windows": 3, "ARI": 1., "NMI": 1.,
                       "purity": 1., "fragmentation": 1.}


def test_postprocess_calls_production_functions_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "smooth_labels",
                        lambda segments, labels, window: calls.append(("smooth", window)) or labels)
    monkeypatch.setattr(benchmark, "partition_overlapping_windows",
                        lambda segments: calls.append(("partition",)) or segments)
    monkeypatch.setattr(benchmark, "merge_short_segments",
                        lambda segments, labels, min_duration: calls.append(("short", min_duration)) or (segments, labels))
    monkeypatch.setattr(benchmark, "merge_consecutive_segments",
                        lambda segments, labels, gap_threshold: calls.append(("consecutive", gap_threshold)) or (segments, labels))
    benchmark.postprocess([(0., 1.)], np.array([0]))
    assert calls == [("smooth", 3), ("partition",), ("short", .3), ("consecutive", 0.)]


def test_runner_loads_one_encoder_and_reuses_embeddings_across_methods(tmp_path, monkeypatch):
    metadata = _conversation(tmp_path, "sample")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen")
    calls = {"model": 0, "vad": 0, "extract": 0, "cluster": 0}

    def load_model(*_args):
        calls["model"] += 1
        return object()

    def vad(*_args, **_kwargs):
        calls["vad"] += 1
        return [(0., 1.)]

    def extract(*_args):
        calls["extract"] += 1
        return [(0., 1.)], np.array([[1., 0.]])

    def cluster(embeddings, **kwargs):
        calls["cluster"] += 1
        assert kwargs["n_speakers"] is None
        if kwargs["diagnostics"] is not None and kwargs["method"] == "ahc":
            kwargs["diagnostics"]["ahc"] = {"distances": [], "gaps": []}
        return np.zeros(len(embeddings), dtype=int)

    metric = {"DER": 0., "miss": 0., "false_alarm": 0., "speaker_error": 0.,
              "duration": 1., "miss_seconds": 0., "false_alarm_seconds": 0.,
              "speaker_error_seconds": 0.,
              "N_ref": 1, "N_pred": 1, "exact_match": True, "abs_error": 0}
    total = {"DER": 0., "miss": 0., "false_alarm": 0., "speaker_error": 0.,
             "speaker_count_accuracy": 1., "speaker_count_mae": 0.,
             "overclustering_rate": 0., "underclustering_rate": 0.}
    monkeypatch.setattr("utils.model.load_model", load_model)
    monkeypatch.setattr("diarize.vad.load_audio", lambda _path: (np.zeros(16000), 16000))
    monkeypatch.setattr("diarize.vad.get_speech_segments", vad)
    monkeypatch.setattr(benchmark, "extract_embeddings_for_segments", extract)
    monkeypatch.setattr(benchmark, "cluster_embeddings", cluster)
    monkeypatch.setattr(benchmark, "evaluate_annotations",
                        lambda *_args, **_kwargs: {"sample": metric, "__total__": total})
    args = benchmark.parse_args(["--conversation-json", str(metadata), "--model",
                                 str(checkpoint), "--split", "holdout",
                                 "--output-dir", str(tmp_path / "out")])
    result = benchmark.run(args)
    assert calls == {"model": 1, "vad": 1, "extract": 2, "cluster": 10}
    assert len(result["recordings"]) == 10
    assert len({row["embedding_sha256"] for row in result["recordings"]
                if row["mode"] == "clustering_only"}) == 1
    row = result["recordings"][0]
    assert (row["split"], row["scenario"], row["reference_speaker_count"],
            row["duration_seconds"], row["mean_turn_duration_seconds"], row["source_datasets"]) == (
                "holdout", "clean", 1, 1., 1., ["fixture"])


def _result(configuration, recording_id, der, speaker_count=2, scenario="clean",
            source_datasets=None, compatible=2, ari=1.):
    return {"split": "holdout", "mode": "clustering_only", "configuration": configuration,
            "recording_id": recording_id, "scenario": scenario,
            "reference_speaker_count": speaker_count,
            "source_datasets": source_datasets or ["fixture"], "DER": der,
            "duration_seconds": 40., "mean_turn_duration_seconds": 2.,
            "scored_duration_seconds": 10., "miss_seconds": der / 10,
            "false_alarm_seconds": 0., "speaker_error_seconds": 0.,
            "count_correct": True, "count_absolute_error": 0,
            "overclustered": False, "underclustered": False, "runtime_seconds": 1.,
            "compatible_windows": compatible, "ARI": ari, "NMI": ari,
            "purity": ari, "fragmentation": 1.}


def test_summary_weighting_excludes_none_from_denominator():
    rows = [_result("fixed_ahc_075", "a", 0., compatible=2, ari=1.),
            _result("fixed_ahc_075", "b", 0., compatible=8, ari=None)]
    total = {"DER": 0., "miss": 0., "false_alarm": 0., "speaker_error": 0.,
             "speaker_count_accuracy": 1., "speaker_count_mae": 0.,
             "overclustering_rate": 0., "underclustering_rate": 0.}
    assert benchmark._summary("clustering_only", benchmark.CONFIGURATIONS[0], rows, total)["ARI"] == 1.


def test_grouped_analysis_and_paired_robust_regressions(tmp_path):
    rows = [_result("fixed_ahc_075", "a", 10.),
            _result("robust_adaptive_ahc_075", "a", 15.),
            _result("fixed_ahc_075", "b", 20., scenario="overlap"),
            _result("robust_adaptive_ahc_075", "b", 12., scenario="overlap")]
    path = tmp_path / "recording_results.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    grouped, regressions, summary = analysis.analyze(path)
    assert {row["group_type"] for row in grouped} == {
        "speaker_count", "scenario", "source_dataset", "duration_seconds",
        "mean_turn_duration_seconds"}
    assert {row["group_value"] for row in grouped
            if row["group_type"] == "duration_seconds"} == {"<45"}
    assert regressions[0]["recording_id"] == "a"
    assert regressions[0]["DER_delta_robust_minus_fixed"] == 5.
    assert summary["improved"] == summary["worsened"] == 1
    assert summary["by_reference_speaker_count"]["2"]["paired_recordings"] == 2
    assert (tmp_path / "grouped_comparisons.csv").is_file()
    assert (tmp_path / "robust_vs_fixed_regressions.csv").is_file()
    assert (tmp_path / "robust_vs_fixed_summary.json").is_file()
