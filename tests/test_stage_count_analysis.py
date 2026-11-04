import csv
import json

import pytest

from experiments import analyze_stage_counts as analysis


def _rttm(path, recording_id, speakers):
    path.write_text("".join(
        f"SPEAKER {recording_id} 1 {index} 1 <NA> <NA> {speaker} <NA> <NA>\n"
        for index, speaker in enumerate(speakers)), encoding="utf-8")


def _artifact(root, recording_id, counts, n_speakers=None, threshold=.65):
    directory = root / recording_id
    directory.mkdir(parents=True)
    stages = {}
    for name, count in zip(analysis.STAGE_NAMES, counts):
        stages[name] = {"labels": list(range(count))}
    (directory / "pipeline_trace.json").write_text(json.dumps({
        "schema_version": 1, "recording_id": recording_id, "stages": stages,
    }), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({
        "schema_version": 1,
        "recording_id": recording_id,
        "inference_configuration": {
            "n_speakers": n_speakers,
            "method": "ahc",
            "threshold": threshold,
            "window_sec": 1.5,
            "step_sec": .75,
        },
    }), encoding="utf-8")


def _metrics(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "DER", "speaker_error"])
        writer.writeheader()
        writer.writerows(rows)


def test_evaluation_loader_ignores_corpus_total_row(tmp_path):
    metrics = tmp_path / "metrics.csv"
    _metrics(metrics, [
        {"file": "sample", "DER": 1, "speaker_error": 2},
        {"file": "__total__", "DER": 1, "speaker_error": 2},
    ])

    assert analysis._load_evaluation(metrics) == {
        "sample": {"DER": 1.0, "SpeakerError": 2.0}}


def test_build_report_attributes_count_errors_and_oracle_identity_loss(tmp_path):
    automatic, oracle, references = (tmp_path / name for name in ("automatic", "oracle", "ref"))
    references.mkdir()
    _rttm(references / "a.rttm", "a", ["A", "B"])
    _rttm(references / "b.rttm", "b", ["A", "B"])
    _artifact(automatic, "a", (3, 2, 2, 2))
    _artifact(automatic, "b", (2, 2, 1, 1))
    _artifact(oracle, "a", (2, 2, 2, 2), n_speakers=2)
    _artifact(oracle, "b", (2, 2, 1, 1), n_speakers=2)
    metrics = tmp_path / "metrics.csv"
    _metrics(metrics, [
        {"file": "a", "DER": 12.5, "speaker_error": 4.0},
        {"file": "b", "DER": 8.0, "speaker_error": 0.0},
    ])

    report = analysis.build_report(
        automatic, references, metrics, expected_recordings=2, oracle_artifacts=oracle)

    assert report["recordings"] == [{
        "recording": "a", "reference_K": 2, "initial_K": 3,
        "after_centroid_K": 2, "after_smoothing_K": 2, "final_K": 2,
        "DER": 12.5, "SpeakerError": 4.0,
    }, {
        "recording": "b", "reference_K": 2, "initial_K": 2,
        "after_centroid_K": 2, "after_smoothing_K": 1, "final_K": 1,
        "DER": 8.0, "SpeakerError": 0.0,
    }]
    assert report["summary"]["count_error_origins"] == {
        "initial_clustering": 1, "centroid_merge": 0, "temporal_smoothing": 1,
        "final_postprocessing": 0, "unresolved_postprocessing": 0, "none": 0,
    }
    assert report["summary"]["postprocessing_count_effect"] == {
        "improved": 1, "worsened": 1, "unchanged": 0,
    }
    assert report["summary"]["count_error_transition"] == {
        "initial_errors": 1, "final_errors": 1, "corrected": 1,
        "introduced": 1, "remained": 0,
    }
    assert report["summary"]["stage_count_changes"]["temporal_smoothing"] == {
        "comparable": 2, "decreased": 1, "increased": 0, "unchanged": 1,
        "labels_removed": 1, "labels_added": 0,
    }
    assert report["summary"]["correct_final_count_with_speaker_error"] == {
        "count": 1, "recordings": ["a"],
    }
    assert report["summary"]["oracle_count_identity_elimination"] == {
        "comparable_recordings": 2, "count": 1, "recordings": ["b"],
    }


def test_writes_csv_and_json_without_modifying_inputs(tmp_path):
    artifacts, references = tmp_path / "artifacts", tmp_path / "ref"
    references.mkdir()
    _rttm(references / "sample.rttm", "sample", ["A"])
    _artifact(artifacts, "sample", (1, 1, 1, 1))
    metrics = tmp_path / "metrics.csv"
    _metrics(metrics, [{"file": "sample", "DER": 0, "speaker_error": 0}])
    output = tmp_path / "report"

    analysis.main([
        "--artifacts", str(artifacts), "--reference-rttm", str(references),
        "--evaluation", str(metrics), "--output-dir", str(output),
        "--expected-recordings", "1",
    ])

    assert list(csv.DictReader((output / "stage_counts.csv").open(encoding="utf-8")))[0][
        "recording"] == "sample"
    assert json.loads((output / "stage_count_analysis.json").read_text(encoding="utf-8"))[
        "summary"]["recordings"] == 1
    assert sorted(path.name for path in artifacts.rglob("*")) == [
        "config.json", "pipeline_trace.json", "sample"]


@pytest.mark.parametrize("mutation,match", [
    ("missing_metric", "IDs differ"),
    ("trace_id", "recording_id"),
    ("automatic_oracle", "Automatic artifact uses n_speakers"),
    ("bad_stage", "after_smoothing"),
])
def test_fails_loudly_on_inconsistent_ids_schema_or_reference_leakage(tmp_path, mutation, match):
    artifacts, references = tmp_path / "artifacts", tmp_path / "ref"
    references.mkdir()
    _rttm(references / "sample.rttm", "sample", ["A"])
    _artifact(artifacts, "sample", (1, 1, 1, 1), n_speakers=1 if mutation == "automatic_oracle" else None)
    trace_path = artifacts / "sample" / "pipeline_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if mutation == "trace_id":
        trace["recording_id"] = "other"
    if mutation == "bad_stage":
        del trace["stages"]["after_smoothing"]
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    metrics = tmp_path / "metrics.csv"
    _metrics(metrics, [] if mutation == "missing_metric" else [
        {"file": "sample", "DER": 0, "speaker_error": 0}])

    with pytest.raises(ValueError, match=match):
        analysis.build_report(artifacts, references, metrics, expected_recordings=1)


def test_oracle_comparison_requires_same_non_count_configuration(tmp_path):
    automatic, oracle, references = (tmp_path / name for name in ("automatic", "oracle", "ref"))
    references.mkdir()
    _rttm(references / "sample.rttm", "sample", ["A"])
    _artifact(automatic, "sample", (1, 1, 1, 1))
    _artifact(oracle, "sample", (1, 1, 1, 1), n_speakers=1, threshold=.75)
    metrics = tmp_path / "metrics.csv"
    _metrics(metrics, [{"file": "sample", "DER": 0, "speaker_error": 0}])

    with pytest.raises(ValueError, match="not comparable"):
        analysis.build_report(
            automatic, references, metrics, expected_recordings=1,
            oracle_artifacts=oracle)


def test_runner_diagnostics_preserve_unavailable_legacy_smoothing_count(tmp_path):
    automatic, oracle, references = (tmp_path / name for name in ("automatic", "oracle", "ref"))
    automatic.mkdir(); oracle.mkdir(); references.mkdir()
    _rttm(references / "sample.rttm", "sample", ["A", "B"])
    common = {"method": "ahc", "ahc_threshold": .75, "window_sec": 1.5}
    (automatic / "config.json").write_text(json.dumps({
        **common, "speaker_mode": "automatic"}), encoding="utf-8")
    (oracle / "config.json").write_text(json.dumps({
        **common, "speaker_mode": "known"}), encoding="utf-8")
    legacy = [{
        "recording_id": "sample", "n_clusters_before_postmerge": 2,
        "n_clusters_after_postmerge": 2, "n_speakers_after_postprocess": 1,
    }]
    (automatic / "diagnostics.json").write_text(json.dumps(legacy), encoding="utf-8")
    (oracle / "diagnostics.json").write_text(json.dumps(legacy), encoding="utf-8")
    metrics = tmp_path / "metrics.csv"
    _metrics(metrics, [{"file": "sample", "DER": 10, "speaker_error": 5}])

    report = analysis.build_report(
        automatic, references, metrics, expected_recordings=1, oracle_artifacts=oracle)

    assert report["recordings"][0]["after_smoothing_K"] is None
    assert report["summary"]["count_error_origins"]["unresolved_postprocessing"] == 1
    assert report["summary"]["oracle_count_identity_elimination"]["recordings"] == ["sample"]
