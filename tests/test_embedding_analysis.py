import json

import numpy as np

from experiments.analyze_embedding_diagnostics import (analyze_recording,
                                                         reference_distinct_merges)


def _artifacts(path):
    path.mkdir()
    embeddings = np.array([[1., 0.], [.99, .01], [0., 1.], [.01, .99]])
    np.savez(path / "embeddings.npz", embeddings=embeddings,
             window_starts=np.arange(4.), window_ends=np.arange(1., 5.))
    stages = {
        "initial_ahc": {"labels": [0, 0, 1, 1]},
        "after_centroid_merge": {"labels": [0, 0, 1, 1]},
        "after_smoothing": {"labels": [0, 0, 0, 1]},
        "partitioned_segments": {"segments": [], "labels": [0, 0, 0, 1]},
        "segments_after_short_merge": {"segments": [], "labels": [0, 0, 0, 1]},
        "final_rttm_segments": {"segments": [[0., 3.], [3., 4.]], "labels": [0, 1]},
    }
    (path / "pipeline_trace.json").write_text(json.dumps({
        "schema_version": 1, "recording_id": "sample", "stages": stages}), encoding="utf-8")
    (path / "config.json").write_text(json.dumps({
        "schema_version": 1, "reference_source": "published_reference"}), encoding="utf-8")
    ahc = {"distances": [.01, .01, 1.], "merges": [
        {"merge_step": 0, "left_cluster_id": 0, "right_cluster_id": 1,
         "merge_distance": .01, "merged_cluster_size": 2},
        {"merge_step": 1, "left_cluster_id": 2, "right_cluster_id": 3,
         "merge_distance": .01, "merged_cluster_size": 2},
        {"merge_step": 2, "left_cluster_id": 4, "right_cluster_id": 5,
         "merge_distance": 1., "merged_cluster_size": 4},
    ]}
    (path / "ahc_merges.json").write_text(json.dumps({"ahc": ahc}), encoding="utf-8")


def test_reference_distinct_merge_uses_reliable_window_labels():
    ahc = {"merges": [
        {"merge_step": 0, "left_cluster_id": 0, "right_cluster_id": 1, "merge_distance": .1},
        {"merge_step": 1, "left_cluster_id": 2, "right_cluster_id": 3, "merge_distance": .2},
    ]}
    classifications = [
        {"classification": "unambiguous_speaker_window", "speaker": "A"},
        {"classification": "unambiguous_speaker_window", "speaker": "B"},
        {"classification": "unannotated_window", "speaker": None},
    ]
    merges = reference_distinct_merges(ahc, classifications)
    assert len(merges) == 1
    assert merges[0]["merge_distance"] == .1
    assert merges[0]["mixed_cluster_existed_before_merge"] is False


def test_analysis_writes_statistics_and_keeps_causes_unconfirmed(tmp_path):
    artifact_dir = tmp_path / "sample"
    _artifacts(artifact_dir)
    reference = [(0., 2., "A"), (2., 4., "B")]
    result = analyze_recording(artifact_dir, reference, count_relation="correct_count")
    assert result["candidate_causes"] == ["temporal_smoothing"]
    assert result["confirmed_causes"] == []
    assert result["evidence"]["ahc_merge_behavior"]["reference_distinct_merges"][-1]["merge_distance"] == 1.
    assert (artifact_dir / "similarity_stats.json").is_file()
    assert (artifact_dir / "ahc_cluster_summary.csv").is_file()
