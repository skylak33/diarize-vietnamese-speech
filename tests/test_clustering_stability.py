import csv
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from experiments.analyze_clustering_stability import (
    NOISE_RADIANS,
    NOISE_SEEDS,
    PREFIX_WINDOWS,
    SUBSAMPLE_FRACTIONS,
    analyze_embeddings,
    analyze_synthetic_controls,
    angular_noise,
    changed_assignment_fraction,
    cluster_method,
    ordered_subsample_indices,
    run,
    summarize,
)


def test_angular_noise_is_deterministic_normalized_and_bounded():
    embeddings = np.eye(4)
    first = angular_noise(embeddings, .05, 11)
    second = angular_noise(embeddings, .05, 11)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first, axis=1) == pytest.approx(1)
    assert np.sum(first * embeddings, axis=1) == pytest.approx(np.cos(.05))


def test_partition_disagreement_is_label_permutation_invariant():
    assert changed_assignment_fraction([0, 0, 1, 1], [7, 7, 3, 3]) == 0
    assert changed_assignment_fraction([0, 0, 1, 1], [0, 1, 0, 1]) == .5


def test_ordered_subsampling_preserves_endpoints_and_order():
    indices = ordered_subsample_indices(10, .5)
    assert indices.tolist() == sorted(indices.tolist())
    assert indices[0] == 0 and indices[-1] == 9 and len(indices) == 5


def test_short_and_identical_inputs_have_explicit_fallback_diagnostics():
    identical = np.ones((3, 4))
    for method in ("fixed_0.65", "legacy_adaptive_0.75", "robust_adaptive_0.75"):
        labels, trace = cluster_method(identical, method)
        assert len(labels) == 3
        assert trace["estimated_k"] == 1
    _, robust = cluster_method(np.eye(3), "robust_adaptive_0.75")
    assert robust["selection_reason"] == "insufficient_candidate_gaps"
    assert robust["selection"] == "fixed_threshold_fallback"
    labels, trace = cluster_method(np.array([[1.0, 0, 0]]), "robust_adaptive_0.75")
    assert labels.tolist() == [0] and trace["estimated_k"] == 1


def test_predefined_perturbations_and_summaries_are_complete():
    embeddings = np.vstack((
        np.array([[1, 0, 0], [.99, .03, 0], [.98, -.04, 0]]) * 1.0,
        np.array([[0, 1, 0], [.03, .99, 0], [-.04, .98, 0]]) * 1.0,
    ))
    rows = analyze_embeddings(embeddings, "recording", "mode")
    expected_per_method = len(NOISE_SEEDS) * len(NOISE_RADIANS) + len(SUBSAMPLE_FRACTIONS)
    expected_per_method += sum(count < len(embeddings) for count in PREFIX_WINDOWS)
    assert len(rows) == 3 * expected_per_method
    assert all(row["partition_ari_vs_unperturbed"] is not None for row in rows)
    counts, partitions = summarize(rows)
    assert len(counts) == len(partitions) == 9
    assert {row["perturbation_family"] for row in counts} == {
        "angular_noise", "ordered_subsample", "short_prefix"}


def test_synthetic_controls_cover_required_cases_and_wrong_partition():
    controls = analyze_synthetic_controls()
    assert set(controls["cases"]) == {
        "clear_separation", "fragmented_speaker", "similar_speakers", "outlier",
        "unequal_sizes", "short_hierarchy", "correct_k_wrong_partition"}
    wrong = controls["cases"]["correct_k_wrong_partition"]["known_k_control"]
    assert wrong["estimated_k"] == 2
    assert wrong["demonstrates_correct_k_wrong_partition"]
    assert wrong["reference_ari"] < 1


def test_run_requires_complete_manifest_and_writes_all_outputs(tmp_path):
    cache = tmp_path / "embeddings.npz"
    embeddings = np.vstack((np.eye(3), np.eye(3)))
    np.savez(cache, embeddings=embeddings, window_starts=np.arange(6), window_ends=np.arange(6) + 1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "complete",
        "caches": [{"recording_id": "r", "mode": "m", "embedding_windows": 6,
                    "cache_path": str(cache)}],
    }), encoding="utf-8")
    output = tmp_path / "out"
    summary = run(manifest, output)
    assert summary["cache_count"] == 1
    for name in ("perturbation_results.csv", "count_stability.csv",
                 "partition_stability.csv", "synthetic_controls.json"):
        assert (output / name).is_file()
    with (output / "perturbation_results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and "gap_margin" in rows[0] and "selection_reason" in rows[0]
    assert "baseline_selected_gap_index" in rows[0] and "selected_gap_changed" in rows[0]


def test_incomplete_manifest_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status": "running", "caches": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        run(manifest, tmp_path / "out")
