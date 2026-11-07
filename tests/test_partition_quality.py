import pytest

from experiments.analyze_partition_quality import (
    associate_reference_windows, partition_quality)


@pytest.mark.parametrize("predicted", [[0, 0, 1, 1], ["b", "b", "a", "a"]])
def test_perfect_and_permuted_partitions(predicted):
    summary, clusters, speakers = partition_quality(["A", "A", "B", "B"], predicted)
    assert summary["ari"] == pytest.approx(1)
    assert summary["nmi"] == pytest.approx(1)
    assert summary["duration_weighted_cluster_purity"] == 1
    assert len(clusters) == len(speakers) == 2


def test_correct_count_can_have_wrong_partition():
    summary, clusters, _ = partition_quality(["A", "A", "B", "B"], [0, 1, 0, 1])
    assert summary["absolute_count_error"] == 0
    assert summary["ari"] < 0
    assert summary["duration_weighted_cluster_purity"] == .5
    assert all(row["entropy_bits"] == 1 for row in clusters)


def test_over_and_under_clustering_are_separate_from_purity():
    over, _, speakers = partition_quality(["A", "A", "B", "B"], [0, 1, 2, 2])
    under, _, _ = partition_quality(["A", "A", "B", "B"], [0, 0, 0, 0])
    assert over["absolute_count_error"] == under["absolute_count_error"] == 1
    assert over["duration_weighted_cluster_purity"] == 1
    assert over["duration_weighted_speaker_fragmentation"] == 1.5
    assert under["duration_weighted_cluster_purity"] == .5
    assert speakers[0]["predicted_cluster_count"] == 2


def test_duration_weighting_and_postprocessing_counts():
    summary, _, _ = partition_quality(
        ["A", "B", "B"], [0, 0, 1], durations=[8, 1, 1],
        initial_predicted_count=4, final_predicted_count=2)
    assert summary["duration_weighted_cluster_purity"] == .9
    assert summary["predicted_identities_eliminated"] == 2


def test_ambiguous_windows_are_explicit_and_excluded():
    associated = associate_reference_windows(
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1.5, "A"), (1.25, 2.5, "B")])
    assert [row["status"] for row in associated] == [
        "single_speaker", "ambiguous", "single_speaker"]
    summary, _, _ = partition_quality(
        [row["reference_label"] for row in associated], [0, 1, 1])
    assert summary["reliable_windows"] == 2
    assert summary["excluded_ambiguous_or_unlabeled_windows"] == 1
    assert summary["ari"] == pytest.approx(1)


def test_all_ambiguous_windows_return_undefined_oracle_metrics():
    summary, clusters, speakers = partition_quality([None], [0])
    assert summary["ari"] is None
    assert clusters == speakers == []

