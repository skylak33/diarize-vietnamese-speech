import pytest

from experiments.analyze_ahc_merge_quality import trace_reference_merges


def test_pure_merges_then_crossing_and_correct_cut():
    summary, merges, cuts = trace_reference_merges(
        [[0, 1], [2, 3], [4, 5]], [.1, .2, .9], ["A", "A", "B", "B"], 2)
    assert [row["reference_crossing"] for row in merges] == [False, False, True]
    assert merges[-1]["purity_before_merge"] == 1
    assert merges[-1]["purity_after_merge"] == .5
    assert summary["earliest_reference_crossing_merge"] == 2
    assert summary["cross_speaker_merges_before_selected_cut"] == 0
    assert summary["pure_clusters_before_selected_cut"] == 2
    assert summary["failure_type"] == "none"
    assert next(row for row in cuts if row["n_clusters"] == 2)["ari"] == 1


def test_incorrect_cut_when_hierarchy_contains_reference_partition():
    summary, _, _ = trace_reference_merges(
        [[0, 1], [2, 3], [4, 5]], [.1, .2, .9], ["A", "A", "B", "B"], 1)
    assert summary["failure_type"] == "incorrect_hierarchy_cut"
    assert summary["cross_speaker_merges_before_selected_cut"] == 1
    assert summary["perfect_reference_partition_available"]


def test_early_cross_speaker_merge_is_irreversible():
    summary, merges, _ = trace_reference_merges(
        [[0, 2], [1, 3], [4, 5]], [.1, .2, .9], ["A", "A", "B", "B"], 2)
    assert merges[0]["introduces_mixing"]
    assert summary["earliest_reference_crossing_merge"] == 0
    assert summary["failure_type"] == "incorrect_early_merge"
    assert not summary["perfect_reference_partition_available"]


def test_degenerate_single_leaf_hierarchy():
    summary, merges, cuts = trace_reference_merges([], [], ["A"], 1)
    assert merges == []
    assert cuts[0]["ari"] == 1
    assert summary["failure_type"] == "none"


def test_tied_merge_distances_flag_ambiguous_cut_boundary():
    summary, _, _ = trace_reference_merges(
        [[0, 1], [2, 3], [4, 5]], [.2, .2, .8], ["A", "A", "B", "B"], 3)
    assert summary["selected_cut_tied_distance"]


def test_ambiguous_leaf_does_not_create_false_crossing():
    summary, merges, _ = trace_reference_merges(
        [[0, 1], [2, 3], [4, 5]], [.1, .2, .9], ["A", None, "B", "B"], 2)
    assert not merges[0]["reference_crossing"]
    assert summary["ambiguous_or_unlabeled_leaves"] == 1


@pytest.mark.parametrize("children,distances", [([[0, 1]], []), ([[0, 0]], [.1])])
def test_invalid_hierarchy_rejected(children, distances):
    with pytest.raises(ValueError):
        trace_reference_merges(children, distances, ["A", "B"], 1)
