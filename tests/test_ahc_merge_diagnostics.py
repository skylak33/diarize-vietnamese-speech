import pytest

from experiments.analyze_ahc_merge_quality import (
    _configuration_labels,
    _critical_recordings,
    trace_reference_merges,
)


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
    assert summary["selected_cut_below_oracle"]
    assert summary["selected_cut_ari_loss_vs_oracle"] > 0


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


def test_oracle_cut_sweep_is_bounded_to_relevant_counts():
    children = [[0, 1]]
    active = 12
    for leaf in range(2, 12):
        children.append([active, leaf])
        active = 12 + len(children) - 1
    summary, _, cuts = trace_reference_merges(
        children, list(range(11)), ["A"] * 6 + ["B"] * 6,
        selected_n_clusters=2, max_cut_clusters=3)
    assert [row["n_clusters"] for row in cuts] == [1, 2, 3]
    assert summary["oracle_best_cut"]["n_clusters"] <= 3


def test_saved_ahc_labels_are_reused_without_reclustering():
    labels, source = _configuration_labels(
        {"clustering_diagnostics": {"ahc": {"labels": [3, 3, 8]}}},
        [[1, 0], [1, 0], [0, 1]],
        {"method": "invalid_if_called", "threshold": .7, "count_estimator": None}, 42, 10)
    assert labels.tolist() == [3, 3, 8]
    assert source == "saved_ahc_trace"


def test_critical_recordings_select_largest_regressions_and_vi_000004():
    rows = []
    for index, delta in enumerate((1, 8, 4, 3, 2, 7), 1):
        recording_id = f"vi_{index:06d}"
        rows.extend([
            {"mode": "full_pipeline", "overlap_condition": "included",
             "configuration": "fixed_ahc_075", "recording_id": recording_id, "DER": 10},
            {"mode": "full_pipeline", "overlap_condition": "included",
             "configuration": "robust_adaptive_ahc_075", "recording_id": recording_id,
             "DER": 10 + delta},
        ])
    selected, regressions = _critical_recordings(rows)
    assert [row["robust_minus_fixed_der"] for row in regressions] == [8, 7, 4, 3, 2]
    assert selected[0] == "vi_000004"
    assert set(selected) == {"vi_000002", "vi_000003", "vi_000004", "vi_000005", "vi_000006"}
