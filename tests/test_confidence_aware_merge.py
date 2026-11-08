import copy

import numpy as np
import pytest

from diarize.cluster import merge_clusters_confidence_aware, merge_similar_clusters


def _directions(*degrees):
    radians = np.deg2rad(degrees)
    return np.column_stack((np.cos(radians), np.sin(radians)))


def _timestamps(count):
    return [(float(index), float(index + 1)) for index in range(count)]


def _cluster_count(labels):
    return len(np.unique(labels))


def _same_cluster(labels, left, right):
    return labels[left] == labels[right]


def test_keeps_clearly_separated_speakers_apart():
    embeddings = _directions(-2, 0, 1, 2, 88, 90, 91, 92)
    initial = np.array([10] * 4 + [20] * 4)

    merged, trace = merge_clusters_confidence_aware(
        embeddings, initial, _timestamps(len(initial)), return_diagnostics=True)

    assert np.array_equal(merged, initial)
    assert trace["accepted_merges"] == 0
    assert trace["final_cluster_count"] == 2
    assert trace["merge_decisions"][0]["accepted"] is False


def test_merges_compact_fragments_of_one_speaker():
    embeddings = _directions(-3, -2, -1, 0, 4, 5, 6, 7)
    initial = np.array([3] * 4 + [8] * 4)

    merged = merge_clusters_confidence_aware(embeddings, initial)

    assert _cluster_count(merged) == 1


def test_rejects_matching_centroids_with_different_distributions():
    # Both centroids point at zero degrees, while half of the cross pairs disagree.
    embeddings = _directions(-35, 35, -35, 35, -35, 35,
                             -65, 65, -65, 65, -65, 65)
    initial = np.array([0] * 6 + [1] * 6)

    merged, trace = merge_clusters_confidence_aware(
        embeddings, initial, return_diagnostics=True)

    decision = trace["merge_decisions"][0]
    assert decision["centroid_similarity"] > .99
    assert decision["cross_similarity"]["median"] < .78
    assert decision["accepted"] is False
    assert np.array_equal(merged, initial)


def test_does_not_absorb_a_small_distinct_speaker():
    embeddings = np.vstack((_directions(-2, -1, 0, 1, 2, -2, -1, 0, 1, 2),
                            _directions(78, 82)))
    initial = np.array([0] * 10 + [1] * 2)

    merged = merge_clusters_confidence_aware(embeddings, initial)

    assert np.array_equal(merged, initial)


def test_one_cross_cluster_outlier_cannot_trigger_a_merge():
    embeddings = np.vstack((_directions(-2, -1, 0, 1, 2, 0),
                            _directions(78, 82, 0)))
    initial = np.array([0] * 6 + [1] * 3)

    merged, trace = merge_clusters_confidence_aware(
        embeddings, initial, return_diagnostics=True)

    assert np.array_equal(merged, initial)
    assert trace["merge_decisions"][0]["cross_similarity"]["median"] < .78


def test_short_clusters_abstain_when_support_is_insufficient():
    embeddings = _directions(-1, 0, 2, 3)
    initial = np.array([0, 0, 1, 1])

    merged, trace = merge_clusters_confidence_aware(
        embeddings, initial, config={"min_cluster_size": 3},
        return_diagnostics=True)

    assert np.array_equal(merged, initial)
    assert trace["accepted_merges"] == 0
    assert "sufficient_support" in trace["merge_decisions"][0]["rejection_reasons"]


@pytest.mark.parametrize("degrees,labels", [
    ([0], [7]),
    ([-2, 0, 2, 1], [7, 7, 7, 7]),
])
def test_singleton_and_existing_one_speaker_partitions_are_valid(degrees, labels):
    initial = np.asarray(labels)
    merged, trace = merge_clusters_confidence_aware(
        _directions(*degrees), initial, return_diagnostics=True)

    assert np.array_equal(merged, initial)
    assert trace["initial_cluster_count"] == trace["final_cluster_count"] == 1
    assert trace["merge_attempts"] == 0


def test_ambiguous_three_way_fragments_abstain():
    embeddings = _directions(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)
    initial = np.repeat([10, 20, 30], 4)

    merged, trace = merge_clusters_confidence_aware(
        embeddings, initial, return_diagnostics=True)

    assert _cluster_count(merged) == 3
    assert trace["accepted_merges"] == 0


def test_chain_merge_does_not_bridge_inconsistent_endpoints():
    embeddings = _directions(0, 0, 0, 0, 12, 12, 12, 12, 24, 24, 24, 24)
    initial = np.repeat([0, 1, 2], 4)
    config = {
        "centroid_similarity_threshold": .96,
        "cross_median_threshold": .96,
        "cross_lower_threshold": .95,
        "min_similarity_margin": 0,
    }

    merged = merge_clusters_confidence_aware(
        embeddings, initial, config=config)

    assert _cluster_count(merged) == 2
    assert not _same_cluster(merged, 0, 8)


def test_diagnostics_are_complete_deterministic_and_inputs_are_unchanged():
    embeddings = _directions(-3, -2, -1, 3, 4, 5, 80, 81, 82)
    labels = np.repeat([9, 4, 7], 3)
    timestamps = _timestamps(len(labels))
    original = embeddings.copy(), labels.copy(), copy.deepcopy(timestamps)

    first_labels, first = merge_clusters_confidence_aware(
        embeddings, labels, timestamps, return_diagnostics=True)
    second_labels, second = merge_clusters_confidence_aware(
        embeddings, labels, timestamps, return_diagnostics=True)

    assert np.array_equal(first_labels, second_labels)
    assert first == second
    assert np.array_equal(embeddings, original[0])
    assert np.array_equal(labels, original[1])
    assert timestamps == original[2]
    assert set(first) >= {
        "strategy", "parameters", "initial_cluster_count", "final_cluster_count",
        "initial_clusters", "final_clusters", "merge_attempts", "accepted_merges",
        "rejected_merges", "merge_decisions",
    }
    required_decision_fields = {
        "merge_step", "cluster_a", "cluster_b", "cluster_a_size", "cluster_b_size",
        "cluster_a_duration", "cluster_b_duration", "centroid_similarity",
        "cross_similarity", "compactness_a", "compactness_b", "merged_compactness",
        "compactness_drop", "similarity_margin", "mutual_nearest",
        "decision_evidence", "accepted",
        "rejection_reason", "rejection_reasons", "cluster_count_before",
        "cluster_count_after",
    }
    assert first["merge_decisions"]
    assert set(first["merge_decisions"][0]) >= required_decision_fields
    assert first["merge_attempts"] == len(first["merge_decisions"])


@pytest.mark.parametrize("embeddings,error", [
    ([[np.nan, 0.]], "finite"),
    ([[np.inf, 1.]], "finite"),
    ([[0., 0.]], "nonzero"),
])
def test_rejects_non_finite_and_zero_norm_embeddings(embeddings, error):
    with pytest.raises(ValueError, match=error):
        merge_clusters_confidence_aware(np.asarray(embeddings), np.array([0]))


def test_empty_input_is_a_valid_no_op():
    merged, trace = merge_clusters_confidence_aware(
        np.empty((0, 2)), np.array([], dtype=int), return_diagnostics=True)

    assert merged.dtype.kind == "i" and merged.size == 0
    assert trace["initial_cluster_count"] == trace["final_cluster_count"] == 0
    assert trace["merge_attempts"] == 0


@pytest.mark.parametrize("labels,timestamps,error", [
    ([], None, "equal length"),
    ([0], [(0., 1.), (1., 2.)], "timestamp"),
    ([0], [(1., 0.)], "timestamp"),
])
def test_rejects_mismatched_or_invalid_metadata(labels, timestamps, error):
    with pytest.raises(ValueError, match=error):
        merge_clusters_confidence_aware(
            _directions(0), np.asarray(labels), timestamps=timestamps)


def test_bounded_completion_with_many_initial_clusters():
    embeddings = _directions(*[angle for base in (0, 40, 80, 120)
                               for angle in (base, base + 1, base, base + 1)])
    labels = np.repeat(np.arange(8), 2)

    merged, trace = merge_clusters_confidence_aware(
        embeddings, labels, return_diagnostics=True)

    initial_count = len(np.unique(labels))
    maximum_pair_evaluations = initial_count * (initial_count + 1) * (initial_count - 1) // 6
    assert trace["merge_attempts"] <= maximum_pair_evaluations
    assert trace["accepted_merges"] <= initial_count - 1
    assert trace["final_cluster_count"] == _cluster_count(merged)


@pytest.mark.parametrize("config", [
    {"unknown_option": 1},
    {"subsample_size": 1},
    {"min_cluster_size": 1},
    {"cross_median_threshold": float('nan')},
    {"max_compactness_drop": True},
    {"min_similarity_margin": "0.1"},
])
def test_rejects_unknown_or_invalid_configuration(config):
    with pytest.raises(ValueError):
        merge_clusters_confidence_aware(
            _directions(0, 1), np.array([0, 1]), config=config)


def test_centroid_baseline_accepts_distribution_mismatch_rejected_by_new_method():
    embeddings = _directions(*([-35, 35] * 5 + [-65, 65]))
    labels = np.array([0] * 10 + [1] * 2)
    assert _cluster_count(merge_similar_clusters(embeddings, labels)) == 1
    assert _cluster_count(merge_clusters_confidence_aware(embeddings, labels)) == 2


def test_mixed_bridge_and_singleton_do_not_join_distinct_speakers():
    embeddings = _directions(-2, 0, 2, 78, 80, 82, 40)
    labels = np.array([0, 0, 0, 1, 1, 1, 2])
    assert np.array_equal(merge_clusters_confidence_aware(embeddings, labels), labels)


def test_identical_embeddings_can_merge_but_cancelled_centroids_abstain():
    assert _cluster_count(merge_clusters_confidence_aware(
        np.ones((4, 2)), np.array([0, 0, 1, 1]))) == 1
    embeddings = np.array([[1., 0], [-1., 0], [0., 1], [0., -1]])
    labels = np.array([0, 0, 1, 1])
    result, trace = merge_clusters_confidence_aware(embeddings, labels, return_diagnostics=True)
    assert np.array_equal(result, labels)
    assert trace['merge_decisions'][0]['centroid_similarity'] is None


def test_merges_recompute_evidence_and_can_reach_one_speaker():
    embeddings = _directions(0, 0, 0, 3, 3, 3, 20, 20, 20)
    labels, trace = merge_clusters_confidence_aware(
        embeddings, np.repeat([0, 1, 2], 3), return_diagnostics=True)
    assert _cluster_count(labels) == 1 and trace['accepted_merges'] == 2
    accepted = [row for row in trace['merge_decisions'] if row['accepted']]
    assert accepted[1]['cluster_a_size'] == 6


def test_duration_unions_overlapping_windows_and_labels_require_integers():
    _, trace = merge_clusters_confidence_aware(
        np.ones((2, 2)), np.array([0, 0]), [(0, 2), (1, 3)], return_diagnostics=True)
    assert trace['initial_clusters'][0]['duration'] == 3
    with pytest.raises(ValueError, match='integers'):
        merge_clusters_confidence_aware(np.ones((2, 2)), np.array([.5, 1.5]))


def test_correct_count_does_not_imply_correct_partition():
    from sklearn.metrics import adjusted_rand_score
    expected = np.repeat([0, 1], 4)
    incorrect = np.tile([0, 1], 4)
    assert _cluster_count(incorrect) == _cluster_count(expected)
    assert adjusted_rand_score(expected, incorrect) < 0
