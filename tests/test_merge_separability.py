import copy

import numpy as np
import pytest

from diarize.cluster import merge_clusters_confidence_aware
from experiments.analyze_merge_separability import (
    label_pair, replay_candidates, initial_population, deduplicate,
    feature_statistics, decision_quality, rejection_summary, operating_points,
    support_bin, dependency_groups, validate_membership_features)
from experiments.analyze_partition_quality import associate_reference_windows


def associations(labels):
    return [dict(status='single_speaker' if label is not None else 'ambiguous',
                 reference_label=label) for label in labels]


@pytest.mark.parametrize('refs,expected', [
    (['a', 'a', 'a', 'a'], 'positive'),
    (['a', 'a', 'b', 'b'], 'negative'),
    (['a', 'b', 'a', 'a'], 'ambiguous'),
    (['a', None, 'b', 'b'], 'ambiguous'),
])
def test_pair_label_requires_fully_reliable_pure_clusters(refs, expected):
    assert label_pair([0, 1], [2, 3], associations(refs))['reference_merge_label'] == expected


def test_overlapping_windows_do_not_imply_mixed_reference_speech():
    windows = [(0., 2.), (1., 3.), (2., 4.)]
    aligned = associate_reference_windows(windows, [(0., 4., 'a')])
    assert label_pair([0, 1], [2], aligned)['reference_merge_label'] == 'positive'
    overlap = associate_reference_windows(windows, [(0., 4., 'a'), (2.5, 4., 'b')])
    assert label_pair([0], [1, 2], overlap)['reference_merge_label'] == 'ambiguous'


def test_reference_speaker_permutation_preserves_labels():
    assert label_pair([0], [1], associations(['a', 'b']))['reference_merge_label'] == (
        label_pair([0], [1], associations(['z', 'q']))['reference_merge_label'])


def trace_fixture():
    angles = np.deg2rad([0, 0, 3, 3, 80, 80, 150, 150])
    embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
    initial = np.repeat(np.arange(4), 2)
    _, trace = merge_clusters_confidence_aware(embeddings, initial, return_diagnostics=True)
    return initial, trace['merge_decisions']


def test_replay_tracks_dynamic_memberships_and_deduplicates_unaffected_pairs():
    initial, decisions = trace_fixture()
    original = initial.copy(), copy.deepcopy(decisions)
    rows = replay_candidates('r', initial, decisions, associations(['a'] * 4 + ['b'] * 2 + ['c'] * 2))
    unique = deduplicate(rows)
    assert len(rows) > len(unique)
    assert any(row['merge_step'] == 1 and len(row['membership_a']) == 4 for row in rows)
    assert any(row['visits'] == 2 for row in unique)
    initial_rows = initial_population(rows, initial)
    assert len(initial_rows) == 6
    assert all(row['merge_step'] == 0 for row in initial_rows)
    assert np.array_equal(initial, original[0]) and decisions == original[1]


def test_incomplete_initial_logs_fail_instead_of_hiding_selection_bias():
    initial, decisions = trace_fixture()
    rows = replay_candidates('r', initial, decisions, associations(['a'] * 8))
    with pytest.raises(ValueError, match='initial pair'):
        initial_population(rows[1:], initial)


def test_replay_rejects_corrupt_membership_sizes():
    initial, decisions = trace_fixture()
    decisions[0]['cluster_a_size'] = 999
    with pytest.raises(ValueError, match='size'):
        replay_candidates('r', initial, decisions, associations(['a'] * 8))


def test_empty_and_one_class_metrics_are_explicitly_undefined():
    assert feature_statistics([], 'centroid_similarity')['ROC_AUC'] is None
    assert feature_statistics([dict(reference_merge_label='positive', centroid_similarity=.7)],
                              'centroid_similarity')['ROC_AUC'] is None
    assert decision_quality([])['precision'] is None
    assert operating_points([], 'centroid_similarity') == []


def test_auc_ap_and_predefined_feature_direction():
    rows = [dict(reference_merge_label=label, centroid_similarity=score,
                 compactness_loss=score) for label, score in
            [('positive', .9), ('negative', .8), ('positive', .7), ('negative', .1)]]
    metric = feature_statistics(rows, 'centroid_similarity')
    assert metric['ROC_AUC'] == pytest.approx(.75)
    assert metric['Average_Precision'] == pytest.approx(5 / 6)
    assert feature_statistics(rows, 'compactness_loss')['ROC_AUC'] == pytest.approx(.25)


def test_missing_and_tied_features_are_not_imputed_or_inverted():
    rows = [dict(reference_merge_label='positive', symmetric_margin=None),
            dict(reference_merge_label='positive', symmetric_margin=.0),
            dict(reference_merge_label='negative', symmetric_margin=.0)]
    stats = feature_statistics(rows, 'symmetric_margin')
    assert stats['missing_positive'] == 1
    assert stats['ROC_AUC'] == .5 and stats['Average_Precision'] == .5


def test_ambiguous_acceptances_are_separate_from_binary_precision():
    rows = [dict(reference_merge_label=label, accepted=accepted)
            for label, accepted in [('positive', True), ('negative', False),
                                    ('positive', False), ('ambiguous', True)]]
    quality = decision_quality(rows)
    assert quality['TP'] == 1 and quality['FN'] == 1 and quality['TN'] == 1
    assert quality['precision'] == 1. and quality['recall'] == .5
    assert quality['accepted_ambiguous'] == 1 and quality['acceptance_rate'] == .5
    assert quality['strict_safe_acceptance_fraction'] == .5


def test_rejection_aggregation_includes_multigate_failures():
    rows = [dict(reference_merge_label='positive', accepted=False,
                 decision_evidence={'centroid_similarity': False, 'sufficient_support': False}),
            dict(reference_merge_label='negative', accepted=False,
                 decision_evidence={'centroid_similarity': True, 'sufficient_support': False})]
    result = {row['condition']: row for row in rejection_summary(rows)}
    assert result['centroid_similarity']['positive_failed'] == 1
    assert result['centroid_similarity']['negative_passed'] == 1
    assert result['sufficient_support']['positive_failed'] == 1


def test_operating_points_keep_ties_and_report_acceptance_counts():
    rows = [dict(reference_merge_label='positive', centroid_similarity=.5),
            dict(reference_merge_label='negative', centroid_similarity=.5)]
    points = operating_points(rows, 'centroid_similarity')
    high = next(row for row in points if row['kind'] == 'recall_at_precision' and row['target'] == .99)
    assert high['accepted'] == 0 and high['recall'] == 0 and high['precision'] is None


def test_support_bins_and_shared_speaker_dependencies():
    assert [support_bin(value) for value in [1, 2, 4, 5, 15, 16]] == [
        'singleton', 'small_2_4', 'small_2_4', 'medium_5_15', 'medium_5_15', 'large_16_plus']
    groups = dependency_groups([dict(recording_id='a', speaker_ids=['x'], source_hashes=['1']),
                                dict(recording_id='b', speaker_ids=['y'], source_hashes=['2']),
                                dict(recording_id='c', speaker_ids=['x', 'y'], source_hashes=['3'])])
    assert groups['a'] == groups['b'] == groups['c']


def test_reference_labels_are_not_an_inference_argument():
    import inspect
    assert 'reference' not in str(inspect.signature(merge_clusters_confidence_aware))


def test_hypothetical_high_precision_points_count_ambiguous_acceptances():
    rows = [dict(reference_merge_label=label, centroid_similarity=score) for label, score in
            [('positive', .9), ('negative', .1), ('ambiguous', .95)]]
    high = next(row for row in operating_points(rows, 'centroid_similarity')
                if row['kind'] == 'recall_at_precision' and row['target'] == .99)
    assert high['accepted'] == 1 and high['precision'] == 1
    assert high['accepted_ambiguous'] == 1 and high['accepted_total'] == 2


def test_only_failed_gate_counts_include_negative_near_misses():
    rows = [dict(reference_merge_label='negative', accepted=False,
                 decision_evidence={'centroid_similarity': False, 'compactness': True})]
    result = {row['condition']: row for row in rejection_summary(rows)}
    assert result['centroid_similarity']['negative_only_failure'] == 1


def test_reconstructed_memberships_are_verified_against_original_feature_calculator():
    embeddings = np.ones((4, 2))
    initial = np.array([0, 0, 1, 1])
    _, trace = merge_clusters_confidence_aware(embeddings, initial, return_diagnostics=True)
    rows = replay_candidates('r', initial, trace['merge_decisions'], associations(['a'] * 4))
    assert validate_membership_features(embeddings, rows, trace['parameters']) == 1
    rows[0]['centroid_similarity'] = .25
    with pytest.raises(ValueError, match='feature mismatch'):
        validate_membership_features(embeddings, rows, trace['parameters'])
