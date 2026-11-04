import numpy as np
import pytest

import diarize.cluster as clustering
from diarize.cluster import estimate_ahc_speaker_count, cluster_embeddings


def _same_partition(left, right):
    return all((left[i] == left[j]) == (right[i] == right[j])
               for i in range(len(left)) for j in range(len(left)))


def test_estimator_finds_clear_three_cluster_hierarchy_with_diagnostics():
    count, trace = estimate_ahc_speaker_count(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.8, 0.9], n_samples=8, max_speakers=5)

    assert count == 3
    assert trace['estimator'] == 'adaptive_ahc'
    assert trace['estimated_n_speakers'] == 3
    assert trace['selected_gap'] == pytest.approx(0.75)
    assert trace['distances'] == [0.01, 0.02, 0.03, 0.04, 0.05, 0.8, 0.9]


@pytest.mark.parametrize(
    ('distances', 'n_samples'),
    [([], 1), ([0.2], 2), ([0.1, 0.11, 0.12, 0.13], 5), ([0, 0, 0], 4)],
)
def test_estimator_returns_one_when_hierarchy_has_insufficient_separation(distances, n_samples):
    count, trace = estimate_ahc_speaker_count(distances, n_samples, max_speakers=10)
    assert count == 1
    assert trace['selected_gap'] is None
    assert trace['selection'] == 'fixed_threshold_fallback'


def test_short_hierarchy_uses_explicit_fixed_threshold_fallback():
    close_count, close_trace = estimate_ahc_speaker_count([0.2], 2, fallback_threshold=.7)
    far_count, far_trace = estimate_ahc_speaker_count([0.9], 2, fallback_threshold=.7)

    assert close_count == 1
    assert far_count == 2
    assert far_trace['fallback_n_speakers'] == 2
    assert far_trace['fallback_threshold'] == .7
    assert close_trace['selection'] == far_trace['selection'] == 'fixed_threshold_fallback'
    assert close_trace['selection_reason'] == 'insufficient_candidate_gaps'


def test_fallback_respects_count_bound_and_fixed_threshold_semantics():
    count, trace = estimate_ahc_speaker_count(
        [0.1, 0.2, 0.8, 0.9], 5, max_speakers=2, fallback_threshold=.7)

    assert count == trace['fallback_n_speakers'] == 2


@pytest.mark.parametrize('estimator', ['adaptive_ahc', 'adaptive_ahc_robust'])
def test_adaptive_ahc_is_deterministic_bounded_and_handles_identical_embeddings(estimator):
    embeddings = np.vstack([
        np.tile([1.0, 0.0, 0.0], (3, 1)),
        np.tile([0.0, 1.0, 0.0], (3, 1)),
        np.tile([0.0, 0.0, 1.0], (3, 1)),
    ])
    diagnostics = {}
    first = cluster_embeddings(embeddings, count_estimator=estimator,
                               max_speakers=2, diagnostics=diagnostics)
    second = cluster_embeddings(embeddings, count_estimator=estimator, max_speakers=2)

    assert np.array_equal(first, second)
    assert len(set(first)) == 2
    assert diagnostics['ahc']['estimated_n_speakers'] == 2
    assert diagnostics['ahc']['estimator'] == estimator
    assert diagnostics['ahc']['gaps']
    assert cluster_embeddings(np.ones((5, 3)), count_estimator=estimator).tolist() == [0] * 5


def test_adaptive_ahc_cuts_the_fitted_hierarchy_without_refitting(monkeypatch):
    embeddings = np.vstack([
        np.tile([1.0, 0.0, 0.0], (3, 1)),
        np.tile([0.0, 1.0, 0.0], (3, 1)),
        np.tile([0.0, 0.0, 1.0], (3, 1)),
    ])
    calls = 0
    original = clustering.cluster_ahc

    def counted_cluster_ahc(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(clustering, 'cluster_ahc', counted_cluster_ahc)
    diagnostics = {}
    labels = cluster_embeddings(
        embeddings, count_estimator='adaptive_ahc', diagnostics=diagnostics)

    assert calls == 1
    assert len(set(labels)) == 3
    assert diagnostics['ahc']['labels'] == labels.tolist()
    assert diagnostics['ahc']['requested_n_speakers'] == 3


@pytest.mark.parametrize('n_clusters', [1, 2, 4, 7])
def test_fitted_hierarchy_cut_matches_sklearn_partition_for_all_boundaries(n_clusters):
    embeddings = np.random.default_rng(7).normal(size=(7, 5))
    _, hierarchy = clustering.cluster_ahc(
        embeddings, n_speakers=1, return_diagnostics=True)

    labels = clustering._cut_ahc_hierarchy(
        hierarchy['children'], len(embeddings), n_clusters)
    expected = clustering.cluster_ahc(embeddings, n_speakers=n_clusters)

    assert _same_partition(labels, expected)
    assert np.array_equal(
        labels,
        clustering._cut_ahc_hierarchy(
            hierarchy['children'], len(embeddings), n_clusters),
    )


def test_estimator_reports_gap_uncertainty_and_rejects_tied_evidence():
    count, trace = estimate_ahc_speaker_count(
        [0.1, 0.2, 0.3, 0.4], n_samples=5, max_speakers=4,
        fallback_threshold=.7, robust=True)

    assert count == 1
    assert trace['selected_gap'] is None
    assert trace['selection_reason'] == 'no_unique_outlier'
    assert trace['eligible_gap_count'] == 3
    assert trace['runner_up_gap'] == pytest.approx(trace['best_gap'])


def test_estimator_uses_unmasked_tukey_baseline_for_a_clear_gap():
    count, trace = estimate_ahc_speaker_count(
        [0.1, 0.11, 0.12, 0.32, 0.4], n_samples=6, max_speakers=5,
        fallback_threshold=.7, robust=True)

    assert count == 3
    assert trace['selection'] == 'outlying_gap'
    assert trace['selected_gap'] == pytest.approx(.2)
    assert trace['outlier_cutoff'] < trace['selected_gap']
    assert trace['gap_margin'] == pytest.approx(.12)


def test_robust_estimator_is_separate_from_legacy_adaptive_behavior():
    distances = [0.1, 0.11, 0.12, 0.32, 0.4]

    legacy_count, legacy = estimate_ahc_speaker_count(distances, n_samples=6)
    robust_count, robust = estimate_ahc_speaker_count(
        distances, n_samples=6, robust=True)

    assert legacy_count == 1
    assert robust_count == 3
    assert legacy['estimator'] == 'adaptive_ahc'
    assert robust['estimator'] == 'adaptive_ahc_robust'


def test_adaptive_estimator_rejects_oracle_and_non_ahc_combinations():
    embeddings = np.eye(3)
    with pytest.raises(ValueError, match='automatic AHC'):
        cluster_embeddings(embeddings, n_speakers=2, count_estimator='adaptive_ahc')
    with pytest.raises(ValueError, match='automatic AHC'):
        cluster_embeddings(embeddings, method='spectral', count_estimator='adaptive_ahc')


def test_fixed_ahc_diagnostics_name_the_unchanged_estimator():
    diagnostics = {}
    labels = cluster_embeddings(np.array([[1., 0.], [0., 1.]]), threshold=.5,
                                diagnostics=diagnostics)

    assert diagnostics['ahc']['estimator'] == 'fixed_threshold'
    assert diagnostics['ahc']['estimated_n_speakers'] == len(set(labels))
