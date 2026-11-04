import numpy as np
import pytest

from diarize.cluster import estimate_ahc_speaker_count, cluster_embeddings


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


def test_fallback_respects_count_bound_and_fixed_threshold_semantics():
    count, trace = estimate_ahc_speaker_count(
        [0.1, 0.2, 0.8, 0.9], 5, max_speakers=2, fallback_threshold=.7)

    assert count == trace['fallback_n_speakers'] == 2


def test_adaptive_ahc_is_deterministic_bounded_and_handles_identical_embeddings():
    embeddings = np.vstack([
        np.tile([1.0, 0.0, 0.0], (3, 1)),
        np.tile([0.0, 1.0, 0.0], (3, 1)),
        np.tile([0.0, 0.0, 1.0], (3, 1)),
    ])
    diagnostics = {}
    first = cluster_embeddings(embeddings, count_estimator='adaptive_ahc',
                               max_speakers=2, diagnostics=diagnostics)
    second = cluster_embeddings(embeddings, count_estimator='adaptive_ahc', max_speakers=2)

    assert np.array_equal(first, second)
    assert len(set(first)) == 2
    assert diagnostics['ahc']['estimated_n_speakers'] == 2
    assert diagnostics['ahc']['gaps']
    assert cluster_embeddings(np.ones((5, 3)), count_estimator='adaptive_ahc').tolist() == [0] * 5


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
