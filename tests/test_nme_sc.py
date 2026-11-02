import numpy as np
import pytest

from diarize.cluster import cluster_embeddings


def _embeddings(n_speakers, samples=20):
    centers = np.eye(max(3, n_speakers))[:n_speakers]
    return np.vstack([np.repeat(center[None, :], samples, axis=0) for center in centers])


def _assert_pure(labels, n_speakers, samples=20):
    assert len(set(labels)) == n_speakers
    for index in range(n_speakers):
        assert len(set(labels[index * samples:(index + 1) * samples])) == 1


@pytest.mark.parametrize('n_speakers', [1, 2, 3])
def test_nme_sc_estimates_well_separated_speakers(n_speakers):
    labels = cluster_embeddings(_embeddings(n_speakers), method='nme_sc',
                                max_speakers=5, seed=7)
    _assert_pure(labels, n_speakers)


def test_nme_sc_handles_short_and_degenerate_inputs():
    assert cluster_embeddings(np.array([[1., 0.]]), method='nme_sc').tolist() == [0]
    assert len(set(cluster_embeddings(np.array([[1., 0.], [0., 1.]]), method='nme_sc'))) == 2
    assert cluster_embeddings(np.ones((8, 3)), method='nme_sc').tolist() == [0] * 8


def test_nme_sc_known_count_and_determinism():
    embeddings = _embeddings(3)
    first = cluster_embeddings(embeddings, method='nme_sc', n_speakers=3, seed=17)
    second = cluster_embeddings(embeddings, method='nme_sc', n_speakers=3, seed=17)
    assert np.array_equal(first, second)
    _assert_pure(first, 3)
