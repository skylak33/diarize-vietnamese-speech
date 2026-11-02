import numpy as np
import pytest

import diarize.cluster as cluster
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
    assert cluster.cluster_nme_sc(np.eye(4), n_speakers=4).tolist() == [0, 1, 2, 3]


def test_nme_sc_known_count_and_determinism():
    embeddings = _embeddings(3)
    first = cluster_embeddings(embeddings, method='nme_sc', n_speakers=3, seed=17)
    second = cluster_embeddings(embeddings, method='nme_sc', n_speakers=3, seed=17)
    assert np.array_equal(first, second)
    _assert_pure(first, 3)


def test_nme_sc_known_count_ignores_larger_degenerate_nullspace(monkeypatch):
    original_eigh = np.linalg.eigh

    def canonical_nullspace(laplacian):
        eigenvalues, eigenvectors = original_eigh(laplacian)
        unseen = set(range(len(laplacian)))
        components = []
        while unseen:
            pending = [min(unseen)]
            unseen.remove(pending[0])
            component = []
            while pending:
                node = pending.pop()
                component.append(node)
                neighbors = np.flatnonzero(laplacian[node] < -1e-12)
                for neighbor in neighbors:
                    neighbor = int(neighbor)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        pending.append(neighbor)
            components.append(component)
        for index, component in enumerate(components):
            eigenvectors[:, index] = 0.0
            eigenvectors[component, index] = 1 / np.sqrt(len(component))
        return eigenvalues, eigenvectors

    monkeypatch.setattr(cluster.numpy.linalg, 'eigh', canonical_nullspace)
    rng = np.random.default_rng(42)
    embeddings = np.vstack([
        rng.normal([1., 0., 0.], 0.01, (8, 3)),
        rng.normal([0., 1., 0.], 0.01, (8, 3)),
    ])

    labels = cluster_embeddings(embeddings, method='nme_sc', n_speakers=2,
                                max_speakers=1, seed=7)

    _assert_pure(labels, 2, samples=8)


def test_nme_sc_known_count_falls_back_when_target_gap_is_zero(monkeypatch):
    monkeypatch.setattr(
        cluster.numpy.linalg,
        'eigh',
        lambda matrix: (np.zeros(len(matrix)), np.eye(len(matrix))),
    )
    expected = np.array([0, 0, 1, 1])
    calls = []

    def fake_ahc(embeddings, n_speakers=None, threshold=0.7):
        calls.append(n_speakers)
        return expected

    monkeypatch.setattr(cluster, 'cluster_ahc', fake_ahc)

    labels = cluster.cluster_nme_sc(np.eye(4), n_speakers=2, max_speakers=1)

    assert calls == [2]
    assert np.array_equal(labels, expected)
