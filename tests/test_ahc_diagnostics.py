import numpy as np

from diarize.cluster import cluster_ahc, cluster_embeddings


def _same_partition(left, right):
    return all((left[i] == left[j]) == (right[i] == right[j])
               for i in range(len(left)) for j in range(len(left)))


def test_ahc_diagnostics_reconstruct_full_hierarchy_without_changing_partition():
    embeddings = np.array([
        [1.0, 0.0], [.99, .01], [.98, .02],
        [0.0, 1.0], [.01, .99], [.02, .98],
    ])
    baseline = cluster_ahc(embeddings, threshold=.25)
    labels, trace = cluster_ahc(embeddings, threshold=.25, return_diagnostics=True)

    assert _same_partition(baseline, labels)
    assert len(trace['children']) == len(embeddings) - 1
    assert len(trace['distances']) == len(trace['merges']) == len(embeddings) - 1
    assert trace['merges'][-1]['merged_cluster_size'] == len(embeddings)
    assert all(current <= following
               for current, following in zip(trace['distances'], trace['distances'][1:]))
    for step, merge in enumerate(trace['merges']):
        assert merge['merge_step'] == step
        assert merge['merge_distance'] == trace['distances'][step]
        assert [merge['left_cluster_id'], merge['right_cluster_id']] == trace['children'][step]


def test_fixed_speaker_count_diagnostics_preserve_partition():
    embeddings = np.array([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
    baseline = cluster_ahc(embeddings, n_speakers=2)
    labels, trace = cluster_ahc(embeddings, n_speakers=2, return_diagnostics=True)
    assert _same_partition(baseline, labels)
    assert trace['requested_n_speakers'] == 2
    assert trace['threshold'] is None


def test_cluster_embeddings_exposes_actual_ahc_trace():
    embeddings = np.array([[1., 0.], [.99, .01], [0., 1.], [.01, .99]])
    diagnostics = {}
    labels = cluster_embeddings(embeddings, method='ahc', threshold=.2,
                                diagnostics=diagnostics)
    assert diagnostics['ahc']['labels'] == labels.tolist()
    assert diagnostics['ahc']['metric'] == 'cosine'
    assert diagnostics['ahc']['linkage'] == 'average'
