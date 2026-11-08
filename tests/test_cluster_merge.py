import numpy as np
import pytest

from diarize.cluster import merge_similar_clusters
from diarize import pipeline


def test_merges_small_fragment_of_same_speaker():
    embeddings = np.array([[1., 0.]] * 6 + [[.99, .05]] * 2)
    labels, trace = merge_similar_clusters(embeddings, np.array([0] * 6 + [1] * 2),
                                            .95, return_diagnostics=True)
    assert labels.tolist() == [0] * 8
    assert trace['merge_attempts'] == trace['accepted_merges'] == 1
    assert trace['merge_candidates'][0]['accepted'] is True
    assert len(trace['merge_candidates'][0]['target_similarities']) == 1


def test_different_and_small_legitimate_speakers_remain_separate():
    embeddings = np.array([[1., 0.]] * 6 + [[0., 1.]] * 2)
    labels, trace = merge_similar_clusters(embeddings, np.array([0] * 6 + [1] * 2),
                                            .80, return_diagnostics=True)
    assert labels.tolist() == [0] * 6 + [1] * 2
    assert trace['rejected_merges'] == 1
    assert trace['merge_rejection_reasons'] == {'below_similarity_threshold': 1}
    assert trace['merge_candidates'][0]['best_similarity'] < .99


def test_selects_clear_best_of_multiple_targets():
    embeddings = np.array([[1., 0.]] * 5 + [[.7, .7]] * 5 + [[.99, .01]] * 2)
    labels = merge_similar_clusters(embeddings, np.array([0] * 5 + [1] * 5 + [2] * 2), .75)
    assert labels.tolist() == [0] * 5 + [1] * 5 + [0] * 2


def test_ambiguous_target_is_rejected_by_margin():
    embeddings = np.array([[1., .1]] * 5 + [[1., -.1]] * 5 + [[1., 0.]] * 2)
    labels = merge_similar_clusters(embeddings, np.array([0] * 5 + [1] * 5 + [2] * 2),
                                    .75, min_similarity_margin=.05)
    assert len(set(labels)) == 3


def test_below_and_exact_threshold():
    embeddings = np.array([[1., 0.]] * 5 + [[.8, .6]])
    initial = np.array([0] * 5 + [1])
    assert len(set(merge_similar_clusters(embeddings, initial, .8001, min_similarity_margin=0))) == 2
    assert len(set(merge_similar_clusters(embeddings, initial, .8, min_similarity_margin=0))) == 1


def test_empty_and_single_cluster_inputs():
    assert merge_similar_clusters(np.empty((0, 2)), np.array([])).dtype.kind == 'i'
    assert merge_similar_clusters(np.ones((2, 2)), np.array([7, 7])).tolist() == [7, 7]


@pytest.mark.parametrize('embeddings,labels,error', [
    ([[0., 0.]], [0], 'finite nonzero'),
    ([[np.nan, 0.]], [0], 'finite nonzero'),
    ([[1., 0.]], [], 'equal length'),
])
def test_invalid_embeddings(embeddings, labels, error):
    with pytest.raises(ValueError, match=error):
        merge_similar_clusters(np.asarray(embeddings), np.asarray(labels))


@pytest.mark.parametrize('kwargs', [
    {'similarity_threshold': 1.1},
    {'max_small_cluster_size': 0},
    {'min_similarity_margin': -1},
])
def test_invalid_merge_options(kwargs):
    with pytest.raises(ValueError):
        merge_similar_clusters(np.ones((1, 2)), np.array([0]), **kwargs)


def test_repeated_execution_is_deterministic_and_does_not_mutate_inputs():
    embeddings = np.array([[1., 0.]] * 5 + [[.99, .01]] * 2)
    labels = np.array([5] * 5 + [9] * 2)
    original = labels.copy()
    first = merge_similar_clusters(embeddings, labels, .9)
    second = merge_similar_clusters(embeddings, labels, .9)
    assert np.array_equal(first, second)
    assert np.array_equal(labels, original)


def test_rejected_merge_preserves_labels_exactly():
    embeddings = np.array([[1., 0.]] * 5 + [[0., 1.]] * 2)
    labels = np.array([9] * 5 + [3] * 2)
    result, trace = merge_similar_clusters(embeddings, labels, .8, return_diagnostics=True)
    assert np.array_equal(result, labels)
    assert trace['accepted_merges'] == 0


def _mock_pipeline(monkeypatch, embeddings):
    monkeypatch.setattr(pipeline, 'load_audio', lambda _: (np.zeros(16000 * len(embeddings)), 16000))
    monkeypatch.setattr(pipeline, 'get_speech_segments', lambda *a, **k: [(0., float(len(embeddings)))])
    segments = [(float(i), float(i + 1)) for i in range(len(embeddings))]
    monkeypatch.setattr(pipeline, 'extract_embeddings_for_segments',
                        lambda *a: (segments, np.asarray(embeddings)))


def test_pipeline_merge_is_opt_in_and_fixes_overclustering(monkeypatch):
    embeddings = [[1., 0.]] * 6 + [[.99, .05]] * 2
    _mock_pipeline(monkeypatch, embeddings)
    initial = np.array([0] * 6 + [1] * 2)
    monkeypatch.setattr(pipeline, 'cluster_embeddings', lambda *a, **k: initial.copy())
    baseline = pipeline.diarize('sample.wav', 'unused', encoder=object())
    assert baseline == pipeline.diarize('sample.wav', 'unused', encoder=object(), centroid_merge=False)
    trace = {}
    merged = pipeline.diarize('sample.wav', 'unused', encoder=object(), centroid_merge=True,
                              centroid_similarity_threshold=.95, diagnostics=trace)
    assert 'SPEAKER_01' in baseline and 'SPEAKER_01' not in merged
    assert trace['n_clusters_before_postmerge'] == 2
    assert trace['n_clusters_after_postmerge'] == 1


def test_known_speaker_count_skips_merge(monkeypatch):
    _mock_pipeline(monkeypatch, [[1., 0.]] * 4 + [[.99, .05]] * 4)
    trace = {}
    pipeline.diarize('sample.wav', 'unused', encoder=object(), n_speakers=2,
                     centroid_merge=True, diagnostics=trace)
    assert trace['n_clusters_before_postmerge'] == trace['n_clusters_after_postmerge'] == 2
    assert trace['centroid_merge_skipped_known_count'] is True
    assert trace['accepted_merges'] == 0


def test_confidence_pipeline_trace_and_opt_in_preserve_baseline(monkeypatch, tmp_path):
    import json
    embeddings = np.array([[1., 0.]] * 4 + [[.99, .05]] * 4)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    _mock_pipeline(monkeypatch, embeddings)
    initial = np.repeat([0, 1], 4)
    monkeypatch.setattr(pipeline, 'cluster_embeddings', lambda *a, **k: initial.copy())
    baseline = pipeline.diarize('sample.wav', 'unused', encoder=object())
    assert baseline == pipeline.diarize('sample.wav', 'unused', encoder=object(),
                                       confidence_aware_merge=False)
    diagnostics = {}
    result = pipeline.diarize('sample.wav', 'unused', encoder=object(),
        confidence_aware_merge=True, diagnostics=diagnostics,
        diagnostic_output_dir=str(tmp_path), diagnostic_counterfactuals=True)
    assert 'SPEAKER_01' in baseline and 'SPEAKER_01' not in result
    assert diagnostics['confidence_aware_merge']['accepted_merges'] == 1
    assert diagnostics['n_clusters_after_postmerge'] == 1
    trace = json.loads((tmp_path / 'sample/pipeline_trace.json').read_text())
    assert trace['stages']['initial_ahc']['labels'] == initial.tolist()
    assert set(trace['stages']['after_confidence_aware_merge']['labels']) == {0}
    assert (tmp_path / 'sample/confidence_merge_decisions.json').exists()


@pytest.mark.parametrize('kwargs', [
    {'n_speakers': 2}, {'method': 'spectral'}, {'centroid_merge': True},
    {'count_estimator': 'adaptive_ahc'}, {'count_estimator': 'adaptive_ahc_robust'},
])
def test_confidence_pipeline_rejects_incompatible_modes_before_loading_audio(kwargs):
    with pytest.raises(ValueError):
        pipeline.diarize('missing.wav', 'unused', confidence_aware_merge=True, **kwargs)
