import json

import numpy as np

from experiments.visualize_embedding_diagnostics import main, plot_ahc_dendrogram


def _artifacts(path):
    np.savez(path / 'embeddings.npz', embeddings=np.array([[1., 0.], [.9, .1], [0., 1.]]),
             window_starts=np.array([0., 1., 2.]), window_ends=np.array([1., 2., 3.]))
    (path / 'pipeline_trace.json').write_text(json.dumps({
        'schema_version': 1,
        'recording_id': 'sample',
        'stages': {'initial_ahc': {'labels': [0, 0, 1]}, 'after_smoothing': [0, 0, 1]},
        'partitioned_segments': [[0, 1, 0], [1, 2, 0], [2, 3, 1]],
        'segments_after_short_merge': [[0, 2, 0], [2, 3, 1]],
        'final_rttm_segments': [{'start': 0, 'end': 2, 'speaker': 'SPEAKER_00'},
                                {'start': 2, 'end': 3, 'speaker': 'SPEAKER_01'}],
        'reference': [[0, 1.5, 'A'], [1.5, 3, 'B']],
        'inference_configuration': {'clustering': {'threshold': .7}},
    }), encoding='utf-8')
    (path / 'ahc_merges.json').write_text(json.dumps({'ahc': {
        'children': [[0, 1], [2, 3]], 'distances': [.1, .9]}}), encoding='utf-8')


def test_cli_writes_all_primary_pngs(tmp_path):
    _artifacts(tmp_path)
    output = tmp_path / 'plots'
    main([str(tmp_path), '--output_dir', str(output)])
    for name in ('similarity_heatmap', 'speaker_timeline', 'similarity_distributions', 'ahc_dendrogram'):
        assert (output / f'{name}.png').stat().st_size > 100


def test_cli_loads_reference_rttm_and_writes_all_svgs(tmp_path):
    _artifacts(tmp_path)
    trace = json.loads((tmp_path / 'pipeline_trace.json').read_text(encoding='utf-8'))
    trace.pop('reference')
    (tmp_path / 'pipeline_trace.json').write_text(json.dumps(trace), encoding='utf-8')
    reference = tmp_path / 'reference.rttm'
    reference.write_text('SPEAKER sample 1 0 1.5 <NA> <NA> A <NA> <NA>\n', encoding='utf-8')
    main([str(tmp_path), '--format', 'svg', '--reference_rttm', str(reference)])
    for name in ('similarity_heatmap', 'speaker_timeline', 'similarity_distributions', 'ahc_dendrogram'):
        assert (tmp_path / f'{name}.svg').stat().st_size > 100
    assert 'Reference' in (tmp_path / 'speaker_timeline.svg').read_text(encoding='utf-8')


def test_dendrogram_rejects_incomplete_real_hierarchy(tmp_path):
    try:
        plot_ahc_dendrogram({'children': [[0, 1]], 'distances': [.1]}, 3, tmp_path / 'bad.png')
    except ValueError as error:
        assert 'incomplete' in str(error)
    else:
        raise AssertionError('Expected an invalid hierarchy error')
