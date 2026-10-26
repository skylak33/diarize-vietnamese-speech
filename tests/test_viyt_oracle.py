"""Count intervention on synthetic embeddings, with the real AHC and postprocessing."""
import copy
import json

import numpy as np
import pytest

from diarize import pipeline
from eval.der import evaluate_annotations, parse_rttm
from experiments.viyt_diagnostics import (comparison_rows, diagnostic_rows,
                                          speech_duration, validate_baseline)
from utils.provenance import sha256


def mock_audio_embeddings(monkeypatch, embeddings):
    monkeypatch.setattr(pipeline, 'load_audio', lambda _: (np.zeros(16000 * len(embeddings)), 16000))
    monkeypatch.setattr(pipeline, 'get_speech_segments', lambda *a, **k: [(0., float(len(embeddings)))])
    segments = [(float(i), float(i + 1)) for i in range(len(embeddings))]
    monkeypatch.setattr(pipeline, 'extract_embeddings_for_segments',
                        lambda *a: (segments, np.asarray(embeddings)))


def test_automatic_unchanged_and_fixed_ahc_ignores_threshold(tmp_path, monkeypatch):
    embeddings = [[1., 0.]] * 4 + [[.8, .6]] * 4
    mock_audio_embeddings(monkeypatch, embeddings)
    trace = {}
    automatic = pipeline.diarize('sample.wav', 'unused', encoder=object(), diagnostics=trace)
    assert automatic == pipeline.diarize('sample.wav', 'unused', encoder=object())
    assert automatic == 'SPEAKER sample 1 0.000 8.000 <NA> <NA> SPEAKER_00 <NA> <NA>'
    assert trace['n_clusters_before_postprocess'] == trace['n_speakers_after_postprocess'] == 1
    outputs = []
    for threshold in (.01, .65, 1.5):
        trace = {}
        outputs.append(pipeline.diarize('sample.wav', 'unused', encoder=object(), n_speakers=2,
                                       threshold=threshold, diagnostics=trace,
                                       output_rttm=str(tmp_path / 'oracle.rttm')))
        assert trace['n_clusters_before_postprocess'] == trace['n_speakers_after_postprocess'] == 2
        assert sorted(trace['cluster_sizes']) == [4, 4]
    assert len(set(outputs)) == 1
    assert len({speaker for _, _, speaker in parse_rttm(tmp_path / 'oracle.rttm')['sample']}) == 2


def test_oracle_reports_cluster_lost_in_smoothing(monkeypatch):
    mock_audio_embeddings(monkeypatch, [[1., 0.]] * 4 + [[.8, .6]] + [[1., 0.]] * 4)
    trace = {}
    pipeline.diarize('sample.wav', 'unused', encoder=object(), n_speakers=2, diagnostics=trace)
    assert trace['n_clusters_before_postprocess'] == 2
    assert trace['n_speakers_after_postprocess'] == 1


def test_too_few_windows_or_empty_vad_is_explicit(monkeypatch):
    mock_audio_embeddings(monkeypatch, [[1., 0.]])
    with pytest.raises(ValueError, match='sample.wav.*2 speakers.*1 embedding'):
        pipeline.diarize('sample.wav', 'unused', encoder=object(), n_speakers=2)
    monkeypatch.setattr(pipeline, 'get_speech_segments', lambda *a, **k: [])
    assert pipeline.diarize('sample.wav', 'unused', encoder=object()) == ''
    with pytest.raises(ValueError, match='VAD found no speech'):
        pipeline.diarize('sample.wav', 'unused', encoder=object(), n_speakers=2)


def test_durations_count_overlap_once_and_delta_sign():
    refs = {'sample': [(0., 2., 'A'), (1., 3., 'B')]}
    hyps = {'sample': [(0., 3., 'X')]}
    assert speech_duration(refs['sample']) == 3.
    metrics = evaluate_annotations(refs, hyps, collar=0, durations={'sample': 4.})
    trace = dict(n_clusters_before_postprocess=2, n_speakers_after_postprocess=1)
    rows = diagnostic_rows([dict(file_id='sample', duration=4.)], refs, hyps, metrics, {'sample': trace})
    assert rows[0]['reference_speech_duration'] == rows[0]['predicted_speech_duration'] == 3.
    assert rows[0]['scored_reference_speaker_time'] == 4.
    assert rows[0]['speakers_lost_in_postprocess'] == 1
    total = metrics['__total__']
    better = dict(total, DER=total['DER'] - 10)
    assert comparison_rows(total, better)[0]['delta'] == -10


def test_baseline_validation_rejects_changed_model_data_and_metrics(tmp_path):
    for folder in ('reference_rttm', 'hypothesis_rttm'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'sample.rttm').write_text('SPEAKER sample 1 0 1 <NA> <NA> A <NA> <NA>\n')
    records = [dict(file_id='sample', duration=2., audio_sha256='test-audio',
                    reference_sha256=sha256(tmp_path / 'reference_rttm' / 'sample.rttm'))]
    manifest = dict(status='complete', speaker_mode='automatic', files_evaluated=1,
                    model_sha256='test-model', environment=dict(python='test', packages={}),
                    collar=0., skip_overlap=False, code_sha256={'eval/der.py': 'same', 'diarize/pipeline.py': 'old'})
    refs = {'sample': [(0., 1., 'A')]}
    metrics = evaluate_annotations(refs, refs, collar=0., durations={'sample': 2.})
    for name, data in [('manifest', manifest), ('recordings', records), ('der_results', metrics)]:
        (tmp_path / f'{name}.json').write_text(json.dumps(data))
    current = copy.deepcopy(manifest)
    current['speaker_mode'] = 'oracle_reference_count'
    current['code_sha256']['diarize/pipeline.py'] = 'instrumented'
    assert validate_baseline(tmp_path, current, records)[1]['__total__']['DER'] == 0.
    with pytest.raises(ValueError, match='model_sha256'):
        validate_baseline(tmp_path, dict(current, model_sha256='other-model'), records)
    with pytest.raises(ValueError, match='recordings'):
        validate_baseline(tmp_path, current, [dict(records[0], audio_sha256='changed')])
    metrics['__total__']['DER'] = 42.
    (tmp_path / 'der_results.json').write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match='metric does not match RTTM'):
        validate_baseline(tmp_path, current, records)
