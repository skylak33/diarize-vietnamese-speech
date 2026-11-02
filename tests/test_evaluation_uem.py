import json
import sys

import numpy as np
import pytest
import soundfile as sf

from eval.der import evaluate_annotations, load_uem, parse_uem, prepare_evaluation
from experiments import run_diarization_experiments as runner


def test_parse_uem_merges_overlapping_and_adjacent_intervals(tmp_path):
    path = tmp_path / 'all.uem'
    path.write_text('# comment\na 1 3 4\na 1 0 1\na 1 1 2\na 1 3.5 5\n')
    assert parse_uem(path) == {'a': [(0.0, 2.0), (3.0, 5.0)]}


@pytest.mark.parametrize('row', [
    'a 1 -1 2', 'a 1 2 2', 'a 1 3 2', 'a 1 nan 2', 'a 1 x 2', 'a 1 0',
])
def test_parse_uem_rejects_invalid_rows(tmp_path, row):
    path = tmp_path / 'bad.uem'
    path.write_text(row + '\n')
    with pytest.raises(ValueError):
        parse_uem(path)


def test_load_uem_requires_data_and_unique_recordings(tmp_path):
    with pytest.raises(ValueError, match='No UEM files'):
        load_uem(tmp_path)
    (tmp_path / 'one.uem').write_text('a 1 0 1\n')
    (tmp_path / 'two.uem').write_text('a 1 1 2\n')
    with pytest.raises(ValueError, match='Duplicate recording IDs'):
        load_uem(tmp_path)


def test_no_uem_preserves_full_audio_scoring():
    ref = {'a': [(2, 4, 'A')]}
    hyp = {'a': [(0, 1, 'X'), (2, 4, 'Y')]}
    implicit = evaluate_annotations(ref, hyp, collar=0, durations={'a': 5})
    explicit_none = evaluate_annotations(ref, hyp, collar=0, durations={'a': 5}, uem_data=None)
    assert implicit == explicit_none
    assert implicit['a']['false_alarm_seconds'] == pytest.approx(1)


def test_external_uem_limits_scoring_and_supports_nonzero_start():
    ref = {'a': [(2, 4, 'A')]}
    hyp = {'a': [(0, 1, 'X'), (2, 4, 'Y')]}
    result = evaluate_annotations(ref, hyp, collar=0, durations={'a': 5},
                                  uem_data={'a': [(2, 3), (3, 4)]})['a']
    assert result['duration'] == pytest.approx(2)
    assert result['DER'] == pytest.approx(0)
    assert result['N_ref'] == result['N_pred'] == 1


def test_uem_ids_and_recording_bounds_are_validated():
    ref = {'a': [(0, 1, 'A')]}
    with pytest.raises(ValueError, match='missing UEM'):
        evaluate_annotations(ref, {}, uem_data={})
    with pytest.raises(ValueError, match='UEM exceeds recording duration'):
        evaluate_annotations(ref, {}, durations={'a': 2}, uem_data={'a': [(0, 3)]})


def test_collar_and_overlap_match_pyannote_preparation():
    ref = {'a': [(0, 1, 'A'), (1, 2, 'B')]}
    hyp = {'a': [(0, 1.1, 'X'), (1.1, 2, 'Y')]}
    assert evaluate_annotations(ref, hyp, collar=.25,
                                uem_data={'a': [(0, 2)]})['a']['DER'] == pytest.approx(0)
    assert evaluate_annotations(ref, hyp, collar=0,
                                uem_data={'a': [(0, 2)]})['a']['speaker_error_seconds'] > 0

    overlap_ref = {'a': [(0, 2, 'A'), (1, 3, 'B')]}
    overlap_hyp = {'a': [(0, 1, 'X'), (2, 3, 'Y')]}
    included = evaluate_annotations(overlap_ref, overlap_hyp, 0, False,
                                    uem_data={'a': [(0, 3)]})['a']
    skipped = evaluate_annotations(overlap_ref, overlap_hyp, 0, True,
                                   uem_data={'a': [(0, 3)]})['a']
    assert included['miss_seconds'] > 0
    assert skipped['DER'] == pytest.approx(0)


def test_prepare_evaluation_aligns_permuted_speaker_labels():
    reference, hypothesis, effective_uem, mapping = prepare_evaluation(
        'a', [(0, 1, 'A'), (1, 2, 'B')], [(0, 1, 'Y'), (1, 2, 'X')],
        collar=0, uem_intervals=[(0, 2)], duration=2)
    assert mapping == {'X': 'B', 'Y': 'A'}
    assert sum(segment.duration for segment in effective_uem) == pytest.approx(2)
    assert set(reference.labels()) == {'A', 'B'}
    assert set(hypothesis.labels()) == {'X', 'Y'}


def test_experiment_runner_records_external_uem(tmp_path, monkeypatch):
    audio, refs, uem = (tmp_path / name for name in ('audio', 'refs', 'uem'))
    for folder in (audio, refs, uem):
        folder.mkdir()
    sf.write(audio / 'a.wav', np.zeros(16000), 16000)
    truth = 'SPEAKER a 1 0 1 <NA> <NA> A <NA> <NA>\n'
    (refs / 'a.rttm').write_text(truth)
    (uem / 'a.uem').write_text('a 1 0 1\n')
    model = tmp_path / 'model.model'
    model.write_bytes(b'synthetic checkpoint')

    def fake_diarize(_audio, _model, **kwargs):
        path = kwargs['output_rttm']
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(truth)

    monkeypatch.setattr('diarize.pipeline.diarize', fake_diarize)
    monkeypatch.setattr('utils.model.load_model', lambda *args: object())
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', [
        'runner', '--audio_dir', str(audio), '--ref_rttm_dir', str(refs),
        '--uem_dir', str(uem), '--output_dir', str(output), '--model', str(model),
        '--experiment', 'speaker_count', '--collar', '0',
    ])
    runner.main()
    manifest = json.loads((output / 'experiment_manifest.json').read_text())
    assert manifest['evaluation']['mode'] == 'external_uem'
    assert manifest['evaluation']['collar'] == 0
    assert manifest['evaluation']['skip_overlap'] is False
    assert list(manifest['evaluation']['uem_sha256']) == [str(uem / 'a.uem')]
