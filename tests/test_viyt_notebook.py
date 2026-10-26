"""Run the notebook's export and scoring cells on tiny in-memory recordings."""
import ast
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.der import load_rttm
from utils.provenance import sha256


NOTEBOOK = Path(__file__).resolve().parents[1] / 'kaggle' / 'ViYT_Diar_DER.ipynb'


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return {cell['id']: ''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code'}


def row(file_id='synthetic'):
    audio = io.BytesIO()
    sf.write(audio, np.zeros((32000, 2), dtype=np.float32), 8000, format='WAV')
    return dict(audio_id=file_id, audio={'bytes': audio.getvalue(), 'path': None},
                speaker=['A', 'B'], timestamps_start=[0., 1.], timestamps_end=[2., 3.])


def test_notebook_syntax_and_no_saved_outputs():
    for code in code_cells().values():
        ast.parse(code)
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert all(cell.get('outputs', []) == [] for cell in notebook['cells'])


def test_export_and_corpus_score(tmp_path):
    audio, ref, hyp = [tmp_path / name for name in ('audio', 'reference', 'hypothesis')]
    for folder in (audio, ref, hyp):
        folder.mkdir()
    namespace = {'sha256': sha256}
    exec(code_cells()['export-helper'], namespace)
    export = namespace['export_record']
    first = export(row('overlap'), audio, ref)
    second_row = row('missed')
    second_row.update(speaker=['A'], timestamps_start=[0.], timestamps_end=[2.])
    second = export(second_row, audio, ref)
    info = sf.info(audio / 'overlap.wav')
    assert (info.samplerate, info.channels, info.duration) == (16000, 1, 4.)
    assert load_rttm(ref)['overlap'] == [(0., 2., 'A'), (1., 3., 'B')]
    # Perfect overlapping prediction for one recording; no detected speech in the other.
    (hyp / 'overlap.rttm').write_text((ref / 'overlap.rttm').read_text())
    (hyp / 'missed.rttm').write_text('')
    namespace.update(records=[first, second], dataset=[first, second],
                     HYPOTHESIS=hyp, REFERENCE=ref, RESULTS=tmp_path,
                     COLLAR=0., SKIP_OVERLAP=False, MODEL=Path('model_0034.model'),
                     DATASET_ID='synthetic-test-only', json=json, manifest={})
    exec(code_cells()['configuration'], config := {})
    namespace['PIPELINE'] = config['PIPELINE']
    namespace['MODE_NAME'] = config['MODE_NAME']
    namespace['SPEAKER_MODE'] = config['SPEAKER_MODE']
    exec(code_cells()['score'], namespace)
    total = namespace['results']['__total__']
    assert total['duration'] == pytest.approx(6.)  # speaker-time, with overlap
    assert total['DER'] == pytest.approx(100 * 2 / 6)  # not per-file mean (50%)
    assert total['speaker_count_accuracy'] == .5
    assert total['speaker_count_mae'] == .5
    assert json.loads((tmp_path / 'manifest.json').read_text())['status'] == 'scored'
    assert (tmp_path / 'corpus_summary.csv').is_file()
    assert (tmp_path / 'der_results.csv').is_file()
    # Re-running an export must not overwrite an earlier recording.
    with pytest.raises(ValueError, match='Duplicate'):
        export(row('overlap'), audio, ref)


@pytest.mark.parametrize('change', [
    {'audio_id': '../escape'},
    {'speaker': ['A']},
    {'timestamps_end': [2., 100.]},
    {'timestamps_start': [float('nan'), 1.]},
])
def test_export_rejects_bad_annotations(tmp_path, change):
    namespace = {'sha256': sha256}
    exec(code_cells()['export-helper'], namespace)
    data = row()
    data.update(change)
    with pytest.raises(ValueError):
        namespace['export_record'](data, tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode', ['automatic', 'oracle'])
def test_notebook_inference_scoring_and_report(tmp_path, monkeypatch, mode):
    from diarize import pipeline
    from eval.der import evaluate_annotations

    file_id = 'C4HFERLpRfI'  # synthetic fixture, not the real recording
    audio, ref, hyp = [tmp_path / name for name in ('audio', 'reference', 'hypothesis')]
    for directory in (audio, ref, hyp):
        directory.mkdir()
    namespace = {'sha256': sha256}
    exec(code_cells()['configuration'], namespace)
    exec(code_cells()['export-helper'], namespace)
    data = row(file_id)
    data.update(timestamps_start=[0., 2.], timestamps_end=[2., 4.])
    record = namespace['export_record'](data, audio, ref)
    monkeypatch.setattr('utils.model.load_model', lambda *a: object())
    # Actual AHC, postprocessing and DER; only the encoder and VAD are synthetic.
    monkeypatch.setattr(pipeline, 'get_speech_segments', lambda *a, **k: [(0., 4.)])
    embeddings = np.array([[1., 0.], [1., 0.], [.8, .6], [.8, .6]])
    monkeypatch.setattr(pipeline, 'extract_embeddings_for_segments',
                        lambda *a: ([(0., 1.), (1., 2.), (2., 3.), (3., 4.)], embeddings))
    baseline = evaluate_annotations(load_rttm(ref), {file_id: [(0., 4., 'X')]}, collar=.25,
                                    durations={file_id: 4.})
    namespace.update(AUDIO=audio, REFERENCE=ref, HYPOTHESIS=hyp, RESULTS=tmp_path,
                     records=[record], dataset=[data], device='cpu', json=json,
                     MODEL=Path('model_0034.model'), manifest={}, SPEAKER_MODE=mode,
                     MODE_NAME='oracle_reference_count' if mode == 'oracle' else 'automatic',
                     tqdm=lambda items, **kwargs: items,
                     baseline_metrics=baseline if mode == 'oracle' else None)
    for cell_id in ('inference', 'score', 'diagnostic-report'):
        exec(code_cells()[cell_id], namespace)
    count = 2 if mode == 'oracle' else 1
    assert namespace['results'][file_id]['N_pred'] == count
    assert namespace['traces'][file_id]['n_clusters_before_postprocess'] == count
    assert namespace['traces'][file_id]['requested_n_speakers'] == (2 if mode == 'oracle' else None)
    assert json.loads((tmp_path / 'manifest.json').read_text())['status'] == 'complete'
    assert (tmp_path / 'diagnostics.csv').is_file()
    if mode == 'oracle':
        comparison = json.loads((tmp_path / 'automatic_vs_oracle.json').read_text())
        assert comparison[0]['delta'] < 0
