"""Integration checks use synthetic fixtures; never publish these as NIR results."""
import csv
import json
import sys
import numpy as np
import soundfile as sf
from argparse import Namespace
from experiments import run_diarization_experiments as runner
from experiments import eval_checkpoints, plot_training


def test_clustering_count_comparison_is_opt_in_and_reuses_one_cache():
    args = Namespace(method='ahc', threshold=.65, vad_threshold=.5, window_sec=1.5,
                     step_sec=None, speaker_mode='automatic', experiment='clustering_count')
    configs = runner.configurations(args)
    assert [(config['method'], config['count_estimator']) for config in configs] == [
        ('ahc', None), ('ahc', 'adaptive_ahc'), ('spectral', None), ('nme_sc', None)]
    assert {config['speaker_mode'] for config in configs} == {'automatic'}


def test_oracle_count_diagnostics_are_paired_and_explicitly_diagnostic():
    row = dict(N_ref=2, N_pred=1, DER=20., speaker_error=15.,
               speaker_error_seconds=6., abs_error=1)
    oracle = dict(N_ref=2, N_pred=2, DER=8., speaker_error=3.,
                  speaker_error_seconds=1., abs_error=0)
    totals = dict(DER=20., speaker_error=15., speaker_count_mae=1.,
                  speaker_error_seconds=6., speaker_count_accuracy=0.,
                  average_predicted_speakers=1.)
    oracle_totals = dict(DER=8., speaker_error=3., speaker_count_mae=0.,
                         speaker_error_seconds=1., speaker_count_accuracy=1.,
                         average_predicted_speakers=2.)
    artifact = runner.oracle_count_diagnostics({
        'automatic': {'a':row, '__total__':totals},
        'known': {'a':oracle, '__total__':oracle_totals},
    })
    assert 'not an automatic diarization result' in artifact['condition']
    assert artifact['aggregate']['DER_delta'] == 12.
    assert artifact['aggregate']['recordings_improved'] == 1
    assert artifact['recordings'][0]['oracle_count_output_speakers'] == 2
    assert artifact['recordings'][0]['speaker_error_delta'] == 12.
    assert artifact['recordings'][0]['speaker_count_delta'] == -1


def test_runner_outputs_and_oracle_count(tmp_path, monkeypatch):
    audio=tmp_path/'audio'; ref=tmp_path/'ref'
    audio.mkdir(); ref.mkdir()
    sf.write(audio/'a.wav',np.zeros(16000),16000)
    truth='SPEAKER a 1 0 1 <NA> <NA> A <NA> <NA>\n'
    (ref/'a.rttm').write_text(truth)
    checkpoint=tmp_path/'model.model'; checkpoint.write_bytes(b'synthetic test fixture')
    counts=[]
    cache_ids=[]
    def fake_diarize(audio_path, model_path, **kwargs):
        from pathlib import Path
        counts.append(kwargs['n_speakers'])
        cache_ids.append(id(kwargs['embedding_cache']))
        path=Path(kwargs['output_rttm']); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(truth)
    monkeypatch.setattr('diarize.pipeline.diarize',fake_diarize)
    monkeypatch.setattr('utils.model.load_model',lambda *args:object())
    out=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',['runner','--audio_dir',str(audio),'--ref_rttm_dir',str(ref),
        '--model',str(checkpoint),'--output_dir',str(out),'--experiment','speaker_count','--collar','0'])
    runner.main()
    rows=list(csv.DictReader((out/'results.csv').open()))
    assert len(rows)==2 and counts==[1,None] and len(set(cache_ids))==1
    assert rows[0]['DER']=='0.0' and rows[0]['SpeakerCountAccuracy']=='1.0'
    details=json.loads((out/'config_001'/'der.json').read_text())
    assert details['a']['N_pred']==1
    diagnostics=list(csv.DictReader((out/'config_001'/'diagnostics.csv').open()))
    required={'recording_id','model_checkpoint','n_clusters_before_postmerge',
              'n_clusters_after_postmerge','n_speakers_after_postprocess','DER',
              'SpeakerError','merge_attempts','accepted_merges','rejected_merges',
              'merge_candidates','merge_rejection_reasons'}
    assert required <= diagnostics[0].keys()
    summary=rows[0]
    assert summary['OverclusteringRate']==summary['UnderclusteringRate']=='0.0'
    assert summary['AveragePredictedSpeakers']==summary['AverageReferenceSpeakers']=='1.0'
    assert float(summary['ElapsedSeconds']) >= 0 and float(summary['PeakPythonMemoryMB']) >= 0
    assert (out/'experiment_manifest.json').is_file()
    summary=json.loads((out/'config_001'/'centroid_merge_summary.json').read_text())
    assert summary['merge_attempts']==0 and summary['best_similarity']['count']==0
    oracle_artifact=json.loads((out/'oracle_count_diagnostics.json').read_text())
    assert oracle_artifact['delta_definition']=='automatic minus oracle-count'
    assert oracle_artifact['recordings'][0]['recording_id']=='a'


def test_centroid_summary_includes_all_targets_and_reasons():
    rows=[dict(merge_attempts=1,accepted_merges=0,rejected_merges=1,
               merge_candidates=[dict(best_similarity=.7,target_similarities=[
                   {'similarity':.7},{'similarity':.65}])],
               merge_rejection_reasons={'below_similarity_threshold':1})]
    summary=runner.centroid_merge_summary(rows)
    assert summary['best_similarity']['mean']==.7
    assert summary['all_target_similarities']['count']==2
    assert summary['rejection_reasons']=={'below_similarity_threshold':1}


def test_baseline_writes_full_corpus_artifact_contract(tmp_path, monkeypatch):
    audio=tmp_path/'audio'; ref=tmp_path/'ref'; audio.mkdir(); ref.mkdir()
    sf.write(audio/'a.wav',np.zeros(16000),16000)
    truth='SPEAKER a 1 0 1 <NA> <NA> A <NA> <NA>\n'
    (ref/'a.rttm').write_text(truth)
    checkpoint=tmp_path/'model.model'; checkpoint.write_bytes(b'synthetic test fixture')
    def fake_diarize(audio_path, model_path, **kwargs):
        from pathlib import Path
        path=Path(kwargs['output_rttm']); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(truth)
        kwargs['diagnostics'].update(n_speakers_after_postprocess=1)
    monkeypatch.setattr('diarize.pipeline.diarize',fake_diarize)
    monkeypatch.setattr('utils.model.load_model',lambda *args:object())
    out=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',['runner','--audio_dir',str(audio),'--ref_rttm_dir',str(ref),
        '--model',str(checkpoint),'--output_dir',str(out),'--experiment','baseline','--collar','0'])
    runner.main()
    assert {'manifest.json','recording_results.json','recording_results.csv',
            'corpus_results.json','hypothesis_rttm'} <= {path.name for path in out.iterdir()}
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['files_evaluated']==1
    assert manifest['encoder_source_sha256']
    record=json.loads((out/'recording_results.json').read_text())[0]
    assert record['audio_sha256'] and record['reference_sha256'] and record['rttm_sha256']
    assert record['speaker_count']==1 and record['runtime_seconds'] >= 0


def test_plot_outputs(tmp_path,monkeypatch):
    log=tmp_path/'score.txt'; log.write_text('1 epoch, LR .001, LOSS 2, ACC 30%\n2 epoch, LR .0009, LOSS 1, ACC 50%\n')
    eer=tmp_path/'eer.csv'; eer.write_text('epoch,EER\n1,50\n2,30\n')
    out=tmp_path/'plots'
    monkeypatch.setattr(sys,'argv',['plot','--log',str(log),'--eer_csv',str(eer),'--output_dir',str(out)])
    plot_training.main()
    assert {p.name for p in out.glob('*.png')}=={'training_loss.png','training_accuracy.png','training_lr.png','eer_by_epoch.png'}
    assert all(p.stat().st_size>1000 for p in out.glob('*.png'))


def test_checkpoint_selection(tmp_path):
    for epoch in [27,30,34]:
        (tmp_path/f'model_{epoch:04d}.model').touch()
    assert [epoch for epoch,_ in eval_checkpoints.select_checkpoints(tmp_path,2)]==[27,34]
