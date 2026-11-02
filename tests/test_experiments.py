"""Integration checks use synthetic fixtures; never publish these as NIR results."""
import csv
import json
import sys
import numpy as np
import soundfile as sf
from experiments import run_diarization_experiments as runner
from experiments import eval_checkpoints, plot_training


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


def test_centroid_summary_includes_all_targets_and_reasons():
    rows=[dict(merge_attempts=1,accepted_merges=0,rejected_merges=1,
               merge_candidates=[dict(best_similarity=.7,target_similarities=[
                   {'similarity':.7},{'similarity':.65}])],
               merge_rejection_reasons={'below_similarity_threshold':1})]
    summary=runner.centroid_merge_summary(rows)
    assert summary['best_similarity']['mean']==.7
    assert summary['all_target_similarities']['count']==2
    assert summary['rejection_reasons']=={'below_similarity_threshold':1}


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
