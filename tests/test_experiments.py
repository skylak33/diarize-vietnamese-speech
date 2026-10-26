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
    def fake_diarize(audio_path, model_path, **kwargs):
        from pathlib import Path
        counts.append(kwargs['n_speakers'])
        path=Path(kwargs['output_rttm']); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(truth)
    monkeypatch.setattr('diarize.pipeline.diarize',fake_diarize)
    monkeypatch.setattr('utils.model.load_model',lambda *args:object())
    out=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',['runner','--audio_dir',str(audio),'--ref_rttm_dir',str(ref),
        '--model',str(checkpoint),'--output_dir',str(out),'--experiment','speaker_count','--collar','0'])
    runner.main()
    rows=list(csv.DictReader((out/'results.csv').open()))
    assert len(rows)==2 and counts==[1,None]
    assert rows[0]['DER']=='0.0' and rows[0]['SpeakerCountAccuracy']=='1.0'
    details=json.loads((out/'config_001'/'der.json').read_text())
    assert details['a']['N_pred']==1
    assert (out/'experiment_manifest.json').is_file()


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
