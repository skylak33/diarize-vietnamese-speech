from argparse import Namespace
import pytest
from eval.datasets import generate_pairs, read_trials, read_manifest
from experiments.run_diarization_experiments import configurations
from experiments.plot_training import read_training_log


def test_deterministic_pairs_and_odd_count():
    records={str(i):{'speaker':str(i//3)} for i in range(9)}
    a=generate_pairs(records,101,42)
    assert a == generate_pairs(records,101,42)
    assert len(a) == 101 and sum(r[0] for r in a) == 50
    assert all(label == int(records[x]['speaker']==records[y]['speaker']) and x!=y for label,x,y in a)


def test_trials_fail_outside_test(tmp_path):
    p=tmp_path/'trials.txt'; p.write_text('1 a b\n0 a train\n')
    with pytest.raises(ValueError,match='outside'):
        read_trials(p, {'a':{'speaker':'A'},'b':{'speaker':'A'}})


def test_manifest_rejects_leakage(tmp_path):
    (tmp_path/'a.wav').touch()
    p=tmp_path/'manifest.csv'
    p.write_text('utterance_id,path,speaker_id,split\na,a.wav,A,train\nb,a.wav,A,test\n')
    with pytest.raises(ValueError,match='leakage'):
        read_manifest(p)


def test_manifest_rejects_paths_outside_root_and_duplicate_audio(tmp_path):
    root=tmp_path/'audio'; root.mkdir()
    (root/'nested').mkdir(); (root/'nested'/'a.wav').touch()
    outside=tmp_path/'outside.wav'; outside.touch()
    manifest=tmp_path/'manifest.csv'
    manifest.write_text('utterance_id,path,speaker_id,split\na,nested/a.wav,A,test\n')
    assert read_manifest(manifest, root)['a']['audio'] == str((root/'nested'/'a.wav').resolve())
    manifest.write_text('utterance_id,path,speaker_id,split\na,../outside.wav,A,test\n')
    with pytest.raises(ValueError,match='escapes'):
        read_manifest(manifest, root)
    manifest.write_text('utterance_id,path,speaker_id,split\na,nested/a.wav,A,test\nb,nested/a.wav,B,test\n')
    with pytest.raises(ValueError,match='Duplicate evaluation audio'):
        read_manifest(manifest, root)


def test_trials_require_complete_manifest_speaker_metadata(tmp_path):
    p=tmp_path/'trials.txt'; p.write_text('1 a b\n0 a c\n')
    records={'a':{'speaker':'A'},'b':{'speaker':None},'c':{'speaker':'C'}}
    with pytest.raises(ValueError,match='missing a speaker'):
        read_trials(p, records)


def test_controlled_ablation():
    args=Namespace(method='ahc',threshold=.65,vad_threshold=.5,window_sec=1.5,
                   step_sec=None,speaker_mode='automatic',experiment='all')
    configs=configurations(args)
    assert len(configs) < 20
    assert all(c['step_sec']==c['window_sec']/2 for c in configs)


def test_training_log(tmp_path):
    p=tmp_path/'score.txt'
    p.write_text('1 epoch, LR 0.001, LOSS 2.5, ACC 30.00%\n2 epoch, LR 9.7e-4, LOSS 2.0, ACC 40.00%\n')
    assert read_training_log(p)[1] == (2, {'LR':0.00097,'LOSS':2.0,'ACC':40.0})


def test_vlsp_notebook_format(tmp_path):
    for name in ('a.wav','b.wav','c.wav'):
        (tmp_path/name).touch()
    path=tmp_path/'public_test_pairs_with_label.txt'
    path.write_text('a.wav,b.wav,1\na.wav,c.wav,0\na.wav,b.wav,1\n',encoding='utf-8')
    records,trials=read_trials(path,audio_root=tmp_path,trials_format='vlsp')
    assert trials == [(1,'a.wav','b.wav'),(0,'a.wav','c.wav'),(1,'a.wav','b.wav')]
    assert len(records)==3  # Entire protocol retained, including repeated trials.


@pytest.mark.parametrize('row',['a.wav,b.wav','a.wav,b.wav,2','a.wav,,1','wav1,wav2,label'])
def test_vlsp_bad_rows_fail(tmp_path,row):
    path=tmp_path/'pairs.txt'; path.write_text(row)
    with pytest.raises(ValueError,match='expected wav1,wav2,label'):
        read_trials(path,audio_root=tmp_path,trials_format='vlsp')


def test_vlsp_missing_audio_not_skipped(tmp_path):
    path=tmp_path/'pairs.txt'; path.write_text('missing.wav,b.wav,1\na.wav,c.wav,0\n')
    with pytest.raises(FileNotFoundError):
        read_trials(path,audio_root=tmp_path,trials_format='vlsp')


def test_direct_trials_reject_path_escape_and_aliases(tmp_path):
    root=tmp_path/'audio'; root.mkdir()
    for name in ('a.wav','b.wav'):
        (root/name).touch()
    outside=tmp_path/'outside.wav'; outside.touch()
    path=tmp_path/'pairs.txt'
    path.write_text('../outside.wav,a.wav,1\na.wav,b.wav,0\n')
    with pytest.raises(ValueError,match='escapes'):
        read_trials(path,audio_root=root,trials_format='vlsp')
    path.write_text('a.wav,sub/../a.wav,1\na.wav,b.wav,0\n')
    with pytest.raises(ValueError,match='same audio'):
        read_trials(path,audio_root=root,trials_format='vlsp')
