import pytest
from eval.der import parse_rttm, speaker_count, evaluate_annotations, compute_der_pyannote
from utils.metrics import verification_metrics


def test_rttm_and_count(tmp_path):
    path = tmp_path / 'test.rttm'
    path.write_text('SPEAKER a 1 0 2 <NA> <NA> A <NA> <NA>\nSPEAKER a 1 0 2 <NA> <NA> B <NA> <NA>\n')
    data = parse_rttm(path)
    assert speaker_count(data['a']) == 2
    result = evaluate_annotations(data, data, collar=0)
    assert result['__total__']['duration'] == 4  # speaker-seconds, preserve overlap tracks
    assert result['__total__']['DER'] == 0


def test_corpus_weighting_and_missing_file():
    ref = {'long':[(0,9,'a')], 'missing':[(0,1,'b')]}
    hyp = {'long':[(0,9,'x')]}
    result = evaluate_annotations(ref,hyp,collar=0)
    assert result['missing']['DER'] == 100
    assert result['__total__']['DER'] == 10  # not mean(0,100)
    assert result['__total__']['miss_seconds'] == 1
    assert result['__total__']['speaker_count_accuracy'] == 0.5
    assert result['__total__']['speaker_count_mae'] == 0.5


def test_components_and_global_mapping():
    ref = {'a':[(0,2,'A'),(2,4,'B')]}
    hyp = {'a':[(0,4,'X'),(4,5,'Y')]}
    result = evaluate_annotations(ref,hyp,collar=0)['__total__']
    assert result['duration'] == 4
    assert result['speaker_error_seconds'] == 2
    assert result['false_alarm_seconds'] == 1
    assert result['DER'] == 75


def test_overlap_and_collar():
    ref={'a':[(0,2,'A'),(1,3,'B')]}
    hyp={'a':[(0,2,'X'),(2,3,'Y')]}
    assert evaluate_annotations(ref,hyp,0)['__total__']['miss_seconds'] == 1
    assert evaluate_annotations(ref,hyp,0,True)['__total__']['DER'] == 0
    assert evaluate_annotations({'a':[(0,2,'a')]}, {}, 0.25)['__total__']['duration'] == pytest.approx(1.75)


def test_directory_missing_and_empty(tmp_path):
    ref=tmp_path/'ref'; hyp=tmp_path/'hyp'
    ref.mkdir(); hyp.mkdir()
    (ref/'a.rttm').write_text('SPEAKER a 1 0 1 <NA> <NA> A <NA> <NA>\n')
    (hyp/'a.rttm').write_text('')
    assert compute_der_pyannote(hyp, ref, 0)['__total__']['DER'] == 100


def test_zero_reference_is_undefined_but_fa_retained():
    result=evaluate_annotations({'silent':[]},{'silent':[(0,2,'x')]},0)['__total__']
    assert result['DER'] is None
    assert result['false_alarm_seconds'] == 2


def test_all_hypotheses_missing(tmp_path):
    ref=tmp_path/'ref'; hyp=tmp_path/'hyp'
    ref.mkdir(); hyp.mkdir()
    (ref/'a.rttm').write_text('SPEAKER a 1 0 1 <NA> <NA> A <NA> <NA>\n')
    assert compute_der_pyannote(hyp,ref,0)['__total__']['DER'] == 100


def test_invalid_rttm_and_unknown_id(tmp_path):
    path=tmp_path/'bad.rttm'
    path.write_text('SPEAKER a 1 0 -1 <NA> <NA> A <NA> <NA>')
    with pytest.raises(ValueError):
        parse_rttm(path)
    with pytest.raises(ValueError,match='absent'):
        evaluate_annotations({'a':[(0,1,'A')]},{'b':[(0,1,'X')]})


@pytest.mark.parametrize('scores,labels,eer', [([1,0],[1,0],0),([0,1],[1,0],100),([0,0],[0,1],50)])
def test_verification_ties(scores,labels,eer):
    result=verification_metrics(scores,labels)
    assert result['EER'] == eer
    assert 0 <= result['minDCF'] <= 1


def test_verification_rejects_missing_class():
    with pytest.raises(ValueError):
        verification_metrics([0,1],[1,1])
