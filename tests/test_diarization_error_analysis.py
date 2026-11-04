import pytest
import eval.der as der

from experiments.analyze_diarization_errors import (
    diagnostic_row, speaker_change_boundaries, speaker_error_summary,
    temporal_diagnostic, temporal_error_intervals,
)


def metric(n_ref, n_pred, speaker_error_seconds, duration=10):
    return dict(N_ref=n_ref, N_pred=n_pred, duration=duration,
                miss_seconds=0, false_alarm_seconds=0,
                speaker_error_seconds=speaker_error_seconds,
                DER=0, miss=0, false_alarm=0, speaker_error=100 * speaker_error_seconds / duration)


def test_partial_reference_evidence_and_error_classification():
    row = diagnostic_row('partial', 60, [(50, 60, 'A')], metric(1, 1, 2),
                         vad_segments=[(0, 40), (50, 60)])
    assert row['possible_partial_reference'] is True
    assert row['vad_speech_before_first_reference_seconds'] == 40
    assert row['vad_speech_outside_reference_seconds'] == 40
    assert row['speaker_error_case'] == 'timing_assignment_with_correct_count'
    assert row['speaker_count_status'] == 'correct_count'
    assert row['observations'] == []
    assert row['reference_status'] == 'unverified'
    assert row['error_intervals'] == []

    mismatch = diagnostic_row('over', 10, [(0, 10, 'A')], metric(1, 2, 3))
    assert mismatch['count_relation'] == 'over'
    assert mismatch['speaker_count_status'] == 'overcount'
    assert mismatch['speaker_error_case'] == 'count_mismatch_with_speaker_error'
    assert mismatch['possible_partial_reference'] is None


def test_speaker_error_summary_separates_count_causes():
    rows = [
        diagnostic_row('exact', 10, [(0, 10, 'A')], metric(1, 1, 2)),
        diagnostic_row('over', 10, [(0, 10, 'A')], metric(1, 2, 3)),
        diagnostic_row('under', 10, [(0, 10, 'A'), (0, 10, 'B')],
                       metric(2, 1, 5, duration=20)),
    ]
    summary = speaker_error_summary(rows)
    assert summary['correct_count']['speaker_error_rate'] == 20
    assert summary['wrong_count']['recordings'] == 2
    assert summary['wrong_count']['speaker_error_seconds'] == 8
    assert summary['wrong_count']['miss_seconds'] == 0
    assert summary['overcount']['recordings'] == summary['undercount']['recordings'] == 1


def test_observations_are_independent_and_validated():
    row = diagnostic_row('under', 10, [(0, 10, 'A')], metric(2, 1, 1),
                         observations=['boundary_error', 'possible_mixed_embedding'])
    assert row['speaker_count_status'] == 'undercount'
    assert row['observations'] == ['boundary_error', 'possible_mixed_embedding']
    with pytest.raises(ValueError, match='Unknown observations'):
        diagnostic_row('bad', 10, [(0, 10, 'A')], metric(1, 1, 1),
                       observations=['guessed_cause'])


def test_temporal_intervals_use_prepared_uem_and_global_mapping():
    reference = [(0, 3, 'A'), (3, 6, 'B')]
    hypothesis = [(0, 2, 'X'), (2, 6, 'Y')]
    intervals = temporal_error_intervals(reference, hypothesis, [(1, 5)], {'X': 'A', 'Y': 'B'})
    assert [(row['start'], row['end'], row['kind']) for row in intervals] == [
        (1, 2, 'correct'),
        (2, 3, 'speaker_confusion'),
        (3, 5, 'correct'),
    ]


def test_temporal_intervals_handle_overlap_and_uem_gaps():
    reference = [(0, 5, 'A'), (2, 3, 'B')]
    hypothesis = [(0, 5, 'X')]
    intervals = temporal_error_intervals(
        reference, hypothesis, [(1, 2.5), (3.5, 4)], {'X': 'A'})
    errors = [row for row in intervals if row['kind'] != 'correct']
    assert [(row['start'], row['end'], row['kind'], row['speaker_seconds'])
            for row in errors] == [(2, 2.5, 'missed_speech', .5)]
    assert any(row['start'] == 2 and row['kind'] == 'correct' for row in intervals)
    assert all(not (2.5 < row['start'] < 3.5) for row in intervals)


def test_overlap_can_have_correct_and_confused_speaker_components():
    intervals = temporal_error_intervals(
        [(0, 1, 'A'), (0, 1, 'B')], [(0, 1, 'X'), (0, 1, 'Y')], [(0, 1)],
        {'X': 'A', 'Y': 'C'})
    assert [(row['kind'], row['speaker_count'], row['speaker_seconds']) for row in intervals] == [
        ('correct', 1, 1), ('speaker_confusion', 1, 1)]


def test_temporal_diagnostic_serializes_long_errors_and_boundaries():
    reference = [(0, 2, 'A'), (2, 6, 'B')]
    hypothesis = [(0, 3, 'X'), (3, 6, 'Y')]
    result = temporal_diagnostic(reference, hypothesis, [(0, 6)], {'X': 'A', 'Y': 'B'},
                                 long_error_threshold=.75, boundary_tolerance=.25)
    assert result['speaker_mapping'] == {'X': 'A', 'Y': 'B'}
    assert result['long_error_intervals'][0]['kind'] == 'speaker_confusion'
    assert result['speaker_change_boundaries'][0] == {
        'reference_time': 2.0,
        'hypothesis_time': 3.0,
        'delta': 1.0,
        'within_tolerance': False,
    }
    row = diagnostic_row('sample', 6, reference, metric(2, 2, 1, duration=6),
                         temporal=result, long_error_threshold=.75)
    assert row['long_confusion_seconds'] == 1
    assert row['boundary_adjacent_confusion_seconds'] == .25


def test_prepared_collar_gaps_are_not_reintroduced():
    # The scorer-prepared reference has already removed the collar around t=2.
    reference = [(0, 1.75, 'A'), (2.25, 4, 'B')]
    hypothesis = [(0, 4, 'X')]
    intervals = temporal_error_intervals(reference, hypothesis, [(0, 1.75), (2.25, 4)],
                                         {'X': 'A'})
    assert not any(row['start'] < 2.25 and row['end'] > 1.75 for row in intervals)
    assert speaker_change_boundaries(reference, hypothesis, [(0, 1.75), (2.25, 4)],
                                     {'X': 'A'}) == []


@pytest.mark.skipif(not hasattr(der, 'prepare_evaluation'),
                    reason='UEM preparation contract is integrated by the lead agent')
def test_temporal_components_match_prepared_scorer_view():
    reference = [(0, 5, 'A'), (2, 3, 'B')]
    hypothesis = [(0, 5, 'X')]
    metric = der.evaluate_annotations({'sample': reference}, {'sample': hypothesis}, collar=0,
                                      durations={'sample': 5})['sample']
    prepared = der.prepare_evaluation('sample', reference, hypothesis, collar=0, duration=5)
    result = temporal_diagnostic(*prepared)
    totals = {kind: sum(row['speaker_seconds'] for row in result['error_intervals']
                        if row['kind'] == kind)
              for kind in ('missed_speech', 'false_alarm', 'speaker_confusion')}
    assert totals == {
        'missed_speech': metric['miss_seconds'],
        'false_alarm': metric['false_alarm_seconds'],
        'speaker_confusion': metric['speaker_error_seconds'],
    }


def test_unmapped_hypothesis_label_cannot_collide_with_reference_label():
    reference = [(0, 1, 'X')]
    hypothesis = [(0, 1, 'A'), (0, 1, 'X')]
    metric = der.evaluate_annotations({'sample': reference}, {'sample': hypothesis}, collar=0,
                                      durations={'sample': 1})['sample']
    result = temporal_diagnostic(*der.prepare_evaluation(
        'sample', reference, hypothesis, collar=0, duration=1))
    false_alarm = sum(row['speaker_seconds'] for row in result['error_intervals']
                      if row['kind'] == 'false_alarm')
    assert false_alarm == metric['false_alarm_seconds'] == 1
    assert len(set(result['speaker_mapping'].values())) == 2


@pytest.mark.skipif(not hasattr(der, 'prepare_evaluation'),
                    reason='UEM preparation contract is integrated by the lead agent')
def test_external_uem_prepared_view_excludes_outside_speakers():
    reference, hypothesis, _, _ = der.prepare_evaluation(
        'sample', [(0, 1, 'A'), (2, 3, 'B')], [(0, 1, 'X')], collar=0,
        uem_intervals=[(0, 1)], duration=3)
    assert {label for _, _, label in reference.itertracks(yield_label=True)} == {'A'}
    assert {label for _, _, label in hypothesis.itertracks(yield_label=True)} == {'X'}
