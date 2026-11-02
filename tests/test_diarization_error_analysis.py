from experiments.analyze_diarization_errors import diagnostic_row, speaker_error_summary


def metric(n_ref, n_pred, speaker_error_seconds, duration=10):
    return dict(N_ref=n_ref, N_pred=n_pred, duration=duration,
                speaker_error_seconds=speaker_error_seconds,
                DER=0, miss=0, false_alarm=0, speaker_error=100 * speaker_error_seconds / duration)


def test_partial_reference_evidence_and_error_classification():
    row = diagnostic_row('partial', 60, [(50, 60, 'A')], metric(1, 1, 2),
                         vad_segments=[(0, 40), (50, 60)])
    assert row['possible_partial_reference'] is True
    assert row['vad_speech_before_first_reference_seconds'] == 40
    assert row['vad_speech_outside_reference_seconds'] == 40
    assert row['speaker_error_case'] == 'timing_assignment_with_correct_count'

    mismatch = diagnostic_row('over', 10, [(0, 10, 'A')], metric(1, 2, 3))
    assert mismatch['count_relation'] == 'over'
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
    assert summary['overcount']['recordings'] == summary['undercount']['recordings'] == 1
