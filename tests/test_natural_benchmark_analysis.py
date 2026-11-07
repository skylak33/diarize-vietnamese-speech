from experiments import analyze_natural_vietnamese_benchmark as analysis


def _row(configuration, recording_id, der, predicted=2):
    return {
        "mode": "full_pipeline", "overlap_condition": "included",
        "configuration": configuration, "recording_id": recording_id,
        "reference_speaker_count": 2, "predicted_speakers": predicted,
        "count_correct": predicted == 2, "DER": der, "Miss": 0.,
        "FalseAlarm": 0., "SpeakerError": der,
        "scored_duration_seconds": 100., "miss_seconds": 0.,
        "false_alarm_seconds": 0., "speaker_error_seconds": der,
    }


def test_paired_bootstrap_is_deterministic_and_candidate_minus_baseline():
    rows = [_row("fixed_ahc_075", "a", 10.),
            _row("robust_adaptive_ahc_075", "a", 20.),
            _row("fixed_ahc_075", "b", 5.),
            _row("robust_adaptive_ahc_075", "b", 5.)]
    first = analysis.paired_bootstrap(
        rows, "fixed_ahc_075", "robust_adaptive_ahc_075", draws=100, seed=42)
    second = analysis.paired_bootstrap(
        rows, "fixed_ahc_075", "robust_adaptive_ahc_075", draws=100, seed=42)
    assert first == second
    assert first[0]["DER_delta_candidate_minus_baseline"] == 5.
    assert (first[0]["improved"], first[0]["worsened"], first[0]["unchanged"]) == (0, 1, 1)


def test_confusion_and_failure_manifest_keep_count_mismatch():
    rows = [_row("fixed_ahc_075", "a", 10., predicted=4),
            _row("fixed_ahc_075", "b", 2.)]
    confusion = analysis.count_confusion(rows)
    assert {(row["predicted_speakers"], row["recordings"]) for row in confusion} == {
        (2, 1), (4, 1)}
    failures = analysis.failure_cases(rows, top_per_stratum=1)
    assert failures == [{
        "mode": "full_pipeline", "overlap_condition": "included",
        "configuration": "fixed_ahc_075", "recording_id": "a",
        "reference_speakers": 2, "predicted_speakers": 4, "DER": 10.,
        "Miss": 0., "FalseAlarm": 0., "SpeakerError": 10.,
        "reasons": "speaker_count_mismatch;top3_DER_in_stratum",
    }]
