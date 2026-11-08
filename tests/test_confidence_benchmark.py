import copy

import numpy as np
import pytest

from experiments.benchmark_confidence_aware_merging import (
    audit_splits, infer, stage_metrics, summarize, reference_merge_audit,
    paired_comparisons, paired_summary)


def test_split_audit_detects_shared_sources_and_speakers():
    records = [dict(conversation_id='a', split='development', speaker_ids=['s1'],
        turns=[dict(source_recording_id='r1', source_hash='h1')]),
        dict(conversation_id='b', split='validation', speaker_ids=['s2'],
        turns=[dict(source_recording_id='r2', source_hash='h2')])]
    assert len(audit_splits(records)['speakers']) == 2
    for field in ('speaker_ids', 'turns'):
        contaminated = copy.deepcopy(records)
        contaminated[1][field] = contaminated[0][field]
        with pytest.raises(ValueError, match='leakage'):
            audit_splits(contaminated)


def test_infer_reuses_inputs_and_exposes_all_stages():
    embeddings = np.repeat(np.eye(2), 4, axis=0)
    windows = [(float(i), float(i + 1)) for i in range(8)]
    original = embeddings.copy()
    stages, trace = infer(embeddings, windows, 'confidence_moderate')
    assert np.array_equal(embeddings, original)
    assert set(stages) == {'initial', 'merged', 'smoothed', 'final'}
    assert len(set(stages['initial'][1])) == 2
    assert trace['accepted_merges'] == 0


def test_stage_evaluation_and_summary_pool_speaker_seconds():
    short = stage_metrics('a', [(0., 1.)], [0], [(0., 1., 'a')], 1., [(0., 1.)], [0])
    long = dict(short, DER=100., SpeakerError=100., duration=10.,
                miss_seconds=0., false_alarm_seconds=0., speaker_error_seconds=10.)
    rows = [dict(short, method='fixed_075', stage='final', runtime_seconds=0.),
            dict(long, method='fixed_075', stage='final', runtime_seconds=0.)]
    pooled = next(row for row in summarize(rows) if row['reference_K'] == 'all')
    assert pooled['DER'] == pytest.approx(1000. / (10. + short['duration']))
    assert pooled['DER'] != 50.


def test_reference_merge_audit_is_separate_from_inference():
    from diarize.cluster import merge_clusters_confidence_aware
    embeddings = np.ones((4, 2))
    windows = [(float(i), float(i + 1)) for i in range(4)]
    initial = np.array([0, 0, 1, 1])
    _, trace = merge_clusters_confidence_aware(embeddings, initial, return_diagnostics=True)
    audit = reference_merge_audit(initial, trace, windows, [(0., 2., 'a'), (2., 4., 'b')])
    assert audit[0]['cross_speaker']
    assert audit[0]['purity_after'] < audit[0]['purity_before']
    assert 'cross_speaker' not in trace['merge_decisions'][0]


def test_paired_differences_are_candidate_minus_baseline():
    base = dict(recording_id='r', stage='final', DER=5., SpeakerError=4.,
                count_error=0, ARI=1., purity=1., fragmentation=1.)
    rows = [dict(base, method='fixed_075'), dict(base, method='centroid'),
            dict(base, method='confidence_strict', DER=3.)]
    pairs = paired_comparisons(rows)
    summary = next(row for row in paired_summary(pairs)
                   if row['method'] == 'confidence_strict' and row['baseline'] == 'fixed_075')
    assert summary['improved'] == 1 and summary['mean_DER_delta'] == -2.
