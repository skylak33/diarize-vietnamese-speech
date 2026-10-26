"""Reports for the paired ViYT count experiment; never used by inference."""
import csv
import json
import math
from pathlib import Path

from eval.der import evaluate_annotations, load_rttm
from utils.provenance import sha256

REVIEW_IDS = ('C4HFERLpRfI', 'Uaq5N3FbQb4', 'ed0Vt0NxJUI')


def validate_baseline(directory, manifest, records):
    """Reject incomparable baselines and recompute their metrics from saved RTTM."""
    directory = Path(directory)
    old = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if old.get('status') != 'complete' or old.get('speaker_mode') != 'automatic':
        raise ValueError('Baseline must be a completed automatic experiment')
    for key in ('dataset', 'revision', 'split', 'files_expected', 'model_sha256',
                'seed', 'pipeline', 'collar', 'skip_overlap', 'device', 'cuda_version',
                'gpu', 'rate_units', 'duration_units'):
        if old.get(key) != manifest.get(key):
            raise ValueError(f'Baseline mismatch: {key}')
    for key in ('python', 'packages'):
        if old['environment'][key] != manifest['environment'][key]:
            raise ValueError(f'Baseline environment mismatch: {key}; use its saved versions')
    # Only pipeline instrumentation and report code may differ in this experiment.
    for file, digest in old['code_sha256'].items():
        if file != 'diarize/pipeline.py' and not file.startswith('experiments/'):
            if manifest['code_sha256'].get(file) != digest:
                raise ValueError(f'Baseline code mismatch: {file}')
    old_records = json.loads((directory / 'recordings.json').read_text(encoding='utf-8'))
    current = {row['file_id']: row for row in records}
    previous = {row['file_id']: row for row in old_records}
    if (len(current) != len(records) or len(previous) != len(old_records)
            or current != previous or len(records) != old['files_evaluated']):
        raise ValueError('Baseline recordings/audio/reference hashes or counts differ')
    for file_id, row in current.items():
        if sha256(directory / 'reference_rttm' / f'{file_id}.rttm') != row['reference_sha256']:
            raise ValueError(f'Baseline reference RTTM changed: {file_id}')
    refs = load_rttm(directory / 'reference_rttm')
    hyps = load_rttm(directory / 'hypothesis_rttm', allow_empty=True)
    if refs.keys() != current.keys() or hyps.keys() != current.keys():
        raise ValueError('Baseline RTTM must contain every recording exactly once')
    measured = evaluate_annotations(refs, hyps, old['collar'], old['skip_overlap'],
                                    {k: r['duration'] for k, r in current.items()})
    saved = json.loads((directory / 'der_results.json').read_text(encoding='utf-8'))
    if saved.keys() != measured.keys():
        raise ValueError('Baseline metric IDs differ from RTTM')
    for file_id, row in measured.items():
        for key, value in row.items():
            original = saved[file_id].get(key)
            if value is None and original is None:
                continue
            if original is None or not math.isclose(value, original, rel_tol=1e-9, abs_tol=1e-7):
                raise ValueError(f'Baseline metric does not match RTTM: {file_id}/{key}')
    return old, measured


def speech_duration(segments):
    """Union of speech intervals, in seconds, BEFORE collar/overlap exclusions."""
    total, last_end = 0., 0.
    for start, end, _ in sorted(segments):
        total += max(0., end - max(start, last_end))
        last_end = max(last_end, end)
    return total


def diagnostic_rows(records, references, hypotheses, results, traces):
    rows = []
    for record in records:
        file_id = record['file_id']
        metric = results[file_id]
        trace = traces[file_id]
        rows.append(dict(audio_id=file_id, N_ref=metric['N_ref'], N_pred=metric['N_pred'],
                         DER=metric['DER'], Miss=metric['miss'], FalseAlarm=metric['false_alarm'],
                         SpeakerError=metric['speaker_error'],
                         reference_speech_duration=speech_duration(references[file_id]),
                         predicted_speech_duration=speech_duration(hypotheses.get(file_id, [])),
                         scored_reference_speaker_time=metric['duration'],
                         audio_duration=record['duration'], **trace,
                         speakers_lost_in_postprocess=(trace['n_clusters_before_postprocess']
                                                      - metric['N_pred'])))
    return rows


def comparison_rows(baseline, oracle):
    """Delta = oracle minus automatic; rate deltas are percentage points."""
    names = [('DER', 'DER'), ('Miss', 'miss'), ('FalseAlarm', 'false_alarm'),
             ('SpeakerError', 'speaker_error'), ('SpeakerCountAccuracy', 'speaker_count_accuracy'),
             ('SpeakerCountMAE', 'speaker_count_mae')]
    return [dict(metric=label, automatic=baseline[key], oracle=oracle[key],
                 delta=(oracle[key] - baseline[key])
                 if oracle[key] is not None and baseline[key] is not None else None)
            for label, key in names]


def save_table(rows, path):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
