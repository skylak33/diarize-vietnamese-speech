"""Audit reference coverage and separate speaker-count from assignment errors."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COUNT_STATUSES = {'correct_count', 'overcount', 'undercount'}
OBSERVATIONS = {
    'speaker_assignment', 'boundary_error', 'vad_miss', 'vad_false_alarm',
    'possible_mixed_embedding', 'reference_issue', 'unknown',
}
CORRECT_COUNT_REVIEW_IDS = ('98EV6H0ZRlU', 'B5rLHPUJ8qc', 'jxTaydBnwIg', 'XSC94UbpoRQ')
REFERENCE_REVIEW_IDS = ('C4HFERLpRfI', 'Uaq5N3FbQb4', 'ed0Vt0NxJUI')


def _merged_intervals(segments):
    merged = []
    for start, end, *_ in sorted(segments):
        if not merged or start > merged[-1][1]:
            merged.append([float(start), float(end)])
        else:
            merged[-1][1] = max(merged[-1][1], float(end))
    return merged


def _duration(intervals):
    return sum(end - start for start, end in intervals)


def _overlap_duration(left, right):
    left, right = _merged_intervals(left), _merged_intervals(right)
    total = i = j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _annotation_segments(annotation):
    if hasattr(annotation, 'itertracks'):
        return [(float(segment.start), float(segment.end), str(label))
                for segment, _, label in annotation.itertracks(yield_label=True)]
    return [(float(start), float(end), str(label)) for start, end, label in annotation]


def _annotation_speaker_count(annotation):
    return len({speaker for _, _, speaker in _annotation_segments(annotation)})


def _timeline_intervals(timeline):
    if hasattr(timeline, 'support'):
        return [(float(segment.start), float(segment.end)) for segment in timeline.support()]
    return [(float(start), float(end)) for start, end, *_ in timeline]


def _active_labels(segments, start, end):
    return sorted({label for left, right, label in segments if left < end and right > start})


def _complete_mapping(reference, hypothesis, mapping):
    """Give unmatched hypothesis labels distinct names outside the reference namespace."""
    result = {str(key): str(value) for key, value in mapping.items()}
    used = {speaker for _, _, speaker in reference} | set(result.values())
    for index, speaker in enumerate(sorted({label for _, _, label in hypothesis})):
        if speaker in result:
            continue
        candidate = f'__unmapped_hypothesis_{index}__'
        while candidate in used:
            candidate += '_'
        result[speaker] = candidate
        used.add(candidate)
    return result


def temporal_error_intervals(reference, hypothesis, effective_uem, mapping):
    """Describe atomic scored intervals from scorer-prepared annotations.

    ``reference``, ``hypothesis`` and ``effective_uem`` must be the views returned
    by ``eval.der.prepare_evaluation``. This function deliberately does not apply
    collar, overlap removal, UEM cropping, or a second speaker mapping.
    """
    reference = _annotation_segments(reference)
    hypothesis = _annotation_segments(hypothesis)
    mapping = _complete_mapping(reference, hypothesis, mapping)
    uem = _timeline_intervals(effective_uem)
    boundaries = {point for start, end in uem for point in (start, end)}
    for start, end, _ in reference + hypothesis:
        for uem_start, uem_end in uem:
            if start < uem_end and end > uem_start:
                boundaries.update((max(start, uem_start), min(end, uem_end)))
    boundaries = sorted(boundaries)
    atoms = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start or not any(left <= start and end <= right for left, right in uem):
            continue
        ref_labels = _active_labels(reference, start, end)
        hyp_labels = _active_labels(hypothesis, start, end)
        mapped_labels = sorted({mapping[label] for label in hyp_labels})
        if not ref_labels and not hyp_labels:
            continue
        common = set(ref_labels) & set(mapped_labels)
        components = (
            [('correct', len(common))] if common else [])
        components += ([('speaker_confusion', min(len(ref_labels), len(mapped_labels)) - len(common))]
                       if min(len(ref_labels), len(mapped_labels)) > len(common) else [])
        components += ([('missed_speech', len(ref_labels) - len(mapped_labels))]
                       if len(ref_labels) > len(mapped_labels) else [])
        components += ([('false_alarm', len(mapped_labels) - len(ref_labels))]
                       if len(mapped_labels) > len(ref_labels) else [])
        for kind, speaker_count in components:
            atoms.append(dict(start=start, end=end, duration=end - start,
                              speaker_seconds=(end - start) * speaker_count,
                              speaker_count=speaker_count, kind=kind,
                              matched_speakers=sorted(common), reference_speakers=ref_labels,
                              hypothesis_speakers=hyp_labels,
                              mapped_hypothesis_speakers=mapped_labels))

    merged = []
    for atom in atoms:
        dynamic = {'start', 'end', 'duration', 'speaker_seconds'}
        comparable = {key: value for key, value in atom.items() if key not in dynamic}
        previous = merged[-1] if merged else None
        previous_comparable = ({key: value for key, value in previous.items()
                                if key not in dynamic} if previous else None)
        if previous and abs(previous['end'] - atom['start']) < 1e-9 and comparable == previous_comparable:
            previous['end'] = atom['end']
            previous['duration'] = previous['end'] - previous['start']
            previous['speaker_seconds'] = previous['duration'] * previous['speaker_count']
        else:
            merged.append(atom)
    return merged


def speaker_change_boundaries(reference, hypothesis, effective_uem, mapping, tolerance=0.25):
    """Match reference change points to the nearest mapped-hypothesis change."""
    if tolerance < 0:
        raise ValueError('tolerance must be non-negative')
    uem = _timeline_intervals(effective_uem)
    mapping = _complete_mapping(_annotation_segments(reference),
                                _annotation_segments(hypothesis), mapping)

    def changes(annotation, label_mapping=None):
        segments = _annotation_segments(annotation)
        points = []
        for point in sorted({value for start, end, _ in segments for value in (start, end)}):
            if not any(start < point < end for start, end in uem):
                continue
            before = _active_labels(segments, point - 1e-7, point)
            after = _active_labels(segments, point, point + 1e-7)
            if label_mapping:
                before = sorted({str(label_mapping.get(label, label)) for label in before})
                after = sorted({str(label_mapping.get(label, label)) for label in after})
            if before != after:
                points.append(point)
        return points

    reference_changes = changes(reference)
    hypothesis_changes = changes(hypothesis, mapping)
    result = []
    for point in reference_changes:
        nearest = min(hypothesis_changes, key=lambda candidate: (abs(candidate - point), candidate),
                      default=None)
        delta = None if nearest is None else nearest - point
        result.append(dict(reference_time=point, hypothesis_time=nearest, delta=delta,
                           within_tolerance=delta is not None and abs(delta) <= tolerance))
    return result


def temporal_diagnostic(reference, hypothesis, effective_uem, mapping,
                        long_error_threshold=2.0, boundary_tolerance=0.25):
    """Serialize scorer-prepared views into deterministic temporal diagnostics."""
    reference_segments = _annotation_segments(reference)
    hypothesis_segments = _annotation_segments(hypothesis)
    mapping = _complete_mapping(reference_segments, hypothesis_segments, mapping)
    intervals = temporal_error_intervals(reference_segments, hypothesis_segments,
                                         effective_uem, mapping)
    errors = [interval for interval in intervals if interval['kind'] != 'correct']
    return dict(
        effective_uem=[dict(start=start, end=end) for start, end in _timeline_intervals(effective_uem)],
        reference_intervals=[dict(start=start, end=end, speaker=speaker)
                             for start, end, speaker in reference_segments],
        mapped_hypothesis_intervals=[
            dict(start=start, end=end, speaker=speaker,
                 mapped_speaker=str(mapping.get(speaker, speaker)))
            for start, end, speaker in hypothesis_segments],
        speaker_mapping={str(key): str(value) for key, value in sorted(mapping.items())},
        intervals=intervals,
        error_intervals=errors,
        long_error_intervals=[interval for interval in errors
                              if interval['duration'] >= long_error_threshold],
        speaker_change_boundaries=speaker_change_boundaries(
            reference, hypothesis, effective_uem, mapping, boundary_tolerance),
    )


def diagnostic_row(file_id, audio_duration, reference, metric, vad_segments=None,
                   max_reference_coverage=0.5, min_vad_outside_reference_sec=5.0,
                   observations=None, temporal=None, reference_status='unverified',
                   long_error_threshold=2., boundary_tolerance=.25):
    """Build one evidence row without changing annotations or scoring."""
    reference_intervals = _merged_intervals(reference)
    reference_speech = _duration(reference_intervals)
    first_reference = reference_intervals[0][0] if reference_intervals else None
    last_reference = reference_intervals[-1][1] if reference_intervals else None
    relation = ('exact' if metric['N_pred'] == metric['N_ref'] else
                'over' if metric['N_pred'] > metric['N_ref'] else 'under')
    count_status = dict(exact='correct_count', over='overcount', under='undercount')[relation]
    speaker_error = metric['speaker_error_seconds']
    error_case = ('none' if speaker_error <= 1e-9 else
                  'timing_assignment_with_correct_count' if relation == 'exact' else
                  'count_mismatch_with_speaker_error')

    vad_total = vad_before_reference = vad_outside_reference = partial = None
    if vad_segments is not None:
        vad_intervals = _merged_intervals(vad_segments)
        vad_total = _duration(vad_intervals)
        vad_before_reference = (_overlap_duration(vad_intervals, [(0, first_reference)])
                                if first_reference is not None else vad_total)
        vad_outside_reference = vad_total - _overlap_duration(vad_intervals, reference_intervals)
        partial = (reference_speech / audio_duration < max_reference_coverage
                   and vad_outside_reference >= min_vad_outside_reference_sec)

    observations = list(dict.fromkeys(observations or []))
    invalid = set(observations) - OBSERVATIONS
    if invalid:
        raise ValueError(f'Unknown observations: {sorted(invalid)}')

    confusion_intervals = ([interval for interval in temporal['error_intervals']
                            if interval['kind'] == 'speaker_confusion'] if temporal else [])
    boundary_zones = (_merged_intervals([
        (max(0., point - boundary_tolerance), point + boundary_tolerance)
        for start, end, _ in reference for point in (start, end)]) if temporal else [])
    boundary_confusion = sum(
        _overlap_duration([(item['start'], item['end'])], boundary_zones) * item['speaker_count']
        for item in confusion_intervals)
    return dict(
        recording_id=file_id,
        audio_duration=audio_duration,
        reference_speech_seconds=reference_speech,
        reference_coverage_ratio=reference_speech / audio_duration if audio_duration else None,
        first_reference_start=first_reference,
        last_reference_end=last_reference,
        vad_speech_seconds=vad_total,
        vad_speech_before_first_reference_seconds=vad_before_reference,
        vad_speech_outside_reference_seconds=vad_outside_reference,
        possible_partial_reference=partial,
        n_ref=metric['N_ref'], n_pred=metric['N_pred'], count_relation=relation,
        speaker_count_status=count_status, observations=observations,
        speaker_error_case=error_case,
        DER=metric['DER'], Miss=metric['miss'], FalseAlarm=metric['false_alarm'],
        SpeakerError=metric['speaker_error'],
        reference_speaker_seconds=metric['duration'],
        miss_seconds=metric['miss_seconds'], false_alarm_seconds=metric['false_alarm_seconds'],
        speaker_error_seconds=speaker_error,
        long_confusion_seconds=sum(item['speaker_seconds'] for item in confusion_intervals
                                   if item['duration'] >= long_error_threshold),
        boundary_adjacent_confusion_seconds=boundary_confusion,
        error_intervals=temporal['error_intervals'] if temporal else [],
        reference_status=reference_status,
        temporal=temporal,
    )


def speaker_error_summary(rows):
    """Aggregate confusion separately for correct and incorrect speaker counts."""
    groups = {}
    for name, selected in (
            ('correct_count', [row for row in rows if row['count_relation'] == 'exact']),
            ('wrong_count', [row for row in rows if row['count_relation'] != 'exact']),
            ('overcount', [row for row in rows if row['count_relation'] == 'over']),
            ('undercount', [row for row in rows if row['count_relation'] == 'under'])):
        reference = sum(row['reference_speaker_seconds'] for row in selected)
        error = sum(row['speaker_error_seconds'] for row in selected)
        groups[name] = dict(recordings=len(selected), reference_speaker_seconds=reference,
                            miss_seconds=sum(row['miss_seconds'] for row in selected),
                            false_alarm_seconds=sum(row['false_alarm_seconds'] for row in selected),
                            speaker_error_seconds=error,
                            speaker_error_rate=100 * error / reference if reference else None)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--ref_rttm_dir', required=True)
    parser.add_argument('--hyp_rttm_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--csv')
    parser.add_argument('--summary_csv', help='Aggregate error components by count category')
    parser.add_argument('--top_csv', help='Ranked, flattened per-recording error table')
    parser.add_argument('--recording_id', action='append')
    parser.add_argument('--run_vad', action='store_true')
    parser.add_argument('--vad_threshold', type=float, default=0.5)
    parser.add_argument('--vad_min_silence_ms', type=int, default=200)
    parser.add_argument('--max_reference_coverage', type=float, default=0.5)
    parser.add_argument('--min_vad_outside_reference_sec', type=float, default=5.0)
    parser.add_argument('--collar', type=float, default=0.25)
    parser.add_argument('--skip_overlap', action='store_true')
    parser.add_argument('--uem')
    parser.add_argument('--speaker_count_status', choices=sorted(COUNT_STATUSES))
    parser.add_argument('--long_error_threshold', type=float, default=2.0)
    args = parser.parse_args()

    import soundfile as sf
    from eval.der import evaluate_annotations, load_rttm, prepare_evaluation

    references = load_rttm(args.ref_rttm_dir)
    hypotheses = load_rttm(args.hyp_rttm_dir, allow_empty=True)
    audios = {path.stem: path for path in Path(args.audio_dir).rglob('*')
              if path.suffix.lower() in {'.wav', '.flac', '.ogg'}}
    selected = sorted(args.recording_id or references)
    missing = set(selected) - references.keys() | (set(selected) - audios.keys())
    if missing:
        parser.error(f'Missing audio or reference IDs: {sorted(missing)}')
    durations = {file_id: sf.info(audios[file_id]).duration for file_id in selected}
    subset_references = {file_id: references[file_id] for file_id in selected}
    subset_hypotheses = {file_id: hypotheses.get(file_id, []) for file_id in selected}
    uem = None
    if args.uem:
        from eval.der import load_uem
        loaded_uem = load_uem(args.uem)
        missing_uem = set(selected) - loaded_uem.keys()
        if missing_uem:
            parser.error(f'Missing UEM IDs: {sorted(missing_uem)}')
        uem = {file_id: loaded_uem[file_id] for file_id in selected}
    metrics = evaluate_annotations(subset_references, subset_hypotheses, collar=args.collar,
                                   skip_overlap=args.skip_overlap, durations=durations,
                                   uem_data=uem)

    vad = {}
    if args.run_vad:
        from diarize.vad import get_speech_segments
        for file_id in selected:
            vad[file_id] = get_speech_segments(
                str(audios[file_id]), threshold=args.vad_threshold,
                min_silence_ms=args.vad_min_silence_ms)
    rows = []
    for file_id in selected:
        prepared = prepare_evaluation(
            file_id, references[file_id], hypotheses.get(file_id, []), args.collar,
            args.skip_overlap, None if uem is None else uem[file_id],
            duration=durations[file_id])
        temporal = temporal_diagnostic(*prepared, long_error_threshold=args.long_error_threshold,
                                       boundary_tolerance=args.collar)
        metric = metrics[file_id]
        if uem is not None:
            metric = dict(metric, N_ref=_annotation_speaker_count(prepared[0]),
                          N_pred=_annotation_speaker_count(prepared[1]))
        row = diagnostic_row(
            file_id, durations[file_id], references[file_id], metric, vad.get(file_id),
            args.max_reference_coverage, args.min_vad_outside_reference_sec, temporal=temporal,
            long_error_threshold=args.long_error_threshold, boundary_tolerance=args.collar)
        if args.speaker_count_status is None or row['speaker_count_status'] == args.speaker_count_status:
            rows.append(row)
    if not rows:
        parser.error('No recordings match the requested speaker-count status')
    payload = dict(
        protocol=dict(collar=args.collar, skip_overlap=args.skip_overlap,
                      vad_enabled=args.run_vad, vad_threshold=args.vad_threshold,
                      vad_min_silence_ms=args.vad_min_silence_ms,
                      partial_reference_heuristic=dict(
                          max_reference_coverage=args.max_reference_coverage,
                          min_vad_outside_reference_sec=args.min_vad_outside_reference_sec)),
        speaker_error_by_count=speaker_error_summary(rows),
        recordings=rows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = payload['speaker_error_by_count']
        fields = ['category', *next(iter(summary.values())).keys()]
        with summary_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for category, values in summary.items():
                writer.writerow({'category': category, **values})
    if args.top_csv:
        fields = [
            'rank', 'recording_id', 'speaker_count_status', 'n_ref', 'n_pred',
            'speaker_error_seconds', 'SpeakerError', 'miss_seconds', 'Miss',
            'false_alarm_seconds', 'FalseAlarm', 'DER', 'long_confusion_seconds',
            'boundary_adjacent_confusion_seconds', 'long_confusion_interval_count']
        ranked = sorted(rows, key=lambda row: (-row['speaker_error_seconds'], row['recording_id']))
        top_path = Path(args.top_csv)
        top_path.parent.mkdir(parents=True, exist_ok=True)
        with top_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rank, row in enumerate(ranked, 1):
                writer.writerow({**{key: row.get(key) for key in fields}, 'rank': rank,
                                 'long_confusion_interval_count': sum(
                                     item['kind'] == 'speaker_confusion'
                                     and item['duration'] >= args.long_error_threshold
                                     for item in row['error_intervals'])})
    print(json.dumps(payload['speaker_error_by_count'], indent=2))
    for row in rows:
        if row['possible_partial_reference']:
            print(f"Possible partial reference: {row['recording_id']}")


if __name__ == '__main__':
    main()
