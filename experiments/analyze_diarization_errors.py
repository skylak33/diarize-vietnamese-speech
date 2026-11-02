"""Audit reference coverage and separate speaker-count from assignment errors."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def diagnostic_row(file_id, audio_duration, reference, metric, vad_segments=None,
                   max_reference_coverage=0.5, min_vad_outside_reference_sec=5.0):
    """Build one evidence row without changing annotations or scoring."""
    reference_intervals = _merged_intervals(reference)
    reference_speech = _duration(reference_intervals)
    first_reference = reference_intervals[0][0] if reference_intervals else None
    last_reference = reference_intervals[-1][1] if reference_intervals else None
    relation = ('exact' if metric['N_pred'] == metric['N_ref'] else
                'over' if metric['N_pred'] > metric['N_ref'] else 'under')
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
        speaker_error_case=error_case,
        DER=metric['DER'], Miss=metric['miss'], FalseAlarm=metric['false_alarm'],
        SpeakerError=metric['speaker_error'],
        reference_speaker_seconds=metric['duration'],
        speaker_error_seconds=speaker_error,
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
    parser.add_argument('--recording_id', action='append')
    parser.add_argument('--run_vad', action='store_true')
    parser.add_argument('--vad_threshold', type=float, default=0.5)
    parser.add_argument('--vad_min_silence_ms', type=int, default=200)
    parser.add_argument('--max_reference_coverage', type=float, default=0.5)
    parser.add_argument('--min_vad_outside_reference_sec', type=float, default=5.0)
    parser.add_argument('--collar', type=float, default=0.25)
    parser.add_argument('--skip_overlap', action='store_true')
    args = parser.parse_args()

    import soundfile as sf
    from eval.der import evaluate_annotations, load_rttm

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
    metrics = evaluate_annotations(subset_references, subset_hypotheses, args.collar,
                                   args.skip_overlap, durations)

    vad = {}
    if args.run_vad:
        from diarize.vad import get_speech_segments
        for file_id in selected:
            vad[file_id] = get_speech_segments(
                str(audios[file_id]), threshold=args.vad_threshold,
                min_silence_ms=args.vad_min_silence_ms)
    rows = [diagnostic_row(
        file_id, durations[file_id], references[file_id], metrics[file_id], vad.get(file_id),
        args.max_reference_coverage, args.min_vad_outside_reference_sec)
        for file_id in selected]
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
    print(json.dumps(payload['speaker_error_by_count'], indent=2))
    for row in rows:
        if row['possible_partial_reference']:
            print(f"Possible partial reference: {row['recording_id']}")


if __name__ == '__main__':
    main()
