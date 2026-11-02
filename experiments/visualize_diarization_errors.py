"""Render scorer-aligned diarization timelines and annotation-review figures."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')


COLORS = {
    'correct': '#4caf50',
    'speaker_confusion': '#e53935',
    'missed_speech': '#fb8c00',
    'false_alarm': '#8e24aa',
}


def _save(fig, output):
    output = Path(output)
    if output.suffix.lower() not in {'.png', '.svg'}:
        raise ValueError('Output must use .png or .svg')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=160)


def plot_timeline(recording, output, title_suffix=''):
    """Plot reference, globally mapped hypothesis, and scorer-aligned error lanes."""
    import matplotlib.pyplot as plt

    temporal = recording['temporal']
    reference = temporal['reference_intervals']
    hypothesis = temporal['mapped_hypothesis_intervals']
    errors = temporal['error_intervals']
    ref_speakers = sorted({segment['speaker'] for segment in reference})
    hyp_speakers = sorted({segment['mapped_speaker'] for segment in hypothesis})
    labels = ([f'Ref {speaker}' for speaker in ref_speakers]
              + [f'Hyp→{speaker}' for speaker in hyp_speakers] + ['Errors'])
    rows = {label: index for index, label in enumerate(reversed(labels))}
    fig, ax = plt.subplots(figsize=(12, max(2.8, .45 * len(labels))))
    for segment in reference:
        label = f"Ref {segment['speaker']}"
        ax.broken_barh([(segment['start'], segment['end'] - segment['start'])],
                       (rows[label] - .32, .64), facecolors='#1976d2')
    for segment in hypothesis:
        label = f"Hyp→{segment['mapped_speaker']}"
        ax.broken_barh([(segment['start'], segment['end'] - segment['start'])],
                       (rows[label] - .32, .64), facecolors='#00897b')
    for interval in errors:
        ax.broken_barh([(interval['start'], interval['duration'])],
                       (rows['Errors'] - .32, .64), facecolors=COLORS[interval['kind']],
                       label=interval['kind'])
    for interval in temporal['effective_uem']:
        ax.axvspan(interval['start'], interval['end'], color='#eeeeee', alpha=.25, zorder=-2)
    handles, legend_labels = ax.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=8)
    error_duration = sum(interval.get('speaker_seconds', interval['duration']) for interval in errors)
    suffix = f' — {title_suffix}' if title_suffix else ''
    ax.set_title(f"{recording['recording_id']} — temporal errors {error_duration:.3f}s{suffix}")
    ax.set_xlabel('Time (s)')
    ax.set_yticks(list(rows.values()), list(rows.keys()))
    ax.grid(axis='x', alpha=.25)
    _save(fig, output)
    plt.close(fig)


def plot_reference_review(recording_id, audio_path, reference, vad, suspected_unannotated,
                          speaker_changes, output):
    """Plot human-review evidence without creating or changing reference labels."""
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, always_2d=True)
    waveform = audio.mean(axis=1)
    times = np.arange(len(waveform)) / sample_rate
    speakers = sorted({speaker for _, _, speaker in reference})
    rows = {speaker: index for index, speaker in enumerate(speakers)}
    fig, (wave_ax, track_ax) = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1]})
    wave_ax.plot(times, waveform, linewidth=.35, color='#37474f')
    for start, end in vad:
        wave_ax.axvspan(start, end, color='#43a047', alpha=.14)
    for start, end in suspected_unannotated:
        wave_ax.axvspan(start, end, color='#f4511e', alpha=.25)
    for change in speaker_changes:
        wave_ax.axvline(change, color='#d81b60', linestyle='--', linewidth=1)
    for start, end, speaker in reference:
        track_ax.broken_barh([(start, end - start)], (rows[speaker] - .32, .64),
                             facecolors='#1976d2')
    track_ax.set_yticks(list(rows.values()), list(rows.keys()))
    track_ax.set_xlabel('Time (s)')
    track_ax.set_ylabel('Reference')
    wave_ax.set_ylabel('Amplitude')
    wave_ax.set_title(f'{recording_id} — annotation review (VAD is evidence, not ground truth)')
    _save(fig, output)
    plt.close(fig)


def _load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return payload if isinstance(payload, list) else payload['recordings']


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Use --speaker_count_status undercount for all undercount cases. '
        'Priority correct-count IDs: 98EV6H0ZRlU, B5rLHPUJ8qc, jxTaydBnwIg, XSC94UbpoRQ. '
        'Reference-review IDs: C4HFERLpRfI, Uaq5N3FbQb4, ed0Vt0NxJUI.'))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--diagnostics')
    source.add_argument('--review_manifest', help=(
        'JSON list with recording_id, audio_path, reference, vad, '
        'suspected_unannotated, and speaker_changes'))
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--recording_id', action='append')
    parser.add_argument('--speaker_count_status', choices=['correct_count', 'overcount', 'undercount'])
    parser.add_argument('--format', choices=['png', 'svg'], default='png')
    args = parser.parse_args()
    selected_ids = set(args.recording_id or [])
    if args.review_manifest:
        manifest_dir = Path(args.review_manifest).resolve().parent
        recordings = [recording for recording in _load_manifest(args.review_manifest)
                      if not selected_ids or recording['recording_id'] in selected_ids]
        if not recordings:
            parser.error('No review recordings match the requested filters')
        for recording in recordings:
            audio_path = Path(recording['audio_path'])
            if not audio_path.is_absolute():
                audio_path = manifest_dir / audio_path
            plot_reference_review(
                recording['recording_id'], audio_path,
                [tuple(segment) for segment in recording['reference']],
                [tuple(interval) for interval in recording.get('vad', [])],
                [tuple(interval) for interval in recording.get('suspected_unannotated', [])],
                recording.get('speaker_changes', []), Path(args.output_dir) /
                f"{recording['recording_id']}.{args.format}")
        return
    recordings = [recording for recording in _load_manifest(args.diagnostics)
                  if (not selected_ids or recording['recording_id'] in selected_ids)
                  and (args.speaker_count_status is None
                       or recording['speaker_count_status'] == args.speaker_count_status)]
    if not recordings:
        parser.error('No recordings match the requested filters')
    protocol = json.loads(Path(args.diagnostics).read_text(encoding='utf-8')).get('protocol', {})
    suffix = f"collar={protocol.get('collar')}, skip_overlap={protocol.get('skip_overlap')}"
    for recording in recordings:
        plot_timeline(recording, Path(args.output_dir) /
                      f"{recording['recording_id']}.{args.format}", suffix)


if __name__ == '__main__':
    main()
