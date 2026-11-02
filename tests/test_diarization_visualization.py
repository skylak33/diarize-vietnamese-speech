import json
import sys

import numpy as np
import soundfile as sf

from experiments.visualize_diarization_errors import main, plot_reference_review, plot_timeline


def recording():
    return {
        'recording_id': 'sample',
        'speaker_count_status': 'correct_count',
        'temporal': {
            'effective_uem': [{'start': 0, 'end': 3}],
            'reference_intervals': [
                {'start': 0, 'end': 1.5, 'speaker': 'A'},
                {'start': 1.5, 'end': 3, 'speaker': 'B'},
            ],
            'mapped_hypothesis_intervals': [
                {'start': 0, 'end': 2, 'speaker': 'X', 'mapped_speaker': 'A'},
                {'start': 2, 'end': 3, 'speaker': 'Y', 'mapped_speaker': 'B'},
            ],
            'error_intervals': [{
                'start': 1.5, 'end': 2, 'duration': .5, 'kind': 'speaker_confusion',
                'speaker_seconds': .5, 'speaker_count': 1, 'matched_speakers': [],
                'reference_speakers': ['B'], 'hypothesis_speakers': ['X'],
                'mapped_hypothesis_speakers': ['A'],
            }],
        },
    }


def test_timeline_writes_png_and_svg(tmp_path):
    for suffix in ('png', 'svg'):
        output = tmp_path / f'timeline.{suffix}'
        plot_timeline(recording(), output, 'collar=0.25')
        assert output.stat().st_size > 100


def test_reference_review_writes_waveform_overlay(tmp_path):
    audio = tmp_path / 'sample.wav'
    sf.write(audio, np.sin(np.linspace(0, 40, 1600)), 1600)
    output = tmp_path / 'review.png'
    plot_reference_review(
        'sample', audio, [(0, .4, 'A'), (.6, 1, 'B')], [(0, .9)],
        [(0, .2)], [.6], output)
    assert output.stat().st_size > 100


def test_visualization_payload_is_json_serializable():
    json.dumps(recording())


def test_review_manifest_cli(tmp_path, monkeypatch):
    audio = tmp_path / 'sample.wav'
    sf.write(audio, np.zeros(1600), 1600)
    manifest = tmp_path / 'review.json'
    manifest.write_text(json.dumps([{
        'recording_id': 'sample', 'audio_path': 'sample.wav',
        'reference': [[0, 1, 'A']], 'vad': [[0, .8]],
        'suspected_unannotated': [[0, .1]], 'speaker_changes': [.5],
    }]), encoding='utf-8')
    output = tmp_path / 'plots'
    monkeypatch.setattr(sys, 'argv', [
        'visualize', '--review_manifest', str(manifest), '--output_dir', str(output)])
    main()
    assert (output / 'sample.png').stat().st_size > 100
