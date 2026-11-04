from pathlib import Path

import pytest

from diarize.config import DEFAULT_CONFIG, load_config
from diarize.pipeline import parse_cli_args


def test_builtin_defaults_preserve_cli_behavior():
    args, config = parse_cli_args(['audio.wav', '--model', 'model.model'])
    assert args.method == 'ahc' and args.threshold == .65
    assert args.count_estimator is None
    assert args.window_sec == 1.5 and args.step_sec == .75
    assert args.vad_min_silence_ms == 100 and args.centroid_merge is False
    assert config == DEFAULT_CONFIG


def test_cli_propagates_diagnostic_options_without_touching_pipeline_config():
    args, config = parse_cli_args([
        'audio.wav', '--model', 'model.model', '--diagnostic_output_dir', 'diagnostics',
        '--diagnostic_counterfactuals', '--evaluation_protocol', 'uem-v1',
        '--reference_source', 'manual_reference',
    ])
    assert args.diagnostic_output_dir == 'diagnostics'
    assert args.diagnostic_counterfactuals is True
    assert args.evaluation_protocol == 'uem-v1'
    assert args.reference_source == 'manual_reference'
    assert config == DEFAULT_CONFIG


def test_yaml_is_loaded_and_only_explicit_cli_values_override(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('''
embedding:
  window_sec: 2.0
  step_sec: 1.0
vad:
  min_silence_ms: 200
clustering:
  method: spectral
  threshold: 0.7
postprocessing:
  centroid_merge:
    enabled: true
    similarity_threshold: 0.85
''')
    args, _ = parse_cli_args(['audio.wav', '--model', 'model.model', '--config', str(path),
                              '--method', 'ahc', '--threshold', '0.75',
                              '--no_centroid_merge'])
    assert args.method == 'ahc' and args.threshold == .75
    assert args.window_sec == 2. and args.step_sec == 1.
    assert args.vad_min_silence_ms == 200
    assert args.centroid_merge is False and args.centroid_similarity_threshold == .85


def test_repository_default_yaml_matches_builtins():
    path = Path(__file__).resolve().parents[1] / 'configs' / 'diarization_default.yaml'
    assert load_config(path) == DEFAULT_CONFIG


def test_adaptive_count_estimator_is_explicit_and_opt_in(tmp_path):
    path = tmp_path / 'adaptive.yaml'
    path.write_text('clustering:\n  count_estimator: adaptive_ahc\n')

    args, _ = parse_cli_args(['audio.wav', '--model', 'model.model', '--config', str(path)])
    assert args.method == 'ahc' and args.count_estimator == 'adaptive_ahc'


@pytest.mark.parametrize('content,error', [
    ('clustering:\n  thresholdd: 0.7\n', 'Unknown configuration key: clustering.thresholdd'),
    ('clustering:\n  method: spectral\n  count_estimator: adaptive_ahc\n', 'requires clustering.method=ahc'),
    ('embedding:\n  window_sec: 1\n  step_sec: 2\n', 'step_sec'),
    ('vad:\n  min_silence_ms: 0\n', 'min_silence_ms'),
    ('postprocessing:\n  centroid_merge:\n    enabled: yes-please\n', 'enabled'),
])
def test_invalid_config_is_actionable(tmp_path, content, error):
    path = tmp_path / 'bad.yaml'
    path.write_text(content)
    with pytest.raises(ValueError, match=error):
        load_config(path)
