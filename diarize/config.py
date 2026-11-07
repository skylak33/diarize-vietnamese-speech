"""Validated configuration for diarization inference."""
import copy
import math
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    'sample_rate': 16000,
    'embedding': {'window_sec': 1.5, 'step_sec': 0.75, 'min_segment_sec': 0.5},
    'vad': {'threshold': 0.5, 'min_speech_ms': 250, 'min_silence_ms': 100,
            'speech_pad_ms': 30},
    'clustering': {'method': 'ahc', 'count_estimator': 'backend_default',
                   'threshold': 0.65, 'max_speakers': 10},
    'postprocessing': {'centroid_merge': {'enabled': False, 'similarity_threshold': 0.80,
                                          'max_small_cluster_size': 4,
                                          'similarity_margin': 0.05}},
    'evaluation': {'collar': 0.25, 'skip_overlap': False},
    'seed': 42,
    'device': 'auto',
}


def _merge_known(base, update, path=''):
    if not isinstance(update, dict):
        raise ValueError(f'{path or "configuration"} must be a mapping')
    for key, value in update.items():
        name = f'{path}.{key}' if path else key
        if key not in base:
            raise ValueError(f'Unknown configuration key: {name}')
        if isinstance(base[key], dict):
            _merge_known(base[key], value, name)
        else:
            base[key] = value


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def validate_config(config):
    _integer(config['sample_rate'], 'sample_rate')
    if config['sample_rate'] != 16000:
        raise ValueError('sample_rate must be 16000 for the current encoder')
    embedding = config['embedding']
    for key in ('window_sec', 'step_sec', 'min_segment_sec'):
        _number(embedding[key], f'embedding.{key}', 0)
        if embedding[key] == 0:
            raise ValueError(f'embedding.{key} must be positive')
    if embedding['step_sec'] > embedding['window_sec']:
        raise ValueError('embedding.step_sec must not exceed embedding.window_sec')
    vad = config['vad']
    _number(vad['threshold'], 'vad.threshold', 0, 1)
    _integer(vad['min_speech_ms'], 'vad.min_speech_ms')
    _integer(vad['min_silence_ms'], 'vad.min_silence_ms')
    _integer(vad['speech_pad_ms'], 'vad.speech_pad_ms', 0)
    clustering = config['clustering']
    if clustering['method'] not in {'ahc', 'spectral', 'nme_sc'}:
        raise ValueError('clustering.method must be ahc, spectral, or nme_sc')
    if clustering['count_estimator'] not in {
            'backend_default', 'adaptive_ahc', 'adaptive_ahc_robust'}:
        raise ValueError(
            'clustering.count_estimator must be backend_default, adaptive_ahc, '
            'or adaptive_ahc_robust')
    if (clustering['count_estimator'] in {'adaptive_ahc', 'adaptive_ahc_robust'}
            and clustering['method'] != 'ahc'):
        raise ValueError('adaptive AHC count estimation requires clustering.method=ahc')
    _number(clustering['threshold'], 'clustering.threshold', 0, 2)
    _integer(clustering['max_speakers'], 'clustering.max_speakers')
    merge = config['postprocessing']['centroid_merge']
    if not isinstance(merge['enabled'], bool):
        raise ValueError('postprocessing.centroid_merge.enabled must be boolean')
    _number(merge['similarity_threshold'],
            'postprocessing.centroid_merge.similarity_threshold', -1, 1)
    _integer(merge['max_small_cluster_size'],
             'postprocessing.centroid_merge.max_small_cluster_size')
    _number(merge['similarity_margin'], 'postprocessing.centroid_merge.similarity_margin', 0, 2)
    evaluation = config['evaluation']
    _number(evaluation['collar'], 'evaluation.collar', 0)
    if not isinstance(evaluation['skip_overlap'], bool):
        raise ValueError('evaluation.skip_overlap must be boolean')
    if isinstance(config['seed'], bool) or not isinstance(config['seed'], int):
        raise ValueError('seed must be an integer')
    if config['device'] not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('device must be auto, cpu, or cuda')
    return config


def load_config(path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        _merge_known(config, {} if loaded is None else loaded)
    return validate_config(config)


def set_config_value(config, dotted_path, value):
    target = config
    parts = dotted_path.split('.')
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def pipeline_kwargs(config):
    embedding, vad = config['embedding'], config['vad']
    clustering = config['clustering']
    merge = config['postprocessing']['centroid_merge']
    return dict(method=clustering['method'],
                count_estimator=(None if clustering['count_estimator'] == 'backend_default'
                                 else clustering['count_estimator']),
                threshold=clustering['threshold'],
                max_speakers=clustering['max_speakers'], window_sec=embedding['window_sec'],
                step_sec=embedding['step_sec'], min_segment_sec=embedding['min_segment_sec'],
                vad_threshold=vad['threshold'], vad_min_speech_ms=vad['min_speech_ms'],
                vad_min_silence_ms=vad['min_silence_ms'], vad_speech_pad_ms=vad['speech_pad_ms'],
                centroid_merge=merge['enabled'],
                centroid_similarity_threshold=merge['similarity_threshold'],
                max_small_cluster_size=merge['max_small_cluster_size'],
                centroid_similarity_margin=merge['similarity_margin'], seed=config['seed'],
                device=config['device'])
