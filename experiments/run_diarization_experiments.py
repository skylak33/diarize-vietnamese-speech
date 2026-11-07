"""Controlled one-factor-at-a-time diarization ablations (not a Cartesian grid)."""
import argparse
import csv
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AHC_THRESHOLDS = [0.65, 0.70, 0.75, 0.80]
AHC_STABILITY_THRESHOLDS = [0.70, 0.72, 0.74, 0.75, 0.77]


def configurations(args):
    base = dict(method=args.method, ahc_threshold=args.threshold,
                count_estimator=getattr(args, 'count_estimator', None),
                vad_threshold=args.vad_threshold, window_sec=args.window_sec,
                step_sec=args.step_sec or args.window_sec / 2, speaker_mode=args.speaker_mode,
                min_segment_sec=getattr(args, 'min_segment_sec', 0.5),
                vad_min_speech_ms=getattr(args, 'vad_min_speech_ms', 250),
                vad_speech_pad_ms=getattr(args, 'vad_speech_pad_ms', 30),
                centroid_merge=getattr(args, 'centroid_merge', False),
                centroid_similarity_threshold=getattr(args, 'centroid_similarity_threshold', 0.80),
                max_small_cluster_size=getattr(args, 'max_small_cluster_size', 4),
                vad_min_silence_ms=getattr(args, 'vad_min_silence_ms', 100))
    if args.experiment == 'baseline':
        return [base]
    if args.experiment == 'clustering_count':
        return [
            {**base, 'method':'ahc', 'count_estimator':None, 'speaker_mode':'automatic'},
            {**base, 'method':'ahc', 'count_estimator':'adaptive_ahc',
             'speaker_mode':'automatic'},
            {**base, 'method':'ahc', 'count_estimator':'adaptive_ahc_robust',
             'speaker_mode':'automatic'},
            {**base, 'method':'spectral', 'count_estimator':None,
             'speaker_mode':'automatic'},
            {**base, 'method':'nme_sc', 'count_estimator':None,
             'speaker_mode':'automatic'},
        ]
    variations = {
        'clustering': ('method', ['ahc', 'spectral', 'nme_sc']),
        'ahc_threshold': ('ahc_threshold', AHC_THRESHOLDS),
        'ahc_stability': ('ahc_threshold', AHC_STABILITY_THRESHOLDS),
        'vad_threshold': ('vad_threshold', [0.3,0.4,0.5,0.6,0.7]),
        'vad_silence': ('vad_min_silence_ms', [100,200,300]),
        'window': ('window_sec', [1.5,2.0,3.0]),
        'speaker_count': ('speaker_mode', ['known','automatic']),
        'centroid_merge': ('centroid_similarity_threshold', [None,0.75,0.80,0.85]),
    }
    selected = ([name for name in variations if name != 'ahc_stability']
                if args.experiment == 'all' else [args.experiment])
    configs = []
    for experiment in selected:
        key, values = variations[experiment]
        for value in values:
            config = {**base, key:value}
            if experiment in {'ahc_threshold', 'ahc_stability'}:
                config.update(method='ahc', speaker_mode='automatic', count_estimator=None)
            elif experiment == 'clustering' or config['method'] != 'ahc':
                config['count_estimator'] = None
            if experiment == 'speaker_count' and value == 'known':
                config['count_estimator'] = None
            if experiment == 'centroid_merge':
                config.update(method='ahc', speaker_mode='automatic', centroid_merge=value is not None)
                if value is None:
                    config['centroid_similarity_threshold'] = base['centroid_similarity_threshold']
            if key == 'window_sec' and args.step_sec is None:
                config['step_sec'] = value / 2
            if not 0 < config['step_sec'] <= config['window_sec']:
                raise ValueError('Require 0 < step_sec <= window_sec in every configuration')
            if config not in configs:
                configs.append(config)
    return configs


def diagnostic_row(file_id, model, config, trace, metric):
    """Combine inference trace and reference metrics without inventing values."""
    before = trace.get('cluster_sizes_before_postmerge', trace.get('cluster_sizes', []))
    after = trace.get('cluster_sizes_after_postmerge', before)
    ahc = trace.get('clustering', {}).get('ahc', {})
    estimator = ahc.get('estimator')
    if estimator is None:
        estimator = {'spectral':'spectral_silhouette', 'nme_sc':'nme_eigengap'}.get(
            config['method'], 'fixed_threshold')
    return dict(
        recording_id=file_id, model_checkpoint=str(model), method=config['method'],
        count_estimator=estimator,
        estimated_speaker_count=trace.get('n_clusters_before_postprocess', 0),
        ahc_threshold=config['ahc_threshold'] if config['method'] == 'ahc' else None,
        ahc_merge_distances=ahc.get('distances', []), ahc_count_gaps=ahc.get('gaps', []),
        clustering_runtime_seconds=trace.get('clustering_runtime_seconds', 0.),
        window_sec=config['window_sec'], step_sec=config['step_sec'],
        vad_min_silence_ms=config.get('vad_min_silence_ms', 100),
        n_embedding_windows=trace.get('n_embedding_windows', 0),
        n_clusters_before_postmerge=trace.get('n_clusters_before_postmerge', 0),
        n_clusters_after_postmerge=trace.get('n_clusters_after_postmerge', 0),
        n_clusters_after_smoothing=trace.get('n_clusters_after_smoothing'),
        n_clusters_after_short_merge=trace.get('n_clusters_after_short_merge'),
        n_speakers_after_postprocess=trace.get('n_speakers_after_postprocess', 0),
        cluster_sizes_before_postmerge=before, cluster_sizes_after_postmerge=after,
        n_singleton_clusters=sum(size == 1 for size in after),
        n_small_clusters=sum(size <= 4 for size in after),
        n_ref=metric.get('N_ref'), n_pred=metric.get('N_pred'),
        speaker_count_absolute_error=metric.get('abs_error'), DER=metric.get('DER'),
        Miss=metric.get('miss'), FalseAlarm=metric.get('false_alarm'),
        SpeakerError=metric.get('speaker_error'),
        merge_attempts=trace.get('merge_attempts', 0),
        accepted_merges=trace.get('accepted_merges', 0),
        rejected_merges=trace.get('rejected_merges', 0),
        merge_similarity_scores=trace.get('merge_similarity_scores', []),
        merge_candidates=trace.get('merge_candidates', []),
        merge_rejection_reasons=trace.get('merge_rejection_reasons', {}),
    )


def centroid_merge_summary(rows):
    def summary(values):
        values = [value for value in values if value is not None]
        return (dict(count=len(values), minimum=min(values), maximum=max(values),
                     mean=statistics.fmean(values), median=statistics.median(values))
                if values else dict(count=0, minimum=None, maximum=None, mean=None, median=None))

    candidates = [candidate for row in rows for candidate in row['merge_candidates']]
    target_scores = [target['similarity'] for candidate in candidates
                     for target in candidate['target_similarities']]
    reasons = {}
    for row in rows:
        for reason, count in row['merge_rejection_reasons'].items():
            reasons[reason] = reasons.get(reason, 0) + count
    return dict(
        merge_attempts=sum(row['merge_attempts'] for row in rows),
        accepted_merges=sum(row['accepted_merges'] for row in rows),
        rejected_merges=sum(row['rejected_merges'] for row in rows),
        rejection_reasons=reasons,
        best_similarity=summary([candidate.get('best_similarity') for candidate in candidates]),
        all_target_similarities=summary(target_scores),
    )


def save_diagnostics(rows, folder):
    (folder / 'diagnostics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (folder / 'centroid_merge_summary.json').write_text(
        json.dumps(centroid_merge_summary(rows), indent=2), encoding='utf-8')
    with (folder / 'diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def oracle_count_diagnostics(results):
    """Build paired automatic-minus-oracle deltas from completed evaluations."""
    def delta(left, right):
        return None if left is None or right is None else left - right

    automatic, oracle = results['automatic'], results['known']
    files = sorted(automatic.keys() - {'__total__'})
    rows = []
    for file_id in files:
        auto, known = automatic[file_id], oracle[file_id]
        rows.append(dict(
            recording_id=file_id,
            reference_speakers=auto['N_ref'],
            automatic_speakers=auto['N_pred'],
            oracle_requested_speakers=known['N_ref'],
            oracle_count_output_speakers=known['N_pred'],
            speaker_count_delta=auto['N_pred'] - known['N_pred'],
            automatic_DER=auto['DER'], oracle_count_DER=known['DER'],
            DER_delta=delta(auto['DER'], known['DER']),
            automatic_speaker_error=auto['speaker_error'],
            oracle_count_speaker_error=known['speaker_error'],
            speaker_error_delta=delta(auto['speaker_error'], known['speaker_error']),
            automatic_speaker_error_seconds=auto['speaker_error_seconds'],
            oracle_count_speaker_error_seconds=known['speaker_error_seconds'],
            speaker_error_seconds_delta=delta(auto['speaker_error_seconds'],
                                              known['speaker_error_seconds']),
            automatic_count_absolute_error=auto['abs_error'],
            oracle_count_absolute_error=known['abs_error'],
            count_absolute_error_delta=auto['abs_error'] - known['abs_error'],
        ))
    auto_total, known_total = automatic['__total__'], oracle['__total__']
    der_deltas = [row['DER_delta'] for row in rows]
    return dict(
        condition='oracle speaker count diagnostic; not an automatic diarization result',
        delta_definition='automatic minus oracle-count',
        aggregate=dict(
            automatic_DER=auto_total['DER'], oracle_count_DER=known_total['DER'],
            DER_delta=delta(auto_total['DER'], known_total['DER']),
            automatic_speaker_error=auto_total['speaker_error'],
            oracle_count_speaker_error=known_total['speaker_error'],
            speaker_error_delta=delta(auto_total['speaker_error'], known_total['speaker_error']),
            automatic_average_speakers=auto_total['average_predicted_speakers'],
            oracle_count_average_speakers=known_total['average_predicted_speakers'],
            average_speaker_count_delta=(auto_total['average_predicted_speakers']
                                         - known_total['average_predicted_speakers']),
            automatic_count_MAE=auto_total['speaker_count_mae'],
            oracle_count_MAE=known_total['speaker_count_mae'],
            count_MAE_delta=(auto_total['speaker_count_mae']
                             - known_total['speaker_count_mae']),
            automatic_count_accuracy=auto_total['speaker_count_accuracy'],
            oracle_count_accuracy=known_total['speaker_count_accuracy'],
            automatic_speaker_error_seconds=auto_total['speaker_error_seconds'],
            oracle_count_speaker_error_seconds=known_total['speaker_error_seconds'],
            speaker_error_seconds_delta=(auto_total['speaker_error_seconds']
                                         - known_total['speaker_error_seconds']),
            recordings_improved=sum(value > 1e-9 for value in der_deltas),
            recordings_worsened=sum(value < -1e-9 for value in der_deltas),
            recordings_unchanged=sum(abs(value) <= 1e-9 for value in der_deltas),
        ),
        recordings=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('audio_dir','ref_rttm_dir','output_dir'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--model', default='exps/exp1/model/model_0034.model')
    parser.add_argument('--experiment', choices=['baseline','clustering','clustering_count','ahc_threshold','ahc_stability','vad_threshold','vad_silence','window','speaker_count','centroid_merge','all'], default='clustering')
    parser.add_argument('--method', choices=['ahc','spectral','nme_sc'], default='ahc')
    parser.add_argument('--count_estimator', choices=['adaptive_ahc', 'adaptive_ahc_robust'])
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--vad_threshold', type=float, default=0.5)
    parser.add_argument('--vad_min_silence_ms', type=int, default=100)
    parser.add_argument('--vad_min_speech_ms', type=int, default=250)
    parser.add_argument('--vad_speech_pad_ms', type=int, default=30)
    parser.add_argument('--window_sec', type=float, default=1.5)
    parser.add_argument('--step_sec', type=float)
    parser.add_argument('--min_segment_sec', type=float, default=0.5)
    parser.add_argument('--speaker_mode', choices=['known','automatic'], default='automatic')
    parser.add_argument('--max_speakers', type=int, default=10)
    parser.add_argument('--centroid_merge', action='store_true')
    parser.add_argument('--centroid_similarity_threshold', type=float, default=0.80)
    parser.add_argument('--max_small_cluster_size', type=int, default=4)
    parser.add_argument('--collar', type=float, default=0.25)
    parser.add_argument('--skip_overlap', action='store_true')
    parser.add_argument('--uem_dir', help='Optional directory of official UEM files')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--recording_ids', nargs='+')
    parser.add_argument('--diagnostic_output_dir')
    parser.add_argument('--diagnostic_ids', nargs='+')
    parser.add_argument('--dry_run', action='store_true', help='Print configurations without loading audio/model')
    parser.add_argument('--max_configs', type=int, default=32, help='Abort if plan exceeds limit')
    args = parser.parse_args()
    configs = configurations(args)
    if len(configs) > args.max_configs:
        parser.error(f'{len(configs)} configurations exceed --max_configs={args.max_configs}')
    print(json.dumps(configs, indent=2))
    print(f'Configurations: {len(configs)}')
    if args.dry_run:
        return
    from diarize.pipeline import diarize
    from utils.model import load_model
    from utils.seed import set_seed
    from utils.provenance import sha256, environment
    from eval.der import load_rttm, load_uem, parse_rttm, speaker_count, evaluate_annotations, save_results
    import soundfile as sf
    refs = load_rttm(args.ref_rttm_dir)
    uem_data = load_uem(args.uem_dir) if args.uem_dir else None
    audios = {}
    for path in sorted(Path(args.audio_dir).rglob('*')):
        if path.suffix.lower() not in {'.wav','.flac','.ogg'}:
            continue
        if path.stem in audios:
            raise ValueError(f'Duplicate audio recording ID: {path.stem}')
        audios[path.stem] = path
    if args.recording_ids:
        requested = set(args.recording_ids)
        missing = requested - refs.keys() | requested - audios.keys()
        if missing:
            raise ValueError(f'Requested recording IDs are unavailable: {sorted(missing)}')
        refs = {key: refs[key] for key in sorted(requested)}
        audios = {key: audios[key] for key in sorted(requested)}
    if refs.keys() != audios.keys():
        raise ValueError(f'Audio/reference IDs differ; missing audio={refs.keys()-audios.keys()}, missing reference={audios.keys()-refs.keys()}')
    if uem_data is not None and uem_data.keys() != refs.keys():
        raise ValueError(f'UEM/reference IDs differ; missing UEM={refs.keys()-uem_data.keys()}, extra UEM={uem_data.keys()-refs.keys()}')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'experiment_manifest.json').exists():
        raise FileExistsError('Output already contains an experiment; choose a new --output_dir')
    uem_files = sorted(Path(args.uem_dir).rglob('*.uem')) if args.uem_dir else []
    encoder_source = Path(__file__).resolve().parents[1] / 'models' / 'ecapa_tdnn.py'
    metadata = dict(status='running', arguments=vars(args), configurations=configs, environment=environment(),
                    model_sha256=sha256(args.model),
                    encoder_source=str(encoder_source), encoder_source_sha256=sha256(encoder_source),
                    audio_sha256={k:sha256(v) for k,v in audios.items()},
                    reference_sha256={k:sha256(Path(args.ref_rttm_dir) / f'{k}.rttm') for k in refs},
                    evaluation=dict(mode='external_uem' if uem_data is not None else 'full_audio',
                                    collar=args.collar, skip_overlap=args.skip_overlap,
                                    uem_sha256={str(p):sha256(p) for p in uem_files}))
    (out / 'experiment_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    durations = {key:sf.info(path).duration for key,path in audios.items()}
    set_seed(args.seed)
    encoder = load_model(args.model, args.device)
    embedding_cache = {}
    fields = ['method','count_estimator','ahc_threshold','vad_threshold','vad_min_speech_ms','vad_min_silence_ms',
              'vad_speech_pad_ms','window_sec','step_sec','min_segment_sec','speaker_mode',
              'centroid_merge','centroid_similarity_threshold','max_small_cluster_size',
              'DER','Miss','FalseAlarm','SpeakerError','SpeakerCountAccuracy','SpeakerCountMAE',
              'OverclusteringRate','UnderclusteringRate','AveragePredictedSpeakers',
              'AverageReferenceSpeakers','EmbeddingCacheHits','ElapsedSeconds',
              'ClusteringSeconds','PeakPythonMemoryMB','config_id']
    speaker_count_results = {}
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, config in enumerate(configs):
            started = time.perf_counter()
            tracemalloc.start()
            config_id = f'config_{index:03d}'
            baseline = args.experiment == 'baseline'
            folder = out if baseline else out / config_id
            if not baseline:
                folder.mkdir()
            (folder / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            hypotheses = {}
            traces = {}
            runtimes = {}
            for file_id, audio in audios.items():
                recording_started = time.perf_counter()
                n = speaker_count(refs[file_id]) if config['speaker_mode'] == 'known' else None
                # Silence has zero reference speakers; an empty hypothesis is the oracle-count case.
                rttm = folder / ('hypothesis_rttm' if baseline else 'rttm') / (file_id + '.rttm')
                if n == 0:
                    rttm.parent.mkdir(exist_ok=True)
                    rttm.write_text('', encoding='utf-8')
                    traces[file_id] = dict(n_embedding_windows=0,
                        n_clusters_before_postmerge=0, n_clusters_after_postmerge=0,
                        n_clusters_after_smoothing=0, n_clusters_after_short_merge=0,
                        n_speakers_after_postprocess=0, cluster_sizes_before_postmerge=[],
                        cluster_sizes_after_postmerge=[], merge_attempts=0, accepted_merges=0,
                        rejected_merges=0, merge_similarity_scores=[], merge_candidates=[],
                        merge_rejection_reasons={})
                else:
                    traces[file_id] = {}
                    diagnostic_root = None
                    if (args.diagnostic_output_dir
                            and (not args.diagnostic_ids or file_id in args.diagnostic_ids)):
                        diagnostic_root = str(
                            Path(args.diagnostic_output_dir) / config_id
                            if len(configs) > 1 else Path(args.diagnostic_output_dir))
                    diarize(str(audio), args.model, n_speakers=n, method=config['method'],
                            count_estimator=config['count_estimator'],
                            threshold=config['ahc_threshold'], vad_threshold=config['vad_threshold'],
                            window_sec=config['window_sec'], step_sec=config['step_sec'],
                            output_rttm=str(rttm), seed=args.seed, device=args.device,
                            max_speakers=args.max_speakers, encoder=encoder,
                            diagnostics=traces[file_id], centroid_merge=config['centroid_merge'],
                            centroid_similarity_threshold=config['centroid_similarity_threshold'],
                            max_small_cluster_size=config['max_small_cluster_size'],
                            vad_min_silence_ms=config['vad_min_silence_ms'],
                            vad_min_speech_ms=config['vad_min_speech_ms'],
                            vad_speech_pad_ms=config['vad_speech_pad_ms'],
                            min_segment_sec=config['min_segment_sec'], embedding_cache=embedding_cache,
                            diagnostic_output_dir=diagnostic_root,
                            evaluation_protocol='published_reference_full_audio',
                            reference_source=str(Path(args.ref_rttm_dir).resolve()))
                hypotheses[file_id] = parse_rttm(str(rttm)).get(file_id, [])
                runtimes[file_id] = time.perf_counter() - recording_started
            result = evaluate_annotations(refs, hypotheses, args.collar, args.skip_overlap,
                                          durations, uem_data)
            if args.experiment == 'speaker_count':
                speaker_count_results[config['speaker_mode']] = result
            save_results(result, folder / 'der.json', folder / 'per_file.csv')
            diagnostic_rows = [diagnostic_row(file_id, args.model, config, traces[file_id], result[file_id])
                               for file_id in sorted(audios)]
            save_diagnostics(diagnostic_rows, folder)
            total = result['__total__']
            if baseline:
                recording_rows = []
                for file_id in sorted(audios):
                    metric = result[file_id]
                    recording_rows.append(dict(recording_id=file_id,
                        audio_sha256=metadata['audio_sha256'][file_id],
                        reference_sha256=metadata['reference_sha256'][file_id],
                        rttm_sha256=sha256(folder / 'hypothesis_rttm' / f'{file_id}.rttm'),
                        effective_configuration=config, speaker_count=metric['N_pred'],
                        metrics=metric, runtime_seconds=runtimes[file_id], failure=None))
                (folder / 'recording_results.json').write_text(
                    json.dumps(recording_rows, indent=2), encoding='utf-8')
                flat_rows = [{**{key:value for key,value in row.items()
                               if key not in {'effective_configuration','metrics'}}, **row['metrics']}
                             for row in recording_rows]
                with (folder / 'recording_results.csv').open('w', newline='', encoding='utf-8') as result_handle:
                    result_writer = csv.DictWriter(result_handle, fieldnames=list(flat_rows[0]))
                    result_writer.writeheader(); result_writer.writerows(flat_rows)
                (folder / 'corpus_results.json').write_text(
                    json.dumps(total, indent=2), encoding='utf-8')
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            writer.writerow({**config, 'config_id':config_id, 'DER':total['DER'],
                'Miss':total['miss'], 'FalseAlarm':total['false_alarm'], 'SpeakerError':total['speaker_error'],
                'SpeakerCountAccuracy':total['speaker_count_accuracy'], 'SpeakerCountMAE':total['speaker_count_mae'],
                'OverclusteringRate':total['overclustering_rate'],
                'UnderclusteringRate':total['underclustering_rate'],
                'AveragePredictedSpeakers':total['average_predicted_speakers'],
                'AverageReferenceSpeakers':total['average_reference_speakers'],
                'EmbeddingCacheHits':sum(trace.get('embedding_cache_hit', False)
                                         for trace in traces.values()),
                'ElapsedSeconds':time.perf_counter() - started,
                'ClusteringSeconds':sum(trace.get('clustering_runtime_seconds', 0.)
                                        for trace in traces.values()),
                'PeakPythonMemoryMB':peak_memory / (1024 * 1024)})
            handle.flush()
            print(f"{config_id}: DER={total['DER']}")
    if args.experiment == 'speaker_count':
        (out / 'oracle_count_diagnostics.json').write_text(
            json.dumps(oracle_count_diagnostics(speaker_count_results), indent=2),
            encoding='utf-8')
    metadata.update(status='complete', files_evaluated=len(audios), failures=[])
    for manifest_name in ('experiment_manifest.json', 'manifest.json'):
        (out / manifest_name).write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
