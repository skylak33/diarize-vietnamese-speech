"""Cache local synthetic frontends once, then compare frozen clustering methods."""

import argparse
import contextlib
import csv
import io
import json
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.cluster import (cluster_embeddings, merge_clusters_confidence_aware,
                             merge_similar_clusters, smooth_labels, merge_short_segments,
                             _confidence_merge_parameters)
from diarize.pipeline import (extract_embeddings_for_segments, partition_overlapping_windows,
                              merge_consecutive_segments)
from eval.der import evaluate_annotations, parse_rttm
from experiments.analyze_clustering_stability import angular_noise
from experiments.run_synthetic_diarization_benchmark import (
    load_conversations, _reference, assign_reference_labels, partition_metrics)
from utils.provenance import sha256, environment


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_HASH = '58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325'
MODERATE = dict(centroid_similarity_threshold=.65, cross_median_threshold=.25,
                cross_lower_threshold=.10)
METHODS = {
    'fixed_065': dict(threshold=.65),
    'fixed_075': dict(threshold=.75),
    'centroid': dict(threshold=.75, merge='centroid'),
    'legacy': dict(threshold=.75, count_estimator='adaptive_ahc'),
    'robust': dict(threshold=.75, count_estimator='adaptive_ahc_robust'),
    'spectral': dict(method='spectral'),
    'nme_sc': dict(method='nme_sc'),
    'confidence_strict': dict(threshold=.75, merge='confidence', config={}),
    'confidence_moderate': dict(threshold=.75, merge='confidence', config=MODERATE),
}
SELECTION = {
    'primary_mode': 'full_pipeline',
    'eligibility': 'pooled DER <= fixed_075 + 0.25 pp; purity loss <= 0.005; '
                   'fraction of recordings with DER regression > 10 pp <= 0.01',
    'ranking': ['DER', 'SpeakerError', 'count_MAE', '-ARI', '-purity',
                'fragmentation', '-count_accuracy', 'method_name'],
    'fallback': 'If neither candidate qualifies, retain the lower-DER candidate only for '
                'exploratory diagnostics and do not recommend deployment.',
    'independence': 'Previously inspected validation/holdout are not independent tests.',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def audit_splits(records):
    """Check actual speaker, source recording, and source-content separation."""
    groups = defaultdict(lambda: defaultdict(set))
    for record in records:
        split = record['split']
        groups['speakers'][split].update(record['speaker_ids'])
        for turn in record['turns']:
            for kind, field in (('recordings', 'source_recording_id'), ('hashes', 'source_hash')):
                if not turn.get(field):
                    raise ValueError(f"{record['conversation_id']}: missing {field}")
                groups[kind][split].add(turn[field])
    for kind, splits in groups.items():
        names = sorted(splits)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if splits[left] & splits[right]:
                    raise ValueError(f'{kind} leakage between {left} and {right}')
    return {kind: {split: sorted(values) for split, values in splits.items()}
            for kind, splits in groups.items()}


def prepare(args):
    """Extract only missing development embeddings; never download source data."""
    import torch
    from diarize.vad import load_audio, get_speech_segments_from_audio
    from utils.model import load_model
    from utils.seed import set_seed

    records = load_conversations(args.conversations)
    audit = audit_splits(records)
    checkpoint = ROOT / 'exps/exp1/model/model_0034.model'
    if sha256(checkpoint) != CHECKPOINT_HASH:
        raise ValueError('Frozen checkpoint hash mismatch')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'split_audit.json', audit)
    historical = read_json(args.historical_manifest)
    if historical['checkpoint_sha256'] != CHECKPOINT_HASH:
        raise ValueError('Historical checkpoint mismatch')
    encoder_hash = sha256(ROOT / 'models/ecapa_tdnn.py')
    if encoder_hash not in historical['code_sha256'].values():
        raise ValueError('Historical encoder mismatch')
    selected = [record for record in records if record['split'] == 'development']
    torch.set_num_threads(2)
    set_seed(42)
    encoder = None
    entries = []
    run_environment = environment()
    for index, record in enumerate(selected):
        rid = record['conversation_id']
        provenance = dict(recording_id=rid, checkpoint_sha256=CHECKPOINT_HASH,
            encoder_source_sha256=encoder_hash, audio_sha256=sha256(record['audio_path']),
            reference_sha256=sha256(record['rttm_path']), metadata_sha256=sha256(record['metadata_path']),
            mode='full_pipeline', window_sec=1.5, step_sec=.75, min_segment_sec=.5,
            vad_threshold=.5, vad_min_silence_ms=100, vad_min_speech_ms=250, vad_speech_pad_ms=30,
            seed=42)
        for field, manifest_key in (('audio_sha256', 'audio_sha256'),
                                     ('reference_sha256', 'reference_sha256'),
                                     ('metadata_sha256', 'conversation_sha256')):
            if provenance[field] != historical[manifest_key][rid]:
                raise ValueError(f'{rid}: historical {field} mismatch')
        archive = output / f'{rid}.npz'
        sidecar = archive.with_suffix('.json')
        if archive.exists():
            cached = read_json(sidecar)
            if cached['provenance'] != provenance or cached['archive_sha256'] != sha256(archive):
                raise ValueError(f'{rid}: incompatible cache')
        else:
            if encoder is None:
                encoder = load_model(str(checkpoint), 'cpu')
            started = time.perf_counter()
            audio, sr = load_audio(str(record['audio_path']))
            regions = get_speech_segments_from_audio(audio, sr)
            with contextlib.redirect_stdout(io.StringIO()):
                windows, embeddings = extract_embeddings_for_segments(encoder, audio, regions)
            if not len(embeddings):
                raise ValueError(f'{rid}: no embedding windows')
            np.savez_compressed(archive, embeddings=embeddings, windows=np.asarray(windows))
            write_json(sidecar, dict(provenance=provenance, archive_sha256=sha256(archive),
                frontend_runtime_seconds=time.perf_counter() - started))
        entries.append(dict(recording_id=rid, archive=str(archive.resolve()),
            archive_sha256=sha256(archive), reference=str(record['rttm_path']),
            duration=record['duration_seconds'], speaker_ids=record['speaker_ids'],
            source_hashes=sorted({turn['source_hash'] for turn in record['turns']}),
            provenance=provenance))
        if (index + 1) % 10 == 0 or index == 0:
            print(f'Cached {index + 1}/{len(selected)}', flush=True)
        write_json(output / 'manifest.json', dict(status='complete' if len(entries) == len(selected)
            else 'running', entries=entries, environment=run_environment,
            historical_manifest_sha256=sha256(args.historical_manifest),
            selection_protocol=SELECTION))


def infer(embeddings, windows, method):
    options = METHODS[method]
    with contextlib.redirect_stdout(io.StringIO()):
        initial = cluster_embeddings(embeddings, method=options.get('method', 'ahc'),
            threshold=options.get('threshold', .75),
            count_estimator=options.get('count_estimator'), seed=42, max_speakers=10)
    labels, trace = initial.copy(), {}
    if options.get('merge') == 'centroid':
        labels, trace = merge_similar_clusters(embeddings, labels, return_diagnostics=True)
    elif options.get('merge') == 'confidence':
        labels, trace = merge_clusters_confidence_aware(embeddings, labels, windows,
            options['config'], return_diagnostics=True)
    smoothed = smooth_labels(windows, labels.copy(), window=3)
    final_windows, final_labels = merge_short_segments(
        partition_overlapping_windows(windows), smoothed.copy(), min_duration=.3)
    final_windows, final_labels = merge_consecutive_segments(final_windows, final_labels, gap_threshold=0.)
    return {'initial': (windows, initial), 'merged': (windows, labels),
            'smoothed': (windows, smoothed), 'final': (final_windows, final_labels)}, trace


def stage_metrics(rid, windows, labels, reference, duration, partition_windows, partition_labels):
    hypothesis = [(float(start), float(end), int(label))
                  for (start, end), label in zip(windows, labels)]
    metric = evaluate_annotations({rid: reference}, {rid: hypothesis},
        collar=.25, skip_overlap=False, durations={rid: duration})[rid]
    return dict(DER=metric['DER'], SpeakerError=metric['speaker_error'],
        duration=metric['duration'], miss_seconds=metric['miss_seconds'],
        false_alarm_seconds=metric['false_alarm_seconds'],
        speaker_error_seconds=metric['speaker_error_seconds'],
        count=int(metric['N_pred']), reference_count=int(metric['N_ref']),
        count_correct=bool(metric['exact_match']), count_error=int(metric['abs_error']),
        **partition_metrics(partition_windows, partition_labels, reference))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['stage'], 'all')].append(row)
        groups[(row['method'], row['stage'], str(row['reference_count']))].append(row)
    result = []
    for (method, stage, k), values in sorted(groups.items()):
        duration = sum(row['duration'] for row in values)
        weight = sum(row['compatible_windows'] for row in values)
        result.append(dict(method=method, stage=stage, reference_K=k, recordings=len(values),
            DER=100 * sum(row['miss_seconds'] + row['false_alarm_seconds'] +
                          row['speaker_error_seconds'] for row in values) / duration,
            SpeakerError=100 * sum(row['speaker_error_seconds'] for row in values) / duration,
            count_accuracy=float(np.mean([row['count_correct'] for row in values])),
            count_MAE=float(np.mean([row['count_error'] for row in values])),
            **{key: sum(row[key] * row['compatible_windows'] for row in values
                        if row[key] is not None) / weight if weight else None
               for key in ('ARI', 'purity', 'fragmentation')},
            runtime_seconds=sum(row['runtime_seconds'] for row in values)))
    return result


def reference_merge_audit(initial, trace, windows, reference):
    indexes, refs = assign_reference_labels(windows, reference)
    dominant = lambda labels, cluster: Counter(np.asarray(refs)[labels[indexes] == cluster]).most_common(1)
    labels = initial.copy()
    result = []
    for decision in trace.get('merge_decisions', []):
        if not decision['accepted']:
            continue
        a, b = decision['cluster_a'], decision['cluster_b']
        before = partition_metrics(windows, labels, reference)
        da, db = dominant(labels, a), dominant(labels, b)
        labels[labels == b] = a
        after = partition_metrics(windows, labels, reference)
        result.append(dict(merge_step=decision['merge_step'], cluster_a=a, cluster_b=b,
            cross_speaker=(da[0][0] != db[0][0]) if da and db else None,
            purity_before=before['purity'], purity_after=after['purity'],
            ARI_before=before['ARI'], ARI_after=after['ARI']))
    return result


def evaluate(args):
    manifest = read_json(args.cache)
    if manifest['status'] != 'complete':
        raise ValueError('Cache extraction is incomplete')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise FileExistsError(f'Experiment output already exists: {output}')
    methods = list(METHODS)
    if args.frozen:
        frozen = read_json(args.frozen)
        selected = frozen['method']
        if METHODS[selected] != frozen['configuration']:
            raise ValueError('Frozen configuration changed')
        if sha256(ROOT / 'diarize/cluster.py') != frozen['cluster_source_sha256']:
            raise ValueError('Frozen clustering source changed')
        methods = ['fixed_075', 'centroid', selected]
    write_json(output / 'protocol.json', dict(selection=SELECTION,
        methods={name: METHODS[name] for name in methods}, cache_sha256=sha256(args.cache),
        environment=environment(), checkpoint_sha256=CHECKPOINT_HASH,
        encoder_source_sha256=sha256(ROOT / 'models/ecapa_tdnn.py'),
        cluster_source_sha256=sha256(ROOT / 'diarize/cluster.py'),
        evaluation=dict(collar=.25, skip_overlap=False),
        partition_metrics='equal unambiguous embedding windows; final labels projected by maximum overlap',
        evidence='development' if not args.frozen else 'exploratory reused-test evaluation'))
    rows, decisions, audit, stability = [], [], [], []
    for index, entry in enumerate(manifest['entries']):
        rid = entry['recording_id']
        if sha256(entry['archive']) != entry['archive_sha256']:
            raise ValueError(f'{rid}: cache archive changed')
        if sha256(entry['reference']) != entry['provenance']['reference_sha256']:
            raise ValueError(f'{rid}: reference changed')
        with np.load(entry['archive'], allow_pickle=False) as archive:
            embeddings = archive['embeddings']
            windows = (archive['windows'] if 'windows' in archive else
                       np.column_stack((archive['window_starts'], archive['window_ends']))).tolist()
        reference = next(iter(parse_rttm(entry['reference']).values()))
        evaluated = {}
        for method in methods:
            if index == 0:
                tracemalloc.start()
            started = time.perf_counter()
            stages, trace = infer(embeddings, windows, method)
            runtime = time.perf_counter() - started
            if index == 0:
                peak_bytes = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            else:
                peak_bytes = None
            for stage, (intervals, labels) in stages.items():
                geometry = intervals if stage == 'final' else partition_overlapping_windows(intervals)
                # Keep partition metrics on the same embedding-window sample at every stage.
                projected = labels if stage != 'final' else np.array([
                    labels[max(range(len(intervals)), key=lambda j:
                        max(0, min(end, intervals[j][1]) - max(start, intervals[j][0])))]
                    for start, end in windows])
                key = (tuple(map(tuple, geometry)), tuple(labels), tuple(projected))
                if key not in evaluated:
                    evaluated[key] = stage_metrics(rid, geometry, labels, reference, entry['duration'],
                                                   windows, projected)
                rows.append(dict(recording_id=rid, method=method, stage=stage,
                                 runtime_seconds=runtime, python_traced_peak_bytes=peak_bytes,
                                 **evaluated[key]))
            if trace.get('merge_decisions'):
                decisions.extend(dict(recording_id=rid, method=method, **row)
                                 for row in trace['merge_decisions'])
                audit.extend(dict(recording_id=rid, method=method, **row)
                             for row in reference_merge_audit(stages['initial'][1], trace, windows, reference))
            if index < args.stability_recordings and method in {
                    'fixed_075', 'centroid', 'robust', 'confidence_strict', 'confidence_moderate'}:
                base = stages['merged'][1]
                for seed in (11, 23):
                    for family in ('angular_0.02', 'subsample_0.75'):
                        rng = np.random.default_rng(seed)
                        ids = (np.arange(len(embeddings)) if family.startswith('angular') else
                               np.sort(rng.choice(len(embeddings), max(1, int(.75 * len(embeddings))), replace=False)))
                        changed = angular_noise(embeddings, .02, seed) if family.startswith('angular') else embeddings[ids]
                        started = time.perf_counter()
                        perturbed, alternative_trace = infer(changed, [windows[i] for i in ids], method)
                        alternative = perturbed['merged'][1]
                        stability.append(dict(recording_id=rid, method=method, family=family, seed=seed,
                            count_changed=len(set(base)) != len(set(alternative)),
                            ARI=float(adjusted_rand_score(base[ids], alternative)),
                            accepted_merges=alternative_trace.get('accepted_merges', 0),
                            merge_count_change=alternative_trace.get('accepted_merges', 0) - trace.get('accepted_merges', 0),
                            runtime_seconds=time.perf_counter() - started))
        if (index + 1) % 25 == 0 or index == 0:
            print(f'Evaluated {index + 1}/{len(manifest["entries"])}', flush=True)
    summary = summarize(rows)
    write_json(output / 'summary.json', summary)
    write_json(output / 'per_recording.json', rows)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'per_recording.csv', rows)
    write_csv(output / 'merge_decisions.csv', decisions)
    write_json(output / 'merge_traces.json', decisions)
    write_csv(output / 'reference_merge_audit.csv', audit)
    write_csv(output / 'stability.csv', stability)
    paired = paired_comparisons(rows)
    write_csv(output / 'paired_comparisons.csv', paired)
    write_json(output / 'paired_summary.json', paired_summary(paired))
    if not args.frozen:
        final = {row['method']: row for row in summary if row['stage'] == 'final' and row['reference_K'] == 'all'}
        baseline = final['fixed_075']
        eligible = []
        for name in ('confidence_strict', 'confidence_moderate'):
            row = final[name]
            pairs = [p for p in paired if p['method'] == name and p['baseline'] == 'fixed_075']
            if (row['DER'] <= baseline['DER'] + .25 and row['purity'] >= baseline['purity'] - .005
                    and np.mean([p['DER_delta'] > 10 for p in pairs]) <= .01):
                eligible.append(name)
        pool = eligible or ['confidence_strict', 'confidence_moderate']
        selected = min(pool, key=lambda name: (final[name]['DER'], final[name]['SpeakerError'],
            final[name]['count_MAE'], -final[name]['ARI'], -final[name]['purity'],
            final[name]['fragmentation'], -final[name]['count_accuracy'], name))
        write_json(output / 'frozen_candidate.json', dict(method=selected,
            configuration=METHODS[selected], eligible=selected in eligible,
            effective_merge_parameters=_confidence_merge_parameters(METHODS[selected]['config']),
            selection_protocol=SELECTION, cluster_source_sha256=sha256(ROOT / 'diarize/cluster.py'),
            checkpoint_sha256=CHECKPOINT_HASH, environment=environment()))


def paired_comparisons(rows):
    final = {(r['recording_id'], r['method']): r for r in rows if r['stage'] == 'final'}
    result = []
    for (rid, method), row in final.items():
        for baseline in ('fixed_075', 'centroid'):
            if method == baseline:
                continue
            base = final[(rid, baseline)]
            result.append(dict(recording_id=rid, method=method, baseline=baseline,
                **{key + '_delta': row[key] - base[key] for key in
                   ('DER', 'SpeakerError', 'count_error', 'ARI', 'purity', 'fragmentation')}))
    return result


def paired_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['baseline'])].append(row)
    return [dict(method=method, baseline=baseline,
        improved=sum(row['DER_delta'] < -1e-9 for row in values),
        worsened=sum(row['DER_delta'] > 1e-9 for row in values),
        unchanged=sum(abs(row['DER_delta']) <= 1e-9 for row in values),
        mean_DER_delta=float(np.mean([row['DER_delta'] for row in values])),
        median_DER_delta=float(np.median([row['DER_delta'] for row in values])),
        worst_regressions=sorted(values, key=lambda row: row['DER_delta'], reverse=True)[:5],
        largest_improvements=sorted(values, key=lambda row: row['DER_delta'])[:5])
        for (method, baseline), values in sorted(groups.items())]


def import_artifacts(args):
    """Index immutable diagnostic artifacts without performing encoder inference."""
    entries = []
    encoder_hash = sha256(ROOT / 'models/ecapa_tdnn.py')
    for folder in sorted(Path(args.artifacts).iterdir()):
        if not (folder / 'embeddings.npz').is_file():
            continue
        config = read_json(folder / 'config.json')
        rid = folder.name
        reference = Path(args.references) / (rid + '.rttm')
        if (config['checkpoint_sha256'] != CHECKPOINT_HASH or
                config['encoder_source_sha256'] != encoder_hash):
            raise ValueError(f'{rid}: incompatible checkpoint/encoder')
        if config.get('mode', 'full_pipeline') != 'full_pipeline':
            raise ValueError('Only full_pipeline caches are eligible here')
        if config.get('reference_sha256') and sha256(reference) != config['reference_sha256']:
            raise ValueError(f'{rid}: reference mismatch')
        parsed = next(iter(parse_rttm(str(reference)).values()))
        # Diagnostic references end within the recording; using inferred end adds no scored speech.
        duration = max(end for _, end, _ in parsed)
        with np.load(folder / 'embeddings.npz', allow_pickle=False) as archive:
            duration = max(duration, float(archive['window_ends'].max()))
        provenance = dict(config, reference_sha256=sha256(reference))
        entries.append(dict(recording_id=rid, archive=str((folder / 'embeddings.npz').resolve()),
            archive_sha256=sha256(folder / 'embeddings.npz'), reference=str(reference.resolve()),
            duration=duration, provenance=provenance))
    if not entries:
        raise ValueError('No diagnostic embeddings found')
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, dict(status='complete', entries=entries,
        evidence='exploratory reused-test', source_artifacts=str(Path(args.artifacts).resolve())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'evaluate', 'import-artifacts'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--conversations')
    parser.add_argument('--historical-manifest')
    parser.add_argument('--cache')
    parser.add_argument('--frozen')
    parser.add_argument('--artifacts')
    parser.add_argument('--references')
    parser.add_argument('--stability-recordings', type=int, default=25)
    args = parser.parse_args()
    if args.action == 'prepare':
        if not args.conversations or not args.historical_manifest:
            parser.error('prepare requires --conversations and --historical-manifest')
        prepare(args)
    elif args.action == 'evaluate':
        if not args.cache:
            parser.error('evaluate requires --cache')
        evaluate(args)
    else:
        if not args.artifacts or not args.references:
            parser.error('import-artifacts requires --artifacts and --references')
        import_artifacts(args)


if __name__ == '__main__':
    main()
