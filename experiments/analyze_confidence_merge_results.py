"""Describe frozen merge outcomes and probe the few accepted-merge cases."""

import argparse
import csv
import json
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark_confidence_aware_merging import (
    ROOT, angular_noise, infer, read_json, write_json, write_csv, sha256)


def stability_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['family'])].append(row)
    return [dict(method=method, family=family, runs=len(values),
        count_changes=sum(str(row['count_changed']) == 'True' for row in values),
        mean_ARI=float(np.mean([float(row['ARI']) for row in values])),
        changed_merge_links=sum(int(row['changed_merge_links']) for row in values))
        for (method, family), values in sorted(groups.items())]


def load_entry(entry):
    with np.load(entry['archive'], allow_pickle=False) as archive:
        embeddings = archive['embeddings']
        windows = (archive['windows'] if 'windows' in archive else np.column_stack((
            archive['window_starts'], archive['window_ends']))).tolist()
    return embeddings, windows


def probe(entry, candidate):
    """Compare perturbations on accepted-merge cases, without consulting references."""
    embeddings, windows = load_entry(entry)
    rows = []
    for method in ('fixed_075', 'centroid', 'robust', candidate):
        stages, trace = infer(embeddings, windows, method)
        initial, baseline = stages['initial'][1], stages['merged'][1]
        for seed in (11, 23):
            for family in ('angular_0.02', 'subsample_0.75'):
                ids = (np.arange(len(embeddings)) if family.startswith('angular') else
                       np.sort(np.random.default_rng(seed).choice(len(embeddings),
                               max(1, int(.75 * len(embeddings))), replace=False)))
                changed = (angular_noise(embeddings, .02, seed)
                           if family.startswith('angular') else embeddings[ids])
                alternative, alternative_trace = infer(changed, [windows[i] for i in ids], method)
                output = alternative['merged'][1]
                links = (initial[ids, None] != initial[ids]) & (baseline[ids, None] == baseline[ids])
                alt_initial = alternative['initial'][1]
                alt_links = (alt_initial[:, None] != alt_initial) & (output[:, None] == output)
                rows.append(dict(recording_id=entry['recording_id'], method=method, family=family,
                    seed=seed, count_changed=len(set(baseline)) != len(set(output)),
                    ARI=float(adjusted_rand_score(baseline[ids], output)),
                    baseline_accepted=trace.get('accepted_merges', 0),
                    alternative_accepted=alternative_trace.get('accepted_merges', 0),
                    changed_merge_links=int(np.count_nonzero(np.triu(links != alt_links, 1)))))
    return rows


def run(root):
    root = Path(root)
    output = root / 'posthoc_analysis'
    if output.exists():
        raise FileExistsError(output)
    frozen = read_json(root / 'development/frozen_candidate.json')
    if sha256(ROOT / 'diarize/cluster.py') != frozen['cluster_source_sha256']:
        raise ValueError('Frozen clustering source changed')
    candidate = frozen['method']
    cache_names = {'development': 'cache_development/manifest.json',
                   'exploratory_datahive': 'datahive_cache.json',
                   'exploratory_viyt': 'viyt_cache.json'}
    result, targeted, profiles = {}, [], []
    for dataset, cache_name in cache_names.items():
        folder = root / dataset
        traces = [row for row in read_json(folder / 'merge_traces.json') if row['method'] == candidate]
        with (folder / 'reference_merge_audit.csv').open(newline='') as handle:
            audit = [row for row in csv.DictReader(handle) if row['method'] == candidate]
        accepted_ids = {row['recording_id'] for row in traces if row['accepted']}
        result[dataset] = dict(proposals=len(traces), accepted=sum(row['accepted'] for row in traces),
            rejected=sum(not row['accepted'] for row in traces),
            first_rejection_reasons=dict(Counter(row['rejection_reason'] for row in traces if not row['accepted'])),
            cross_speaker_merges=sum(row['cross_speaker'] == 'True' for row in audit),
            accepted_recordings=sorted(accepted_ids))
        entries = read_json(root / cache_name)['entries']
        for entry in entries:
            if entry['recording_id'] in accepted_ids:
                targeted.extend(dict(dataset=dataset, **row) for row in probe(entry, candidate))
        # Profile a multi-speaker development input and a long natural recording once.
        profile_id = 'development_000005' if dataset == 'development' else 'vi_000007'
        examples = [entry for entry in entries if entry['recording_id'] == profile_id]
        for entry in examples:
            embeddings, windows = load_entry(entry)
            for method in ('fixed_075', 'centroid', candidate):
                tracemalloc.start()
                started = time.perf_counter()
                stages, trace = infer(embeddings, windows, method)
                seconds = time.perf_counter() - started
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                profiles.append(dict(dataset=dataset, recording_id=entry['recording_id'],
                    windows=len(windows), method=method, runtime_seconds=seconds,
                    python_traced_peak_bytes=peak, initial_K=len(set(stages['initial'][1])),
                    merged_K=len(set(stages['merged'][1])), accepted=trace.get('accepted_merges', 0)))
    with (root / 'development/stability.csv').open(newline='') as handle:
        result['balanced_development_stability'] = stability_summary(list(csv.DictReader(handle)))
    result['targeted_accepted_case_stability'] = stability_summary(targeted)
    result['profiles'] = profiles
    result['frozen_candidate_sha256'] = sha256(root / 'development/frozen_candidate.json')
    write_json(output / 'summary.json', result)
    write_csv(output / 'accepted_case_stability.csv', targeted)
    write_csv(output / 'profiles.csv', profiles)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/confidence_aware_merging')
    run(parser.parse_args().root)
