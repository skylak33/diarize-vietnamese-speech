"""Offline separability diagnostics for immutable COCAM proposal traces."""

import argparse
import contextlib
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.cluster import cluster_embeddings
from experiments.analyze_partition_quality import associate_reference_windows
from experiments.analyze_confidence_merge_results import load_entry
from experiments.benchmark_confidence_aware_merging import read_json, write_json, write_csv
from eval.der import parse_rttm
from utils.provenance import environment, sha256


ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    'centroid_similarity': 1, 'median_cross_similarity': 1, 'q20_cross_similarity': 1,
    'cross_similarity_dispersion': -1, 'compactness_loss': -1,
    'symmetric_margin': 1, 'support': 1,
}
POLICY = {
    'labels': 'Any mixed/unlabeled window or multiple reference identities within either '
              'cluster makes the pair ambiguous. Otherwise equal identities are positive, '
              'different identities negative. Silence padding is allowed when speech is attributable.',
    'alignment': 'Source turns are authoritative; disagreement with rounded RTTM window '
                 'attribution is marked uncertain and ambiguous.',
    'population_A': 'All actual observations; primary pair-level analysis keeps the first '
                    'observation of each unordered membership pair. Later decisions remain separate.',
    'population_B': 'All nonempty initial-cluster pairs, including singletons; no support prefilter.',
    'missing': 'No imputation. Missing margin means no alternative, not a zero margin.',
    'directions': FEATURES,
    'operating_points': 'Development-only empirical ranking envelopes, never production thresholds.',
    'precision_targets': [.90, .95, .99, 1.],
    'recall_targets': [.10, .25, .50, .75, .90],
    'support_bins': 'minimum size: 1, 2-4, 5-15, >=16; high imbalance max/min >=8',
    'overlap': 'Intersection of class-normalized 20-bin histograms on the pooled feature range.',
    'closest_failed_gate': 'Smallest domain-normalized numeric deficit: cosine /2, '
                           'compactness drop and symmetric margin /4, support /minimum size. '
                           'Boolean or missing evidence is not ranked.',
    'uncertainty': '200 source-speaker/content connected-component bootstrap draws, seed 42, '
                   'only if >=10 components and no component covers >50% of recordings. '
                   'Otherwise no confidence interval.',
}


def label_pair(members_a, members_b, associations):
    compositions = []
    pure = []
    for members in (members_a, members_b):
        rows = [associations[index] for index in members]
        counts = Counter(row['reference_label'] for row in rows if row['status'] == 'single_speaker')
        uncertain = sum(row['status'] != 'single_speaker' for row in rows)
        compositions.append(dict(reliable_speakers=dict(counts), uncertain_windows=uncertain))
        pure.append(next(iter(counts)) if not uncertain and len(counts) == 1 else None)
    label = ('ambiguous' if None in pure else 'positive' if pure[0] == pure[1] else 'negative')
    return dict(reference_merge_label=label, reference_a=compositions[0], reference_b=compositions[1])


def support_bin(size):
    return ('singleton' if size <= 1 else 'small_2_4' if size <= 4 else
            'medium_5_15' if size <= 15 else 'large_16_plus')


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()[:20]


def replay_candidates(recording_id, initial, decisions, associations):
    """Attach offline labels while replaying only saved accepted transitions."""
    labels = np.asarray(initial).copy()
    if labels.ndim != 1 or len(labels) != len(associations):
        raise ValueError('Initial labels and associations must align')
    rows, by_step = [], defaultdict(list)
    for decision in decisions:
        by_step[int(decision['merge_step'])].append(decision)
    for expected_step, (step, proposals) in enumerate(sorted(by_step.items())):
        if step != expected_step:
            raise ValueError('Nonconsecutive merge steps')
        partition = sorted([np.flatnonzero(labels == label).tolist() for label in np.unique(labels)])
        state_id = _digest(partition)
        accepted, seen = [], set()
        for decision in proposals:
            a, b = decision['cluster_a'], decision['cluster_b']
            members_a, members_b = np.flatnonzero(labels == a).tolist(), np.flatnonzero(labels == b).tolist()
            if (not members_a or not members_b or a == b or
                    len(members_a) != decision['cluster_a_size'] or len(members_b) != decision['cluster_b_size']):
                raise ValueError('Saved cluster size/membership mismatch')
            pair = tuple(sorted((a, b)))
            if pair in seen:
                raise ValueError('Duplicate proposal within a merge step')
            seen.add(pair)
            if decision['cluster_count_before'] != len(partition):
                raise ValueError('Saved cluster count mismatch')
            pair_id = recording_id + ':' + _digest(sorted([members_a, members_b]))
            support = min(len(members_a), len(members_b))
            cross = decision['cross_similarity']
            rows.append(dict(recording_id=recording_id, candidate_id=pair_id,
                observation_id=pair_id + f':{step}', merge_step=step, partition_state_id=state_id,
                cluster_a=a, cluster_b=b, membership_a=members_a, membership_b=members_b,
                cluster_a_size=len(members_a), cluster_b_size=len(members_b),
                **label_pair(members_a, members_b, associations),
                centroid_similarity=decision['centroid_similarity'],
                median_cross_similarity=cross['median'], q20_cross_similarity=cross['lower_quantile'],
                cross_similarity_dispersion=cross['iqr'],
                compactness_before=[decision['compactness_a'], decision['compactness_b']],
                compactness_after=decision['merged_compactness'], compactness_loss=decision['compactness_drop'],
                symmetric_margin=decision['similarity_margin'], support=support,
                support_bin=support_bin(support), size_ratio=max(len(members_a), len(members_b)) / support,
                highly_imbalanced=max(len(members_a), len(members_b)) >= 8 * support,
                accepted=decision['accepted'], all_gates_pass=all(decision['decision_evidence'].values()),
                rejection_reasons=decision['rejection_reasons'], decision_evidence=decision['decision_evidence']))
            if decision['accepted']:
                if decision['cluster_count_after'] != len(partition) - 1:
                    raise ValueError('Invalid accepted merge count')
                accepted.append((a, b))
        if len(accepted) > 1:
            raise ValueError('More than one accepted merge at a step')
        for a, b in accepted:
            labels[labels == b] = a
    return rows


def initial_population(rows, initial):
    rows = [dict(row) for row in rows if row['merge_step'] == 0]
    expected = set(combinations(np.unique(initial).tolist(), 2))
    observed = {tuple(sorted((row['cluster_a'], row['cluster_b']))) for row in rows}
    if expected != observed or len(rows) != len(expected):
        raise ValueError('Incomplete or duplicate initial pair population')
    return rows


def deduplicate(rows):
    groups = {}
    for row in rows:
        key = row['candidate_id']
        if key not in groups:
            groups[key] = dict(row, visits=0, ever_accepted=False, seen_steps=[])
        first = groups[key]
        first['visits'] += 1
        first['seen_steps'].append(row['merge_step'])
        first['ever_accepted'] |= bool(row['accepted'])
    return list(groups.values())


def _feature_values(rows, feature):
    eligible = [row for row in rows if row['reference_merge_label'] in {'positive', 'negative'}]
    valid = [row for row in eligible if row.get(feature) is not None and np.isfinite(row[feature])]
    y = np.array([row['reference_merge_label'] == 'positive' for row in valid], dtype=int)
    x = np.array([row[feature] for row in valid], dtype=float)
    return eligible, valid, y, x


def feature_statistics(rows, feature):
    eligible, valid, y, x = _feature_values(rows, feature)
    result = dict(feature=feature, direction=FEATURES[feature],
        positive=int(sum(y)), negative=int(len(y) - sum(y)),
        missing_positive=sum(row['reference_merge_label'] == 'positive' for row in eligible) - int(sum(y)),
        missing_negative=sum(row['reference_merge_label'] == 'negative' for row in eligible) - int(len(y) - sum(y)),
        ROC_AUC=None, Average_Precision=None, histogram_overlap=None)
    for label, value in [('positive', 1), ('negative', 0)]:
        subset = x[y == value]
        quantiles = np.quantile(subset, [.05, .25, .5, .75, .95]).tolist() if len(subset) else [None] * 5
        result.update({label + '_' + name: quantile for name, quantile in
                       zip(('q05', 'q25', 'median', 'q75', 'q95'), quantiles)})
        result[label + '_IQR'] = quantiles[3] - quantiles[1] if len(subset) else None
    if len(np.unique(y)) == 2:
        scores = FEATURES[feature] * x
        result.update(ROC_AUC=float(roc_auc_score(y, scores)),
                      Average_Precision=float(average_precision_score(y, scores)))
        low, high = x.min(), x.max()
        edges = np.linspace(low - .5 if low == high else low, high + .5 if low == high else high, 21)
        a, b = (np.histogram(x[y == value], edges)[0] / np.sum(y == value) for value in (1, 0))
        result['histogram_overlap'] = float(np.minimum(a, b).sum())
    return result


def decision_quality(rows, field='accepted'):
    counts = Counter((row['reference_merge_label'], bool(row[field])) for row in rows)
    tp, fp = counts['positive', True], counts['negative', True]
    fn, tn = counts['positive', False], counts['negative', False]
    ambiguous = counts['ambiguous', True] + counts['ambiguous', False]
    accepted = tp + fp + counts['ambiguous', True]
    return dict(total=len(rows), TP=tp, FP=fp, FN=fn, TN=tn, ambiguous=ambiguous,
        accepted_ambiguous=counts['ambiguous', True],
        precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
        false_merge_rate=fp / (tp + fp) if tp + fp else None,
        false_positive_rate=fp / (fp + tn) if fp + tn else None,
        acceptance_rate=accepted / len(rows) if rows else None,
        label_coverage=(len(rows) - ambiguous) / len(rows) if rows else None,
        strict_safe_acceptance_fraction=tp / accepted if accepted else None)


def rejection_summary(rows):
    conditions = sorted({key for row in rows for key in row['decision_evidence']})
    return [dict(condition=key,
        positive_failed=sum(row['reference_merge_label'] == 'positive' and not row['accepted'] and
                            not row['decision_evidence'][key] for row in rows),
        positive_only_failure=sum(row['reference_merge_label'] == 'positive' and not row['accepted'] and
            not row['decision_evidence'][key] and sum(not v for v in row['decision_evidence'].values()) == 1
            for row in rows),
        negative_passed=sum(row['reference_merge_label'] == 'negative' and row['decision_evidence'][key]
                            for row in rows)) for key in conditions]


def operating_points(rows, feature):
    _, _, y, x = _feature_values(rows, feature)
    if len(np.unique(y)) < 2:
        return []
    fpr, recall, thresholds = roc_curve(y, FEATURES[feature] * x, drop_intermediate=False)
    tp = np.rint(recall * np.sum(y)).astype(int)
    fp = np.rint(fpr * np.sum(1 - y)).astype(int)
    accepted = tp + fp
    precision = np.divide(tp, accepted, out=np.zeros(len(tp)), where=accepted > 0)
    rows = []
    for kind, targets in [('precision_at_recall', POLICY['recall_targets']),
                          ('recall_at_precision', POLICY['precision_targets'])]:
        for target in targets:
            indices = np.flatnonzero((recall >= target) if kind == 'precision_at_recall' else
                                     ((precision >= target) & (accepted > 0)))
            index = (max(indices, key=lambda i: (precision[i] if kind == 'precision_at_recall' else
                                                recall[i], -accepted[i])) if len(indices) else 0)
            rows.append(dict(feature=feature, kind=kind, target=target,
                precision=float(precision[index]) if accepted[index] else None,
                recall=float(recall[index]), false_positive_rate=float(fpr[index]),
                accepted=int(accepted[index]), TP=int(tp[index]), FP=int(fp[index]),
                score_threshold=float(thresholds[index]) if np.isfinite(thresholds[index]) else None,
                direction=FEATURES[feature], diagnostic_only=True))
    return rows


def dependency_groups(entries):
    parent = {entry['recording_id']: entry['recording_id'] for entry in entries}
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    owners = {}
    for entry in entries:
        rid = entry['recording_id']
        for kind, field in [('speaker', 'speaker_ids'), ('content', 'source_hashes')]:
            for source in entry[field]:
                key = kind, source
                if key in owners:
                    a, b = find(rid), find(owners[key])
                    parent[max(a, b)] = min(a, b)
                owners[key] = rid
    return {rid: find(rid) for rid in parent}


def annotate_deficits(rows, parameters):
    gates = {
        'centroid_similarity': ('centroid_similarity', 'centroid_similarity_threshold', 2, 1),
        'cross_median': ('median_cross_similarity', 'cross_median_threshold', 2, 1),
        'cross_lower_quantile': ('q20_cross_similarity', 'cross_lower_threshold', 2, 1),
        'compactness': ('compactness_loss', 'max_compactness_drop', 4, -1),
        'similarity_margin': ('symmetric_margin', 'min_similarity_margin', 4, 1),
    }
    for row in rows:
        deficits = {}
        for gate, (feature, threshold, domain, direction) in gates.items():
            if not row['decision_evidence'][gate] and row[feature] is not None:
                deficits[gate] = max(0., direction * (parameters[threshold] - row[feature]) / domain)
        if not row['decision_evidence']['sufficient_support']:
            deficits['sufficient_support'] = (parameters['min_cluster_size'] - row['support']) / parameters['min_cluster_size']
        row['numeric_gate_deficits'] = deficits
        row['closest_failed_numeric_gate'] = min(deficits, key=deficits.get) if deficits else None
        row['failed_gate_count'] = sum(not passed for passed in row['decision_evidence'].values())


def support_analysis(rows):
    slices = {name: [row for row in rows if row['support_bin'] == name]
              for name in ('singleton', 'small_2_4', 'medium_5_15', 'large_16_plus')}
    slices['highly_imbalanced'] = [row for row in rows if row['highly_imbalanced']]
    result = []
    for name, values in slices.items():
        counts = Counter(row['reference_merge_label'] for row in values)
        stats = feature_statistics(values, 'centroid_similarity')
        result.append(dict(slice=name, total=len(values), positive=counts['positive'], negative=counts['negative'],
            ambiguous=counts['ambiguous'], positive_prevalence=counts['positive'] / (
                counts['positive'] + counts['negative']) if counts['positive'] + counts['negative'] else None,
            accepted=sum(row['accepted'] for row in values),
            centroid_ROC_AUC=stats['ROC_AUC'], positive_centroid_median=stats['positive_median'],
            negative_centroid_median=stats['negative_median'],
            positive_support_pass=sum(row['reference_merge_label'] == 'positive' and
                                     row['decision_evidence']['sufficient_support'] for row in values)))
    return result


def grouped_uncertainty(rows, groups):
    sizes = Counter(groups.values())
    info = dict(components=len(sizes), largest_component_recordings=max(sizes.values(), default=0),
                recordings=len(groups), bootstrap_replicates=200, intervals=[])
    if len(sizes) < 10 or max(sizes.values(), default=0) > .5 * len(groups):
        info['reason'] = 'Too few independent source components or a dominant component; no CI reported.'
        return info
    blocks = defaultdict(list)
    for row in rows:
        blocks[groups[row['recording_id']]].append(row)
    ids = sorted(sizes)
    estimates = defaultdict(list)
    rng = np.random.default_rng(42)
    for _ in range(200):
        sample = [row for group in rng.choice(ids, len(ids), replace=True) for row in blocks[group]]
        for feature in FEATURES:
            metric = feature_statistics(sample, feature)
            if metric['ROC_AUC'] is not None:
                estimates[feature].append(metric['ROC_AUC'])
    for feature, values in estimates.items():
        low, high = np.quantile(values, [.025, .975])
        info['intervals'].append(dict(feature=feature, ROC_AUC_low=float(low), ROC_AUC_high=float(high),
                                      valid_bootstrap_replicates=len(values)))
    return info


def figures(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
    for ax, feature in zip(axes.flat, FEATURES):
        _, _, y, x = _feature_values(rows, feature)
        for value, label, color in [(1, 'Positive', '#1b8a5a'), (0, 'Negative', '#c84a44')]:
            subset = np.sort(x[y == value])
            if len(subset):
                ax.step(subset, np.arange(1, len(subset) + 1) / len(subset), where='post',
                        label=f'{label} n={len(subset)}', color=color)
        ax.set(title=feature.replace('_', ' '), ylabel='Empirical CDF')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    fig.savefig(output / 'feature_distributions.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for feature in FEATURES:
        _, _, y, x = _feature_values(rows, feature)
        if len(np.unique(y)) < 2:
            continue
        scores = FEATURES[feature] * x
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        axes[0].plot(fpr, tpr, label=f'{feature}: {roc_auc_score(y, scores):.3f}')
        axes[1].step(recall, precision, where='post', label=f'{feature}: {average_precision_score(y, scores):.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    for ax, title, xlabel, ylabel in [(axes[0], 'ROC (legend: AUC)', 'False-positive rate', 'Recall'),
                                      (axes[1], 'Precision-recall (legend: AP)', 'Recall', 'Precision')]:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, loc='lower right' if ax is axes[0] else 'lower left')
        ax.grid(alpha=.2)
    fig.savefig(output / 'roc_pr.png', dpi=160)
    plt.close(fig)
    names = list(FEATURES)
    known = [row for row in rows if row['reference_merge_label'] != 'ambiguous']
    matrix = np.full((len(names), len(names)), np.nan)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            values = [(row[left], row[right]) for row in known if row[left] is not None and row[right] is not None
                      and np.isfinite(row[left]) and np.isfinite(row[right])]
            if len(values) > 1:
                a, b = np.asarray(values).T
                if np.ptp(a) and np.ptp(b):
                    matrix[i, j] = spearmanr(a, b).statistic
    fig, ax = plt.subplots(figsize=(9, 8), layout='constrained')
    heat = ax.imshow(np.ma.masked_invalid(matrix), vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           title='Spearman feature correlation: unambiguous pairs')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(heat, ax=ax, shrink=.7)
    fig.savefig(output / 'feature_correlation.png', dpi=160)
    plt.close(fig)
    return dict(features=names, spearman=[[float(value) if np.isfinite(value) else None for value in row]
                                         for row in matrix])


def representative_cases(rows):
    examples = []
    categories = {
        'rejected_positive': sorted([row for row in rows if row['reference_merge_label'] == 'positive'
            and not row['accepted']], key=lambda row: row['centroid_similarity'] or -1, reverse=True)[:3],
        'negative_passes_many_gates': sorted([row for row in rows if row['reference_merge_label'] == 'negative'],
            key=lambda row: (row['failed_gate_count'], -(row['centroid_similarity'] or -1)))[:3],
        'accepted_positive': [row for row in rows if row['reference_merge_label'] == 'positive' and row['accepted']],
    }
    positives = [row for row in rows if row['reference_merge_label'] == 'positive'
                 and row['centroid_similarity'] is not None]
    negatives = [row for row in rows if row['reference_merge_label'] == 'negative'
                 and row['centroid_similarity'] is not None]
    if positives and negatives:
        positive, negative = min(((a, b) for a in positives for b in negatives),
                                 key=lambda pair: abs(pair[0]['centroid_similarity'] - pair[1]['centroid_similarity']))
        categories['overlapping_centroid_positive'] = [positive]
        categories['overlapping_centroid_negative'] = [negative]
    for category, values in categories.items():
        examples.extend(dict(category=category, **row) for row in values)
    return examples


def _initial_labels(entry, embeddings):
    trace_path = Path(entry['archive']).parent / 'pipeline_trace.json'
    if trace_path.is_file():
        return np.asarray(read_json(trace_path)['stages']['initial_ahc']['labels'], dtype=int), 'saved_artifact'
    with contextlib.redirect_stdout(io.StringIO()):
        labels = cluster_embeddings(embeddings, method='ahc', threshold=.75, seed=42)
    return labels, 'reconstructed_once_missing_membership_artifact'


def run(args):
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    frozen = read_json(source / 'development/frozen_candidate.json')
    if sha256(ROOT / 'diarize/cluster.py') != frozen['cluster_source_sha256']:
        raise ValueError('Frozen clustering source changed')
    checkpoint_hash = sha256(ROOT / 'exps/exp1/model/model_0034.model')
    if checkpoint_hash != frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint mismatch')
    manifest = read_json(source / 'cache_development/manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete development cache')
    saved = read_json(source / 'development/merge_traces.json')
    by_recording = defaultdict(list)
    for row in saved:
        if row['method'] == frozen['method']:
            by_recording[row['recording_id']].append(row)
    protocol = dict(policy=POLICY, frozen_configuration=frozen, environment=environment(),
        analysis_source_sha256=sha256(__file__), checkpoint_sha256=checkpoint_hash,
        source_manifest_sha256=sha256(source / 'cache_development/manifest.json'),
        source_trace_sha256=sha256(source / 'development/merge_traces.json'))
    write_json(output / 'protocol.json', protocol)
    actual, initial_pairs, provenance, membership_artifacts = [], [], [], {}
    source_owners, alignment_counts = {}, Counter()
    parameters = frozen['effective_merge_parameters']
    for entry in manifest['entries']:
        rid = entry['recording_id']
        metadata_path = Path(entry['reference']).with_suffix('.json')
        metadata = read_json(metadata_path)
        if metadata['split'] != 'development' or metadata['conversation_id'] != rid:
            raise ValueError('Only the declared synthetic development split is permitted')
        for path, digest in [(metadata_path, entry['provenance']['metadata_sha256']),
                             (Path(entry['reference']), entry['provenance']['reference_sha256']),
                             (Path(entry['archive']), entry['archive_sha256'])]:
            if sha256(path) != digest:
                raise ValueError(f'{rid}: provenance mismatch: {path}')
        audio_path = metadata_path.parent / metadata['audio_path']
        if sha256(audio_path) != entry['provenance']['audio_sha256']:
            raise ValueError(f'{rid}: audio hash mismatch')
        if (set(metadata['speaker_ids']) != set(entry['speaker_ids']) or
                {turn['source_hash'] for turn in metadata['turns']} != set(entry['source_hashes'])):
            raise ValueError(f'{rid}: source identity/hash mismatch')
        for turn in metadata['turns']:
            if turn['speaker_id'] not in metadata['speaker_ids']:
                raise ValueError('Unknown source speaker')
            key = turn['source_utterance_id']
            identity = turn['speaker_id'], turn['source_hash']
            if key in source_owners and source_owners[key] != identity:
                raise ValueError('Source utterance identity/content conflict')
            source_owners[key] = identity
        embeddings, windows = load_entry(entry)
        source_reference = [(turn['start'], turn['end'], turn['speaker_id']) for turn in metadata['turns']]
        reference = next(iter(parse_rttm(entry['reference']).values()))
        expected = sorted(source_reference)
        observed = sorted(reference)
        if (len(expected) != len(observed) or any(a[2] != b[2] or
                abs(a[0] - b[0]) > .001 or abs(a[1] - b[1]) > .001 for a, b in zip(expected, observed))):
            raise ValueError(f'{rid}: source-turn/RTTM mismatch')
        associations = associate_reference_windows(windows, source_reference)
        rounded = associate_reference_windows(windows, reference)
        for row, other in zip(associations, rounded):
            if (row['status'], row['reference_label']) != (other['status'], other['reference_label']):
                row['status'], row['reference_label'] = 'alignment_disagreement', None
            alignment_counts[row['status']] += 1
        initial, origin = _initial_labels(entry, embeddings)
        membership_artifacts[rid] = initial.tolist()
        candidates = replay_candidates(rid, initial, by_recording[rid], associations)
        actual.extend(candidates)
        initial_pairs.extend(initial_population(candidates, initial))
        provenance.append(dict(recording_id=rid, initial_K=len(set(initial.tolist())),
            windows=len(windows), label_origin=origin, archive_sha256=entry['archive_sha256'],
            metadata_sha256=entry['provenance']['metadata_sha256'], speaker_ids=entry['speaker_ids']))
    if len(actual) != sum(len(rows) for rows in by_recording.values()):
        raise ValueError('Some saved proposals could not be replayed')
    annotate_deficits(actual, parameters)
    annotate_deficits(initial_pairs, parameters)
    unique = deduplicate(actual)
    populations = {'actual_raw': actual, 'actual_unique_first': unique, 'initial_all': initial_pairs}
    write_csv(output / 'candidate_pairs.csv', [dict(population=name, **row)
        for name, rows in populations.items() if name != 'actual_unique_first' for row in rows])
    write_csv(output / 'unique_candidate_pairs.csv', unique)
    write_json(output / 'initial_memberships.json', membership_artifacts)
    write_csv(output / 'provenance.csv', provenance)
    feature_rows, confusion_rows, rejection_rows, support_rows, operating_rows = [], [], [], [], []
    summary = dict(populations={}, source_speakers=len({speaker for entry in manifest['entries'] for speaker in entry['speaker_ids']}),
                   source_utterances=len(source_owners), alignment_counts=dict(alignment_counts))
    for name, rows in populations.items():
        feature_rows.extend(dict(population=name, **feature_statistics(rows, feature)) for feature in FEATURES)
        confusion_rows.append(dict(population=name, decision='selected', **decision_quality(rows)))
        confusion_rows.append(dict(population=name, decision='all_gates_pass', **decision_quality(rows, 'all_gates_pass')))
        rejection_rows.extend(dict(population=name, **row) for row in rejection_summary(rows))
        support_rows.extend(dict(population=name, **row) for row in support_analysis(rows))
        operating_rows.extend(dict(population=name, **point) for feature in FEATURES for point in operating_points(rows, feature))
        counts = Counter(row['reference_merge_label'] for row in rows)
        summary['populations'][name] = dict(total=len(rows), labels=dict(counts),
            recordings_with_pairs=len({row['recording_id'] for row in rows}),
            positive_recordings=len({row['recording_id'] for row in rows if row['reference_merge_label'] == 'positive'}),
            positive_source_speakers=len({speaker for row in rows if row['reference_merge_label'] == 'positive'
                for speaker in row['reference_a']['reliable_speakers']}),
            later_accepted_pairs=sum(row.get('ever_accepted', False) and not row['accepted'] for row in rows))
    groups = dependency_groups(manifest['entries'])
    summary['uncertainty'] = grouped_uncertainty(unique, groups)
    summary['feature_correlations'] = figures(unique, output / 'figures')
    summary['recordings_total'] = len(manifest['entries'])
    summary['initial_generation_missing_pairs'] = 0
    summary['dynamic_revisits'] = len(actual) - len(unique)
    write_csv(output / 'feature_summary.csv', feature_rows)
    write_csv(output / 'decision_confusion.csv', confusion_rows)
    write_csv(output / 'rejection_analysis.csv', rejection_rows)
    write_csv(output / 'support_analysis.csv', support_rows)
    write_csv(output / 'operating_points.csv', operating_rows)
    rejected_positive = [row for row in unique if row['reference_merge_label'] == 'positive' and not row['accepted']]
    write_csv(output / 'rejected_positive_candidates.csv', rejected_positive)
    examples = representative_cases(unique)
    # The requested historical case is kept outside development metrics and operating points.
    viyt_manifest = read_json(source / 'viyt_cache.json')
    case = next(entry for entry in viyt_manifest['entries'] if entry['recording_id'] == 'V-cuzXOX9sM')
    embeddings, windows = load_entry(case)
    initial, _ = _initial_labels(case, embeddings)
    reference = next(iter(parse_rttm(case['reference']).values()))
    case_trace = [row for row in read_json(source / 'exploratory_viyt/merge_traces.json')
                  if row['recording_id'] == 'V-cuzXOX9sM' and row['method'] == frozen['method']]
    case_rows = replay_candidates('V-cuzXOX9sM', initial, case_trace,
                                  associate_reference_windows(windows, reference))
    annotate_deficits(case_rows, parameters)
    write_csv(output / 'exploratory_viyt_case.csv', case_rows)
    examples.extend(dict(category='exploratory_accepted', **row) for row in case_rows if row['accepted'])
    write_csv(output / 'representative_failures.csv', examples)
    summary['exploratory_case_accepted'] = [row for row in case_rows if row['accepted']]
    write_json(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'feature_correlations'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/confidence_aware_merging')
    parser.add_argument('--output', default='results/merge_separability')
    run(parser.parse_args())
