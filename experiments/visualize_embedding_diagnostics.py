"""Render embedding-analysis artifacts without rerunning diarization."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

from eval.der import load_rttm


STAGES = ('initial_ahc', 'after_centroid_merge', 'after_smoothing')
COLORS = ('#1976d2', '#00897b', '#ef6c00', '#8e24aa', '#d81b60', '#546e7a')


def _save(fig, output):
    output = Path(output)
    if output.suffix.lower() not in {'.png', '.svg'}:
        raise ValueError('Output must use .png or .svg')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=160)


def _read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def load_artifacts(artifact_dir):
    """Load the optional diagnostic files produced for one recording."""
    artifact_dir = Path(artifact_dir)
    archive = np.load(artifact_dir / 'embeddings.npz')
    required = {'embeddings', 'window_starts', 'window_ends'}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"embeddings.npz is missing: {', '.join(sorted(missing))}")
    embeddings = np.asarray(archive['embeddings'])
    starts, ends = np.asarray(archive['window_starts']), np.asarray(archive['window_ends'])
    if embeddings.ndim != 2 or len(embeddings) != len(starts) or len(starts) != len(ends):
        raise ValueError('Embedding arrays must have matching window counts')
    trace = _read_json(artifact_dir / 'pipeline_trace.json', {})
    if trace and trace.get('schema_version') not in (None, 1):
        raise ValueError('Unsupported pipeline_trace.json schema_version')
    return dict(embeddings=embeddings, window_starts=starts, window_ends=ends,
                trace=trace, config=_read_json(artifact_dir / 'config.json', {}),
                merges=_read_json(artifact_dir / 'ahc_merges.json'))


def _stage_labels(trace, name):
    stages = trace.get('stages', trace)
    value = stages.get(name) if isinstance(stages, dict) else None
    if isinstance(value, dict):
        value = value.get('labels', value.get('window_labels'))
    return None if value is None else np.asarray(value)


def _segments(trace, name):
    value = trace.get(name, trace.get('stages', {}).get(name, []))
    labels = None
    if isinstance(value, dict):
        labels = value.get('labels')
        value = value.get('segments', value.get('intervals', []))
    result = []
    for segment in value or []:
        if isinstance(segment, dict):
            start, end = segment.get('start'), segment.get('end')
            label = segment.get('speaker', segment.get('label', segment.get('cluster')))
        else:
            start, end, *rest = segment
            label = rest[0] if rest else None
        if label is None and labels is not None and len(result) < len(labels):
            label = labels[len(result)]
        if start is not None and end is not None:
            result.append((float(start), float(end), str(label) if label is not None else 'unknown'))
    return result


def _reference(trace):
    value = trace.get('reference', trace.get('reference_segments', []))
    if isinstance(value, dict):
        value = value.get('segments', value.get('intervals', []))
    return _segments({'reference': value}, 'reference')


def plot_similarity_heatmap(embeddings, output, reference=None, window_starts=None, window_ends=None):
    """Plot chronological pairwise cosine similarities."""
    import matplotlib.pyplot as plt
    if not len(embeddings):
        raise ValueError('No embeddings to plot')
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or (norm == 0).any():
        raise ValueError('Embeddings must be finite nonzero vectors')
    similarity = (embeddings / norm) @ (embeddings / norm).T
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(similarity, vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest')
    if reference and window_starts is not None and window_ends is not None:
        centers = (np.asarray(window_starts) + np.asarray(window_ends)) / 2
        for _, boundary, _ in reference[:-1]:
            index = np.searchsorted(centers, boundary)
            ax.axvline(index - .5, color='black', linewidth=.5, alpha=.55)
            ax.axhline(index - .5, color='black', linewidth=.5, alpha=.55)
    fig.colorbar(image, ax=ax, label='Cosine similarity')
    ax.set(xlabel='Embedding window index', ylabel='Embedding window index',
           title='Chronological embedding similarity')
    _save(fig, output)
    plt.close(fig)


def _timeline_row(ax, segments, row, color, label):
    for start, end, _ in segments:
        if end > start:
            ax.broken_barh([(start, end - start)], (row - .34, .68), facecolors=color)
    return label


def plot_speaker_timeline(starts, ends, trace, output):
    """Plot independent reference and predicted-label lanes on one time axis."""
    import matplotlib.pyplot as plt
    lanes = []
    reference = _reference(trace)
    if reference:
        lanes.append(('Reference', reference))
    for stage in STAGES:
        labels = _stage_labels(trace, stage)
        if labels is not None and len(labels) == len(starts):
            lanes.append((stage, [(s, e, label) for s, e, label in zip(starts, ends, labels)]))
    for name in ('partitioned_segments', 'segments_after_short_merge', 'final_rttm_segments'):
        segments = _segments(trace, name)
        if segments:
            lanes.append((name, segments))
    if not lanes:
        raise ValueError('Trace contains no timeline stages')
    fig, ax = plt.subplots(figsize=(12, max(2.5, .7 * len(lanes))))
    for row, (name, segments) in enumerate(reversed(lanes)):
        by_label = {}
        for segment in segments:
            by_label.setdefault(segment[2], []).append(segment)
        for index, (label, group) in enumerate(sorted(by_label.items())):
            _timeline_row(ax, group, row, COLORS[index % len(COLORS)], label)
        ax.text(-.01, row, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8)
    ax.set(yticks=[], xlabel='Time (s)', title='Reference and predicted stages (labels are not aligned)')
    ax.grid(axis='x', alpha=.25)
    _save(fig, output)
    plt.close(fig)


def plot_similarity_distributions(embeddings, labels, output, reference_labels=None):
    """Compare within/between predicted clusters and optional reference speakers."""
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    if len(labels) != len(embeddings):
        raise ValueError('Labels must match embeddings')
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    upper = np.triu_indices(len(embeddings), k=1)
    pairs = [(labels[upper[0]] == labels[upper[1]], 'Within predicted cluster', '#1976d2'),
             (labels[upper[0]] != labels[upper[1]], 'Between predicted clusters', '#ef6c00')]
    if reference_labels is not None and len(reference_labels) == len(embeddings):
        reference_labels = np.asarray(reference_labels, dtype=object)
        valid = (reference_labels[upper[0]] != None) & (reference_labels[upper[1]] != None)
        pairs.extend([(valid & (reference_labels[upper[0]] == reference_labels[upper[1]]), 'Same reference speaker', '#00897b'),
                      (valid & (reference_labels[upper[0]] != reference_labels[upper[1]]), 'Different reference speakers', '#8e24aa')])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    values = similarity[upper]
    for mask, name, color in pairs:
        if mask.any():
            ax.hist(values[mask], bins=30, range=(-1, 1), density=True, histtype='step', linewidth=1.8,
                    label=name, color=color)
    ax.set(xlabel='Cosine similarity', ylabel='Density', title='Embedding similarity distributions')
    ax.legend(fontsize=8)
    _save(fig, output)
    plt.close(fig)


def _linkage(merges, n_samples):
    payload = merges or {}
    if isinstance(payload, dict) and isinstance(payload.get('ahc'), dict):
        payload = payload['ahc']
    children = payload.get('children') if isinstance(payload, dict) else None
    distances = payload.get('distances') if isinstance(payload, dict) else None
    if children is None and isinstance(payload, dict):
        records = payload.get('merges', [])
        children = [[item.get('left_cluster_id'), item.get('right_cluster_id')] for item in records]
        distances = [item.get('merge_distance') for item in records]
    if children is None or distances is None or len(children) != len(distances):
        raise ValueError('AHC merge artifact needs matching children and distances')
    children = np.asarray(children, dtype=int)
    distances = np.asarray(distances, dtype=float)
    if children.shape != (max(0, n_samples - 1), 2) or not np.isfinite(distances).all():
        raise ValueError('AHC hierarchy is incomplete or invalid')
    counts = np.zeros(len(children))
    for index, pair in enumerate(children):
        counts[index] = sum(1 if child < n_samples else counts[child - n_samples] for child in pair)
    return np.column_stack((children, distances, counts))


def plot_ahc_dendrogram(merges, n_samples, output, threshold=None):
    """Render the exact hierarchy recorded by AgglomerativeClustering."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    linkage = _linkage(merges, n_samples)
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage, ax=ax, labels=[str(index) for index in range(n_samples)], leaf_rotation=90)
    if threshold is not None:
        ax.axhline(float(threshold), color='#d81b60', linestyle='--', label='AHC threshold')
        ax.legend(fontsize=8)
    ax.set(xlabel='Chronological embedding window index', ylabel='Cosine distance', title='AHC merge hierarchy')
    _save(fig, output)
    plt.close(fig)


def _reference_window_labels(trace, starts, ends):
    labels = trace.get('reference_window_labels')
    if labels is not None:
        return labels
    reference = _reference(trace)
    if not reference:
        return None
    result = []
    for start, end in zip(starts, ends):
        coverage = {}
        for ref_start, ref_end, speaker in reference:
            coverage[speaker] = coverage.get(speaker, 0) + max(0, min(end, ref_end) - max(start, ref_start))
        best = max(coverage, key=coverage.get, default=None)
        total = sum(coverage.values())
        result.append(best if best is not None and coverage[best] / (end - start) >= .8
                      and total <= end - start + 1e-9 else None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--format', choices=('png', 'svg'), default='png')
    parser.add_argument('--reference_rttm')
    args = parser.parse_args(argv)
    data = load_artifacts(args.artifact_dir)
    if args.reference_rttm:
        recording_id = data['trace'].get('recording_id', Path(args.artifact_dir).name)
        references = load_rttm(args.reference_rttm)
        if recording_id not in references:
            parser.error(f'Reference RTTM has no recording {recording_id}')
        data['trace']['reference'] = references[recording_id]
    output_dir = Path(args.output_dir or args.artifact_dir)
    suffix = args.format
    plot_similarity_heatmap(data['embeddings'], output_dir / f'similarity_heatmap.{suffix}', _reference(data['trace']),
                            data['window_starts'], data['window_ends'])
    plot_speaker_timeline(data['window_starts'], data['window_ends'], data['trace'], output_dir / f'speaker_timeline.{suffix}')
    labels = _stage_labels(data['trace'], 'after_smoothing')
    if labels is None:
        labels = _stage_labels(data['trace'], 'initial_ahc')
    if labels is not None and len(labels) == len(data['embeddings']):
        plot_similarity_distributions(data['embeddings'], labels, output_dir / f'similarity_distributions.{suffix}',
                                      _reference_window_labels(data['trace'], data['window_starts'], data['window_ends']))
    if data['merges'] is not None and len(data['embeddings']) > 1:
        threshold = data['config'].get('inference_configuration', {}).get('threshold')
        plot_ahc_dendrogram(data['merges'], len(data['embeddings']), output_dir / f'ahc_dendrogram.{suffix}', threshold)


if __name__ == '__main__':
    main()
