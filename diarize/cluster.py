"""
cluster.py — Кластеризация эмбеддингов говорящих.

Methods:
    - AHC  (Agglomerative Hierarchical Clustering) — простой, быстрый
    - Spectral — кластеризация по матрице сходства
    - NME-SC — auto-tuned spectral clustering using normalized maximum eigengap
"""

import heapq

import numpy
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


def merge_similar_clusters(
    embeddings: numpy.ndarray,
    labels: numpy.ndarray,
    similarity_threshold: float = 0.80,
    max_small_cluster_size: int = 4,
    min_similarity_margin: float = 0.05,
    return_diagnostics: bool = False,
):
    """Merge a small cluster into a clearly similar, strictly larger cluster."""
    embeddings = numpy.asarray(embeddings)
    labels = numpy.asarray(labels)
    diagnostics = dict(merge_attempts=0, accepted_merges=0, rejected_merges=0,
                       merge_similarity_scores=[], merge_candidates=[],
                       merge_rejection_reasons={})
    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1")
    if max_small_cluster_size < 1:
        raise ValueError("max_small_cluster_size must be positive")
    if not 0 <= min_similarity_margin <= 2:
        raise ValueError("min_similarity_margin must be between 0 and 2")
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must have equal length")
    if len(embeddings) == 0:
        result = numpy.array([], dtype=int)
        return (result, diagnostics) if return_diagnostics else result
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("embeddings must be 2-D and labels must be 1-D")
    if (not numpy.isfinite(embeddings).all()
            or (numpy.linalg.norm(embeddings, axis=1) == 0).any()):
        raise ValueError("Embeddings must be finite nonzero vectors")

    result = labels.copy()
    while True:
        unique, counts = numpy.unique(result, return_counts=True)
        if len(unique) < 2:
            break
        sizes = dict(zip(unique.tolist(), counts.tolist()))
        merged = False
        for fragment in sorted(unique, key=lambda label: (sizes[label], label)):
            if sizes[fragment] > max_small_cluster_size:
                continue
            targets = [label for label in unique if sizes[label] > sizes[fragment]]
            if not targets:
                continue
            fragment_centroid = embeddings[result == fragment].mean(axis=0)
            target_centroids = numpy.vstack([embeddings[result == label].mean(axis=0)
                                             for label in targets])
            fragment_norm = numpy.linalg.norm(fragment_centroid)
            target_norms = numpy.linalg.norm(target_centroids, axis=1)
            diagnostics['merge_attempts'] += 1
            if fragment_norm == 0 or (target_norms == 0).any():
                diagnostics['merge_similarity_scores'].append(None)
                diagnostics['merge_candidates'].append(dict(
                    fragment_label=int(fragment), fragment_size=sizes[fragment],
                    target_similarities=[], accepted=False,
                    rejection_reasons=['invalid_centroid']))
                diagnostics['merge_rejection_reasons']['invalid_centroid'] = (
                    diagnostics['merge_rejection_reasons'].get('invalid_centroid', 0) + 1)
                diagnostics['rejected_merges'] += 1
                continue
            similarities = target_centroids @ fragment_centroid / (target_norms * fragment_norm)
            order = numpy.argsort(-similarities, kind='stable')
            best_index = int(order[0])
            best = float(similarities[best_index])
            second = float(similarities[order[1]]) if len(order) > 1 else -1.0
            diagnostics['merge_similarity_scores'].append(best)
            reasons = []
            if best < similarity_threshold:
                reasons.append('below_similarity_threshold')
            if best - second < min_similarity_margin:
                reasons.append('insufficient_similarity_margin')
            diagnostics['merge_candidates'].append(dict(
                fragment_label=int(fragment), fragment_size=sizes[fragment],
                target_similarities=[dict(target_label=int(label), target_size=sizes[label],
                                          similarity=float(similarity))
                                     for label, similarity in zip(targets, similarities)],
                best_target_label=int(targets[best_index]), best_similarity=best,
                second_best_similarity=second, similarity_margin=best - second,
                accepted=not reasons, rejection_reasons=reasons))
            if not reasons:
                result[result == fragment] = targets[best_index]
                diagnostics['accepted_merges'] += 1
                merged = True
                break
            for reason in reasons:
                diagnostics['merge_rejection_reasons'][reason] = (
                    diagnostics['merge_rejection_reasons'].get(reason, 0) + 1)
            diagnostics['rejected_merges'] += 1
        if not merged:
            break

    return (result, diagnostics) if return_diagnostics else result

def cluster_ahc(
    embeddings: numpy.ndarray,
    n_speakers: int   = None,
    threshold: float  = 0.7,
    return_diagnostics: bool = False,
) -> numpy.ndarray | tuple[numpy.ndarray, dict]:
    """
    Агломеративная иерархическая кластеризация (AHC).
    """
    emb = normalize(embeddings, norm="l2")

    options = dict(metric="cosine", linkage="average")
    if return_diagnostics:
        options.update(compute_distances=True, compute_full_tree=True)
    if n_speakers is not None:
        model = AgglomerativeClustering(n_clusters=n_speakers, **options)
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, **options)

    labels = model.fit_predict(emb)
    if not return_diagnostics:
        return labels

    sizes = {index: 1 for index in range(len(emb))}
    merges = []
    for step, ((left, right), distance) in enumerate(zip(model.children_, model.distances_)):
        left, right = int(left), int(right)
        merged_size = sizes[left] + sizes[right]
        sizes[len(emb) + step] = merged_size
        merges.append(dict(
            merge_step=step,
            left_cluster_id=left,
            right_cluster_id=right,
            merge_distance=float(distance),
            merged_cluster_size=merged_size,
        ))
    diagnostics = dict(
        metric="cosine",
        linkage="average",
        threshold=None if n_speakers is not None else float(threshold),
        requested_n_speakers=n_speakers,
        children=model.children_.astype(int).tolist(),
        distances=model.distances_.astype(float).tolist(),
        merges=merges,
        labels=labels.astype(int).tolist(),
        cluster_sizes=numpy.unique(labels, return_counts=True)[1].astype(int).tolist(),
    )
    return labels, diagnostics


def _cut_ahc_hierarchy(children: list[list[int]], n_samples: int,
                       n_clusters: int) -> numpy.ndarray:
    """Cut an already fitted AHC tree into a deterministic flat partition."""
    nodes = [-(n_samples + len(children) - 1)]
    for _ in range(n_clusters - 1):
        left, right = children[-nodes[0] - n_samples]
        heapq.heappushpop(nodes, -left)
        heapq.heappush(nodes, -right)

    labels = numpy.empty(n_samples, dtype=int)
    for label, node in enumerate(nodes):
        descendants = [-node]
        while descendants:
            descendant = descendants.pop()
            if descendant < n_samples:
                labels[descendant] = label
            else:
                descendants.extend(children[descendant - n_samples])
    return labels


def estimate_ahc_speaker_count(
    distances: list[float] | numpy.ndarray,
    n_samples: int,
    max_speakers: int = 10,
    fallback_threshold: float = 0.7,
    robust: bool = False,
) -> tuple[int, dict]:
    """Estimate speaker count from an AHC hierarchy using an outlying merge gap.

    Robust mode excludes the candidate gap from the Tukey reference sample so
    a true outlier cannot inflate its own fence. Short hierarchies without
    comparison evidence use the fixed AHC threshold fallback.
    """
    distances = numpy.asarray(distances, dtype=float)
    if (n_samples < 1 or max_speakers < 1
            or not numpy.isfinite(fallback_threshold) or fallback_threshold < 0):
        raise ValueError("Invalid n_samples, max_speakers, or fallback_threshold")
    if len(distances) != n_samples - 1:
        raise ValueError("A complete AHC hierarchy must have n_samples - 1 distances")
    if not numpy.isfinite(distances).all() or numpy.any(numpy.diff(distances) < -1e-12):
        raise ValueError("AHC distances must be finite and nondecreasing")

    candidates = []
    for merge_index, gap in enumerate(numpy.diff(distances), start=1):
        count = n_samples - merge_index
        candidates.append(dict(
            n_speakers=int(count),
            gap=float(max(gap, 0.0)),
            distance_before=float(distances[merge_index - 1]),
            distance_after=float(distances[merge_index]),
            eligible=2 <= count <= max_speakers,
        ))
    eligible = [candidate for candidate in candidates if candidate['eligible']]
    selected = None
    cutoff = None
    best = (max(eligible,
                key=lambda candidate: (candidate['gap'], -candidate['n_speakers']))
            if eligible else None)
    remaining = [] if best is None else [candidate['gap'] for candidate in eligible
                                         if candidate is not best]
    runner_up = max(remaining) if remaining else None
    gap_margin = None if runner_up is None else best['gap'] - runner_up
    if robust and (not eligible or not remaining):
        selection_reason = 'insufficient_candidate_gaps'
    elif robust:
        q1, q3 = numpy.percentile(remaining, [25, 75])
        cutoff = float(q3 + 1.5 * (q3 - q1))
        if numpy.isclose(best['gap'], runner_up, rtol=1e-12, atol=1e-12):
            selection_reason = 'no_unique_outlier'
        elif best['gap'] > cutoff:
            selected = best
            selection_reason = 'outlying_gap'
        else:
            selection_reason = 'no_outlying_gap'
    elif eligible:
        gaps = numpy.asarray([candidate['gap'] for candidate in eligible])
        q1, q3 = numpy.percentile(gaps, [25, 75])
        cutoff = float(q3 + 1.5 * (q3 - q1))
        flat_then_jump = float(gaps.min()) <= 1e-12 < best['gap']
        if best['gap'] > cutoff or flat_then_jump:
            selected = best
            selection_reason = 'outlying_gap' if best['gap'] > cutoff else 'flat_then_jump'
        else:
            selection_reason = 'no_outlying_gap'
    else:
        selection_reason = 'insufficient_candidate_gaps'

    fallback_count = min(max_speakers, n_samples - int(numpy.sum(distances < fallback_threshold)))
    count = selected['n_speakers'] if selected else fallback_count
    diagnostics = dict(
        estimator='adaptive_ahc_robust' if robust else 'adaptive_ahc',
        estimated_n_speakers=int(count),
        max_speakers=int(max_speakers),
        distances=distances.astype(float).tolist(),
        gaps=candidates,
        outlier_cutoff=cutoff,
        selected_gap=None if selected is None else selected['gap'],
        selection='outlying_gap' if selected is not None else 'fixed_threshold_fallback',
        selection_reason=selection_reason,
        eligible_gap_count=len(eligible),
        best_gap=None if best is None else best['gap'],
        runner_up_gap=runner_up,
        gap_margin=gap_margin,
        fallback_threshold=float(fallback_threshold),
        fallback_n_speakers=int(fallback_count),
    )
    return int(count), diagnostics


def cluster_spectral(
    embeddings: numpy.ndarray,
    n_speakers: int   = None,
    max_speakers: int = 10,
    seed: int = 42,
) -> numpy.ndarray:
    """
    Спектральная кластеризация 
    """
    emb = normalize(embeddings, norm="l2")

    sim_matrix = numpy.dot(emb, emb.T)
    sim_matrix = numpy.clip(sim_matrix, 0, 1)

    if n_speakers is not None:
        model = SpectralClustering(
            n_clusters        = n_speakers,
            affinity          = "precomputed",
            random_state      = seed,
            assign_labels     = "kmeans",
        )
        labels = model.fit_predict(sim_matrix)
        return labels

    # автоподбор числа спикеров
    best_k      = 1
    best_score  = -1
    best_labels = numpy.zeros(len(embeddings), dtype=int)
    n_max = min(max_speakers, len(embeddings) - 1)

    for k in range(2, n_max + 1):
        model = SpectralClustering(
            n_clusters    = k,
            affinity      = "precomputed",
            random_state  = seed,
            assign_labels = "kmeans",
        )
        lbls = model.fit_predict(sim_matrix)
        # силуэтный коэффициент — чем выше тем лучше разделение
        score = silhouette_score(emb, lbls, metric="cosine")
        if score > best_score:
            best_score  = score
            best_k      = k
            best_labels = lbls

    print(f"[cluster] Автоподбор: {best_k} спикеров (silhouette={best_score:.3f})")
    return best_labels


def cluster_nme_sc(
    embeddings: numpy.ndarray,
    n_speakers: int = None,
    max_speakers: int = 10,
    seed: int = 42,
) -> numpy.ndarray:
    """Normalized maximum eigengap spectral clustering (Park et al., 2019)."""
    embeddings = normalize(embeddings, norm="l2")
    n_samples = len(embeddings)
    if n_speakers == n_samples:
        return numpy.arange(n_samples)
    if n_samples < 8:
        # NME cannot auto-tune while floor(N/4) offers fewer than two p values.
        return cluster_ahc(embeddings, n_speakers=n_speakers, threshold=0.65)

    affinity = embeddings @ embeddings.T
    # Self-loops cancel in D - A and must not consume an informative neighbor.
    numpy.fill_diagonal(affinity, -numpy.inf)
    max_neighbors = max(1, n_samples // 4)
    max_clusters = min(max_speakers, n_samples - 1)
    gap_limit = (max_clusters if n_speakers is None
                 else min(n_samples - 1, max(max_clusters, n_speakers)))
    best_ratio = numpy.inf
    best_vectors = None
    best_gaps = None
    epsilon = 1e-10

    for p in range(1, max_neighbors + 1):
        pruned = numpy.zeros_like(affinity)
        for row in range(n_samples):
            neighbors = numpy.argsort(-affinity[row], kind="stable")[:p]
            pruned[row, neighbors] = 1.0
        adjacency = (pruned + pruned.T) / 2.0
        laplacian = numpy.diag(adjacency.sum(axis=1)) - adjacency
        eigenvalues, eigenvectors = numpy.linalg.eigh(laplacian)
        eigenvalues = numpy.maximum(eigenvalues, 0.0)
        gaps = numpy.diff(eigenvalues)[:gap_limit]
        candidate_gap = (float(gaps.max(initial=0.0)) if n_speakers is None
                         else float(gaps[n_speakers - 1]))
        if (n_speakers is not None
                and (not numpy.isfinite(candidate_gap) or candidate_gap <= epsilon)):
            continue
        normalized_gap = candidate_gap / (float(eigenvalues[-1]) + epsilon)
        ratio = p / normalized_gap if normalized_gap > epsilon else numpy.inf
        if n_speakers is not None and not numpy.isfinite(ratio):
            continue
        if best_vectors is None or ratio < best_ratio:
            best_ratio, best_vectors, best_gaps = ratio, eigenvectors, gaps

    if best_vectors is None:
        if n_speakers is not None:
            return cluster_ahc(embeddings, n_speakers=n_speakers)
        return numpy.zeros(n_samples, dtype=int)
    estimated = int(numpy.argmax(best_gaps) + 1) if best_gaps.max(initial=0.0) > epsilon else 1
    k = n_speakers if n_speakers is not None else estimated
    if k == 1:
        return numpy.zeros(n_samples, dtype=int)
    spectral_embedding = best_vectors[:, :k]
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(spectral_embedding)


def cluster_embeddings(
    embeddings: numpy.ndarray,
    method: str       = "ahc",
    n_speakers: int   = None,
    threshold: float  = 0.7,
    max_speakers: int = 10,
    seed: int = 42,
    diagnostics: dict | None = None,
    count_estimator: str | None = None,
) -> numpy.ndarray:
    """
    Кластеризация эмбеддингов говорящих.
    """
    embeddings = numpy.asarray(embeddings)
    if method not in {"ahc", "spectral", "nme_sc"} or max_speakers < 1:
        raise ValueError("Invalid method or max_speakers")
    if count_estimator not in {None, "adaptive_ahc", "adaptive_ahc_robust"}:
        raise ValueError(
            "count_estimator must be None, 'adaptive_ahc', or 'adaptive_ahc_robust'")
    if count_estimator is not None and (method != "ahc" or n_speakers is not None):
        raise ValueError("count_estimator requires automatic AHC clustering")
    if n_speakers is not None and not 1 <= n_speakers <= len(embeddings):
        raise ValueError("n_speakers must be between 1 and number of embeddings")
    if len(embeddings) == 0:
        return numpy.array([], dtype=int)

    if embeddings.ndim != 2 or not numpy.isfinite(embeddings).all() or (numpy.linalg.norm(embeddings, axis=1) == 0).any():
        raise ValueError("Embeddings must be finite nonzero vectors")
    if len(embeddings) == 1 or n_speakers == 1 or (n_speakers is None and max_speakers == 1):
        labels = numpy.zeros(len(embeddings), dtype=int)
        if diagnostics is not None and method == "ahc":
            if len(embeddings) == 1:
                diagnostics['ahc'] = dict(
                    metric="cosine", linkage="average",
                    threshold=None if n_speakers == 1 or max_speakers == 1 else float(threshold),
                    requested_n_speakers=n_speakers, children=[], distances=[], merges=[],
                    labels=[0], cluster_sizes=[1])
            else:
                _, diagnostics['ahc'] = cluster_ahc(
                    embeddings, n_speakers=1, return_diagnostics=True)
            if count_estimator in {'adaptive_ahc', 'adaptive_ahc_robust'}:
                _, estimator_diagnostics = estimate_ahc_speaker_count(
                    diagnostics['ahc']['distances'], len(embeddings), max_speakers,
                    fallback_threshold=threshold,
                    robust=count_estimator == 'adaptive_ahc_robust')
                diagnostics['ahc'].update(estimator_diagnostics)
            else:
                diagnostics['ahc'].update(
                    estimator='known_count' if n_speakers == 1 else 'fixed_threshold',
                    estimated_n_speakers=1,
                )
        return labels
    if n_speakers == len(embeddings):
        labels = numpy.arange(len(embeddings))
        if diagnostics is not None and method == "ahc":
            _, diagnostics['ahc'] = cluster_ahc(
                embeddings, n_speakers=n_speakers, return_diagnostics=True)
            diagnostics['ahc']['labels'] = labels.tolist()
            diagnostics['ahc'].update(estimator='known_count',
                                      estimated_n_speakers=len(labels))
        return labels
    if method in {"spectral", "nme_sc"} and n_speakers is None:
        # The spectral estimators need a one-speaker/small-sample fallback.
        gate = cluster_ahc(embeddings, threshold=threshold)
        if len(set(gate)) == 1 or len(embeddings) == 2:
            return gate

    if method == "ahc" and count_estimator in {"adaptive_ahc", "adaptive_ahc_robust"}:
        _, hierarchy = cluster_ahc(embeddings, n_speakers=1, return_diagnostics=True)
        estimated, estimator_diagnostics = estimate_ahc_speaker_count(
            hierarchy['distances'], len(embeddings), max_speakers,
            fallback_threshold=threshold,
            robust=count_estimator == 'adaptive_ahc_robust')
        labels = _cut_ahc_hierarchy(hierarchy['children'], len(embeddings), estimated)
        if diagnostics is not None:
            diagnostics['ahc'] = hierarchy
            diagnostics['ahc'].update(
                threshold=None,
                requested_n_speakers=estimated,
                labels=labels.astype(int).tolist(),
                cluster_sizes=numpy.unique(
                    labels, return_counts=True)[1].astype(int).tolist(),
            )
            diagnostics['ahc'].update(estimator_diagnostics)
    elif method == "ahc":
        result = cluster_ahc(embeddings, n_speakers=n_speakers, threshold=threshold,
                             return_diagnostics=diagnostics is not None)
        if diagnostics is None:
            labels = result
        else:
            labels, diagnostics['ahc'] = result
            diagnostics['ahc'].update(
                estimator='known_count' if n_speakers is not None else 'fixed_threshold',
                estimated_n_speakers=len(set(labels)),
            )
        if n_speakers is None and len(set(labels)) > max_speakers:
            if diagnostics is not None:
                diagnostics['ahc_threshold'] = diagnostics['ahc']
            result = cluster_ahc(embeddings, n_speakers=max_speakers,
                                 return_diagnostics=diagnostics is not None)
            if diagnostics is None:
                labels = result
            else:
                labels, diagnostics['ahc'] = result
    elif method == "spectral":
        labels = cluster_spectral(embeddings, n_speakers=n_speakers, max_speakers=max_speakers, seed=seed)
    elif method == "nme_sc":
        labels = cluster_nme_sc(embeddings, n_speakers=n_speakers,
                                max_speakers=max_speakers, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    n_found = len(set(labels))
    print(f"[cluster] Метод: {method}, найдено спикеров: {n_found}")
    return labels


def merge_short_segments(
    segments: list[tuple[float, float]],
    labels: numpy.ndarray,
    min_duration: float = 0.5,
) -> tuple[list[tuple[float, float]], numpy.ndarray]:
    """
    Объединить слишком короткие сегменты с соседними.
    Короткий сегмент получает метку предыдущего сегмента.
    """
    if len(segments) != len(labels):
        raise ValueError("segments and labels must have equal length")
    if min_duration < 0:
        raise ValueError("min_duration must be non-negative")
    if len(segments) == 0:
        return segments, labels

    new_segments = []
    new_labels   = []

    for i, ((start, end), label) in enumerate(zip(segments, labels)):
        duration = end - start
        if duration < min_duration and len(new_labels) > 0 and start <= new_segments[-1][1]:
            prev_start, _ = new_segments[-1]
            new_segments[-1] = (prev_start, end)
        else:
            new_segments.append((start, end))
            new_labels.append(label)

    return new_segments, numpy.array(new_labels)


def smooth_labels(
    segments: list[tuple[float, float]],
    labels: numpy.ndarray,
    window: int = 3,
) -> numpy.ndarray:
    """
    Сгладить метки спикеров скользящим большинством.
    Убирает одиночные ошибочные переключения.
    """
    if len(segments) != len(labels):
        raise ValueError("segments and labels must have equal length")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if len(labels) <= window:
        return labels

    smoothed = labels.copy()
    half = window // 2
    for i in range(half, len(labels) - half):
        window_labels = labels[i - half: i + half + 1]
        unique, counts = numpy.unique(window_labels, return_counts=True)
        smoothed[i] = unique[numpy.argmax(counts)]

    return smoothed


if __name__ == "__main__":
    # Тест на синтетических данных
    numpy.random.seed(42)
    D = 192  # размерность эмбеддингов ECAPA

    # 3 спикера по 5 сегментов каждый
    spk1 = numpy.random.randn(5, D) + numpy.array([1.0] + [0.0] * (D - 1))
    spk2 = numpy.random.randn(5, D) + numpy.array([0.0, 1.0] + [0.0] * (D - 2))
    spk3 = numpy.random.randn(5, D) + numpy.array([0.0, 0.0, 1.0] + [0.0] * (D - 3))
    embeddings = numpy.vstack([spk1, spk2, spk3])

    print("=== Тест AHC ===")
    labels_ahc = cluster_embeddings(embeddings, method="ahc", threshold=0.7)
    print("Метки:", labels_ahc)

    print("\n=== Тест Spectral ===")
    labels_spec = cluster_embeddings(embeddings, method="spectral", max_speakers=5)
    print("Метки:", labels_spec)
