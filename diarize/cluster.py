"""
cluster.py — Кластеризация эмбеддингов говорящих.

Methods:
    - AHC  (Agglomerative Hierarchical Clustering) — простой, быстрый
    - Spectral — кластеризация по матрице сходства
    - NME-SC — auto-tuned spectral clustering using normalized maximum eigengap
"""

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
                       merge_similarity_scores=[])
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
                diagnostics['rejected_merges'] += 1
                continue
            similarities = target_centroids @ fragment_centroid / (target_norms * fragment_norm)
            order = numpy.argsort(-similarities, kind='stable')
            best_index = int(order[0])
            best = float(similarities[best_index])
            second = float(similarities[order[1]]) if len(order) > 1 else -1.0
            diagnostics['merge_similarity_scores'].append(best)
            if best >= similarity_threshold and best - second >= min_similarity_margin:
                result[result == fragment] = targets[best_index]
                diagnostics['accepted_merges'] += 1
                merged = True
                break
            diagnostics['rejected_merges'] += 1
        if not merged:
            break

    return (result, diagnostics) if return_diagnostics else result

def cluster_ahc(
    embeddings: numpy.ndarray,
    n_speakers: int   = None,
    threshold: float  = 0.7,
) -> numpy.ndarray:
    """
    Агломеративная иерархическая кластеризация (AHC).
    """
    emb = normalize(embeddings, norm="l2")

    if n_speakers is not None:
        model = AgglomerativeClustering(
            n_clusters     = n_speakers,
            metric         = "cosine",
            linkage        = "average",
        )
    else:
        model = AgglomerativeClustering(
            n_clusters        = None,
            distance_threshold = threshold,
            metric            = "cosine",
            linkage           = "average",
        )

    labels = model.fit_predict(emb)
    return labels


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
    if n_samples < 4:
        # The paper searches p in [1, floor(N/4)], which is empty for N < 4.
        return cluster_ahc(embeddings, n_speakers=n_speakers, threshold=0.65)

    affinity = embeddings @ embeddings.T
    max_neighbors = max(1, n_samples // 4)
    max_clusters = min(max_speakers, n_samples - 1)
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
        gaps = numpy.diff(eigenvalues)[:max_clusters]
        maximum_gap = float(gaps.max(initial=0.0))
        normalized_gap = maximum_gap / (float(eigenvalues[-1]) + epsilon)
        ratio = p / normalized_gap if normalized_gap > epsilon else numpy.inf
        if best_vectors is None or ratio < best_ratio:
            best_ratio, best_vectors, best_gaps = ratio, eigenvectors, gaps

    if best_vectors is None:
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
) -> numpy.ndarray:
    """
    Кластеризация эмбеддингов говорящих.
    """
    embeddings = numpy.asarray(embeddings)
    if method not in {"ahc", "spectral", "nme_sc"} or max_speakers < 1:
        raise ValueError("Invalid method or max_speakers")
    if n_speakers is not None and not 1 <= n_speakers <= len(embeddings):
        raise ValueError("n_speakers must be between 1 and number of embeddings")
    if len(embeddings) == 0:
        return numpy.array([], dtype=int)

    if embeddings.ndim != 2 or not numpy.isfinite(embeddings).all() or (numpy.linalg.norm(embeddings, axis=1) == 0).any():
        raise ValueError("Embeddings must be finite nonzero vectors")
    if len(embeddings) == 1 or n_speakers == 1 or (n_speakers is None and max_speakers == 1):
        return numpy.zeros(len(embeddings), dtype=int)
    if n_speakers == len(embeddings):
        return numpy.arange(len(embeddings))
    if method in {"spectral", "nme_sc"} and n_speakers is None:
        # The spectral estimators need a one-speaker/small-sample fallback.
        gate = cluster_ahc(embeddings, threshold=threshold)
        if len(set(gate)) == 1 or len(embeddings) == 2:
            return gate

    if method == "ahc":
        labels = cluster_ahc(embeddings, n_speakers=n_speakers, threshold=threshold)
        if n_speakers is None and len(set(labels)) > max_speakers:
            labels = cluster_ahc(embeddings, n_speakers=max_speakers)
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
