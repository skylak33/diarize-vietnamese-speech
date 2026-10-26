"""
cluster.py — Кластеризация эмбеддингов говорящих.

Два метода:
    - AHC  (Agglomerative Hierarchical Clustering) — простой, быстрый
    - Spectral — кластеризация по матрице сходства
"""

import numpy
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

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
    if method not in {"ahc", "spectral"} or max_speakers < 1:
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
    if method == "spectral" and n_speakers is None:
        # AHC gate allows a one-speaker solution, undefined for silhouette selection.
        gate = cluster_ahc(embeddings, threshold=threshold)
        if len(set(gate)) == 1 or len(embeddings) == 2:
            return gate

    if method == "ahc":
        labels = cluster_ahc(embeddings, n_speakers=n_speakers, threshold=threshold)
        if n_speakers is None and len(set(labels)) > max_speakers:
            labels = cluster_ahc(embeddings, n_speakers=max_speakers)
    elif method == "spectral":
        labels = cluster_spectral(embeddings, n_speakers=n_speakers, max_speakers=max_speakers, seed=seed)
    else:
        raise ValueError(f"Неизвестный метод: {method}. Используй 'ahc' или 'spectral'")

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
