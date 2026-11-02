"""
pipeline.py — Полный пайплайн диаризации говорящего.
    audio.wav → VAD → сегментация → ECAPA-TDNN → кластеризация → RTTM
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ecapa_tdnn import ECAPA_TDNN
from utils.model import load_model
from utils.seed import set_seed
from diarize.vad import load_audio, get_speech_segments
from diarize.cluster import (cluster_embeddings, merge_short_segments,
                             merge_similar_clusters, smooth_labels)
from diarize.config import (load_config, pipeline_kwargs, set_config_value,
                            validate_config)

SAMPLE_RATE = 16000


def extract_embedding(encoder: ECAPA_TDNN, audio: numpy.ndarray,
                      start: int, end: int) -> numpy.ndarray:
    chunk = audio[start:end].astype(numpy.float32)
    if not len(chunk):
        raise ValueError("Cannot embed an empty audio segment")
    min_len = 8000
    if len(chunk) < min_len:
        chunk = numpy.pad(chunk, (0, min_len - len(chunk)), "wrap")
    with torch.no_grad():
        emb = encoder(torch.from_numpy(chunk).unsqueeze(0).to(next(encoder.parameters()).device), aug=False)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy()


def extract_embeddings_for_segments(
    encoder: ECAPA_TDNN,
    audio: numpy.ndarray,
    segments: list[tuple[float, float]],
    window_sec: float = 1.5,
    step_sec: float = 0.75,
    min_segment_sec: float = 0.5,
) -> tuple[list[tuple[float, float]], numpy.ndarray]:
    if not 0 < step_sec <= window_sec or min_segment_sec <= 0:
        raise ValueError("Require 0 < step_sec <= window_sec and min_segment_sec > 0")
    sub_segments = []
    embeddings = []
    window_samples = max(1, int(window_sec * SAMPLE_RATE))
    step_samples = max(1, int(step_sec * SAMPLE_RATE))
    min_samples = max(1, int(min_segment_sec * SAMPLE_RATE))

    for seg_start, seg_end in segments:
        s = int(seg_start * SAMPLE_RATE)
        e = int(seg_end * SAMPLE_RATE)
        seg_len = e - s
        if seg_len < min_samples:
            continue
        if seg_len <= window_samples:
            sub_segments.append((seg_start, seg_end))
            embeddings.append(extract_embedding(encoder, audio, s, e))
        else:
            pos = s
            while pos + window_samples <= e:
                sub_segments.append((pos / SAMPLE_RATE, (pos + window_samples) / SAMPLE_RATE))
                embeddings.append(extract_embedding(encoder, audio, pos, pos + window_samples))
                pos += step_samples
            if sub_segments[-1][1] < seg_end:
                pos = max(s, e - window_samples)
                sub_segments.append((pos / SAMPLE_RATE, e / SAMPLE_RATE))
                embeddings.append(extract_embedding(encoder, audio, pos, e))

    print(f"[diarize] Извлечено эмбеддингов: {len(embeddings)}")
    return sub_segments, numpy.array(embeddings)


def segments_to_rttm(segments: list[tuple[float, float]],
                     labels: numpy.ndarray, file_id: str = "audio") -> str:
    lines = []
    for (start, end), label in zip(segments, labels):
        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} "
            f"<NA> <NA> SPEAKER_{label:02d} <NA> <NA>"
        )
    return "\n".join(lines)


def merge_consecutive_segments(
    segments: list[tuple[float, float]],
    labels: numpy.ndarray,
    gap_threshold: float = 0.3,
) -> tuple[list[tuple[float, float]], numpy.ndarray]:
    if not segments:
        return segments, labels
    if len(segments) != len(labels):
        raise ValueError("segments and labels must have equal length")
    merged_segs = [segments[0]]
    merged_labels = [labels[0]]
    for i in range(1, len(segments)):
        gap = segments[i][0] - merged_segs[-1][1]
        if labels[i] == merged_labels[-1] and gap <= gap_threshold:
            merged_segs[-1] = (merged_segs[-1][0], max(merged_segs[-1][1], segments[i][1]))
        else:
            merged_segs.append(segments[i])
            merged_labels.append(labels[i])
    return merged_segs, numpy.array(merged_labels)


def partition_overlapping_windows(segments):
    """Assign a single speaker per time point; split overlap at window-center midpoints."""
    result = [list(segment) for segment in segments]
    for i in range(1, len(segments)):
        if segments[i][0] < segments[i-1][1]:
            boundary = sum(segments[i-1] + segments[i]) / 4
            boundary = max(segments[i][0], min(segments[i-1][1], boundary))
            result[i-1][1] = boundary
            result[i][0] = boundary
    return [tuple(segment) for segment in result]


def diarize(
    audio_path: str,
    model_path: str,
    n_speakers: int = None,
    method: str = "ahc",
    threshold: float = 0.65,
    vad_threshold: float = 0.5,
    output_rttm: str = None,
    window_sec: float = 1.5,
    step_sec: float = 0.75,
    device: str = "auto",
    seed: int = 42,
    max_speakers: int = 10,
    encoder=None,
    min_segment_sec: float = 0.5,
    diagnostics: dict | None = None,
    centroid_merge: bool = False,
    centroid_similarity_threshold: float = 0.80,
    max_small_cluster_size: int = 4,
    centroid_similarity_margin: float = 0.05,
    vad_min_silence_ms: int = 100,
    vad_min_speech_ms: int = 250,
    vad_speech_pad_ms: int = 30,
) -> str:
    set_seed(seed)
    if diagnostics is not None:
        diagnostics.update(requested_n_speakers=n_speakers, n_embedding_windows=0,
                           n_clusters_before_postprocess=0, n_speakers_after_postprocess=0,
                           cluster_sizes=[], n_clusters_before_postmerge=0,
                           n_clusters_after_postmerge=0,
                           cluster_sizes_before_postmerge=[],
                           cluster_sizes_after_postmerge=[], merge_attempts=0,
                           accepted_merges=0, rejected_merges=0,
                           merge_similarity_scores=[],
                           centroid_merge_requested=centroid_merge,
                           centroid_merge_skipped_known_count=False)
    if not 0 < step_sec <= window_sec or not 0 <= vad_threshold <= 1:
        raise ValueError("Require 0 < step_sec <= window_sec and 0 <= vad_threshold <= 1")
    if not isinstance(vad_min_silence_ms, int) or vad_min_silence_ms <= 0:
        raise ValueError("vad_min_silence_ms must be a positive integer")
    if not isinstance(vad_min_speech_ms, int) or vad_min_speech_ms <= 0:
        raise ValueError("vad_min_speech_ms must be a positive integer")
    if not isinstance(vad_speech_pad_ms, int) or vad_speech_pad_ms < 0:
        raise ValueError("vad_speech_pad_ms must be a non-negative integer")
    if output_rttm:
        Path(output_rttm).parent.mkdir(parents=True, exist_ok=True)
        Path(output_rttm).write_text("", encoding="utf-8")
    print(f"\n[diarize] === Шаг 1: Загрузка аудио ===")
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr
    if duration == 0:
        raise ValueError(f"Empty audio: {audio_path}")
    print(f"[diarize] Файл: {audio_path}, длительность: {duration:.1f} сек")

    print(f"\n[diarize] === Шаг 2: VAD ===")
    speech_segments = get_speech_segments(audio_path, threshold=vad_threshold,
                                          min_speech_ms=vad_min_speech_ms,
                                          min_silence_ms=vad_min_silence_ms,
                                          speech_pad_ms=vad_speech_pad_ms,
                                          return_seconds=True)
    speech_total = sum(e - s for s, e in speech_segments)
    print(f"[diarize] Речевых сегментов: {len(speech_segments)}, "
          f"суммарно: {speech_total:.1f} сек ({100*speech_total/duration:.0f}%)")
    if not speech_segments:
        if n_speakers is not None:
            raise ValueError(f"{audio_path}: requested {n_speakers} speakers, but VAD found no speech")
        print("[diarize] Речь не обнаружена.")
        return ""

    print(f"\n[diarize] === Шаг 3: Извлечение эмбеддингов ===")
    encoder = encoder if encoder is not None else load_model(model_path, device)
    sub_segs, embeddings = extract_embeddings_for_segments(
        encoder, audio, speech_segments, window_sec, step_sec, min_segment_sec)
    if diagnostics is not None:
        diagnostics['n_embedding_windows'] = len(embeddings)
    if n_speakers is not None and not 1 <= n_speakers <= len(embeddings):
        raise ValueError(f"{audio_path}: requested {n_speakers} speakers, "
                         f"but only {len(embeddings)} embedding windows are available")
    if len(embeddings) == 0:
        print("[diarize] Не удалось извлечь эмбеддинги.")
        return ""

    print(f"\n[diarize] === Шаг 4: Кластеризация ===")
    labels = cluster_embeddings(embeddings, method=method, n_speakers=n_speakers,
                                threshold=threshold, max_speakers=max_speakers, seed=seed)
    n_clusters = len(set(labels))
    if n_speakers is not None and n_clusters != n_speakers:
        raise RuntimeError(f"{audio_path}: expected {n_speakers} clusters, got {n_clusters}")
    if diagnostics is not None:
        sizes = numpy.unique(labels, return_counts=True)[1].tolist()
        diagnostics.update(n_clusters_before_postprocess=n_clusters,
                           cluster_sizes=sizes,
                           n_clusters_before_postmerge=n_clusters,
                           n_clusters_after_postmerge=n_clusters,
                           cluster_sizes_before_postmerge=sizes,
                           cluster_sizes_after_postmerge=sizes)

    if centroid_merge and n_speakers is None:
        labels, merge_diagnostics = merge_similar_clusters(
            embeddings, labels, centroid_similarity_threshold, max_small_cluster_size,
            centroid_similarity_margin, return_diagnostics=True)
        if diagnostics is not None:
            sizes_after = numpy.unique(labels, return_counts=True)[1].tolist()
            diagnostics.update(merge_diagnostics,
                               n_clusters_after_postmerge=len(set(labels)),
                               cluster_sizes_after_postmerge=sizes_after)
    elif centroid_merge and diagnostics is not None:
        diagnostics['centroid_merge_skipped_known_count'] = True

    print(f"\n[diarize] === Шаг 5: Постобработка ===")
    labels = smooth_labels(sub_segs, labels, window=3)
    sub_segs = partition_overlapping_windows(sub_segs)
    sub_segs, labels = merge_short_segments(sub_segs, labels, min_duration=0.3)
    sub_segs, labels = merge_consecutive_segments(sub_segs, labels, gap_threshold=0.0)
    if diagnostics is not None:
        diagnostics['n_speakers_after_postprocess'] = len(set(labels))
    print(f"[diarize] Итого спикеров: {len(set(labels))}, сегментов: {len(sub_segs)}")

    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    rttm_str = segments_to_rttm(sub_segs, labels, file_id=file_id)

    if output_rttm:
        with open(output_rttm, "w") as f:
            f.write(rttm_str + "\n")
        print(f"[diarize] RTTM сохранён: {output_rttm}")

    return rttm_str


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="ECAPA-TDNN Speaker Diarization")
    parser.add_argument("audio",             type=str)
    parser.add_argument("--model",           type=str, required=True)
    parser.add_argument("--config")
    parser.add_argument("--n_speakers",      type=int)
    parser.add_argument("--method",          type=str, default=argparse.SUPPRESS,
                        choices=["ahc", "spectral"])
    parser.add_argument("--threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--vad_threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--vad_min_speech_ms", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--vad_min_silence_ms", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--vad_speech_pad_ms", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--output", type=str)
    parser.add_argument("--window_sec", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--step_sec", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--min_segment_sec", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_speakers", type=int, default=argparse.SUPPRESS)
    merge_group = parser.add_mutually_exclusive_group()
    merge_group.add_argument("--centroid_merge", dest="centroid_merge", action="store_true",
                             default=argparse.SUPPRESS)
    merge_group.add_argument("--no_centroid_merge", dest="centroid_merge", action="store_false",
                             default=argparse.SUPPRESS)
    parser.add_argument("--centroid_similarity_threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max_small_cluster_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--centroid_similarity_margin", type=float, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = {
        'method': 'clustering.method', 'threshold': 'clustering.threshold',
        'max_speakers': 'clustering.max_speakers', 'window_sec': 'embedding.window_sec',
        'step_sec': 'embedding.step_sec', 'min_segment_sec': 'embedding.min_segment_sec',
        'vad_threshold': 'vad.threshold', 'vad_min_speech_ms': 'vad.min_speech_ms',
        'vad_min_silence_ms': 'vad.min_silence_ms', 'vad_speech_pad_ms': 'vad.speech_pad_ms',
        'centroid_merge': 'postprocessing.centroid_merge.enabled',
        'centroid_similarity_threshold': 'postprocessing.centroid_merge.similarity_threshold',
        'max_small_cluster_size': 'postprocessing.centroid_merge.max_small_cluster_size',
        'centroid_similarity_margin': 'postprocessing.centroid_merge.similarity_margin',
        'seed': 'seed', 'device': 'device',
    }
    for name, path in paths.items():
        if hasattr(args, name):
            set_config_value(config, path, getattr(args, name))
    validate_config(config)
    effective = argparse.Namespace(audio=args.audio, model=args.model,
        config=args.config, n_speakers=args.n_speakers, output=args.output,
        **pipeline_kwargs(config))
    return effective, config


if __name__ == "__main__":
    args, effective_config = parse_cli_args()
    print("[diarize] Effective configuration:")
    print(json.dumps(effective_config, indent=2))

    rttm = diarize(
        audio_path=args.audio,
        model_path=args.model,
        n_speakers=args.n_speakers,
        method=args.method,
        threshold=args.threshold,
        vad_threshold=args.vad_threshold,
        vad_min_silence_ms=args.vad_min_silence_ms,
        output_rttm=args.output,
        window_sec=args.window_sec, step_sec=args.step_sec,
        device=args.device, seed=args.seed, max_speakers=args.max_speakers,
        centroid_merge=args.centroid_merge,
        centroid_similarity_threshold=args.centroid_similarity_threshold,
        max_small_cluster_size=args.max_small_cluster_size,
        centroid_similarity_margin=args.centroid_similarity_margin,
        min_segment_sec=args.min_segment_sec,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
    )
    print("\n=== RTTM ===")
    print(rttm)
