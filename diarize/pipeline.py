"""
pipeline.py — Полный пайплайн диаризации говорящего.
    audio.wav → VAD → сегментация → ECAPA-TDNN → кластеризация → RTTM
"""

import os
import sys
import argparse
from pathlib import Path

import numpy
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ecapa_tdnn import ECAPA_TDNN
from utils.model import load_model
from utils.seed import set_seed
from diarize.vad import load_audio, get_speech_segments
from diarize.cluster import cluster_embeddings, merge_short_segments, smooth_labels

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
) -> str:
    set_seed(seed)
    if diagnostics is not None:
        diagnostics.update(requested_n_speakers=n_speakers, n_embedding_windows=0,
                           n_clusters_before_postprocess=0, n_speakers_after_postprocess=0,
                           cluster_sizes=[])
    if not 0 < step_sec <= window_sec or not 0 <= vad_threshold <= 1:
        raise ValueError("Require 0 < step_sec <= window_sec and 0 <= vad_threshold <= 1")
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
    speech_segments = get_speech_segments(audio_path, threshold=vad_threshold, return_seconds=True)
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
        diagnostics.update(n_clusters_before_postprocess=n_clusters,
                           cluster_sizes=numpy.unique(labels, return_counts=True)[1].tolist())

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECAPA-TDNN Speaker Diarization")
    parser.add_argument("audio",             type=str)
    parser.add_argument("--model",           type=str, required=True)
    parser.add_argument("--n_speakers",      type=int, default=None)
    parser.add_argument("--method",          type=str, default="ahc",
                        choices=["ahc", "spectral"])
    parser.add_argument("--threshold",       type=float, default=0.65)
    parser.add_argument("--vad_threshold",   type=float, default=0.5)
    parser.add_argument("--output",          type=str, default=None)
    parser.add_argument("--window_sec", type=float, default=1.5)
    parser.add_argument("--step_sec", type=float, default=0.75)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_speakers", type=int, default=10)
    args = parser.parse_args()

    rttm = diarize(
        audio_path=args.audio,
        model_path=args.model,
        n_speakers=args.n_speakers,
        method=args.method,
        threshold=args.threshold,
        vad_threshold=args.vad_threshold,
        output_rttm=args.output,
        window_sec=args.window_sec, step_sec=args.step_sec,
        device=args.device, seed=args.seed, max_speakers=args.max_speakers,
    )
    print("\n=== RTTM ===")
    print(rttm)
