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
import soundfile
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ecapa_tdnn import ECAPA_TDNN
from diarize.vad import load_audio, get_speech_segments
from diarize.cluster import cluster_embeddings, merge_short_segments, smooth_labels

WINDOW_SEC  = 1.5
STEP_SEC    = 0.75
MIN_SEG_SEC = 0.5
SAMPLE_RATE = 16000


def load_model(model_path: str) -> ECAPA_TDNN:
    print(f"[diarize] Загружаем модель: {model_path}")
    encoder = ECAPA_TDNN(C=1024).cuda()
    encoder.eval()
    state = torch.load(model_path, map_location="cuda")
    clean = {
        k.replace("speaker_encoder.module.", "").replace("speaker_encoder.", ""): v
        for k, v in state.items()
    }
    missing = [k for k in encoder.state_dict() if k not in clean]
    if missing:
        print(f"[diarize] Предупреждение: {len(missing)} ключей не загружены")
    encoder.load_state_dict(clean, strict=False)
    print("[diarize] Модель загружена.")
    return encoder


def extract_embedding(encoder: ECAPA_TDNN, audio: numpy.ndarray,
                      start: int, end: int) -> numpy.ndarray:
    chunk = audio[start:end].astype(numpy.float32)
    min_len = 8000
    if len(chunk) < min_len:
        chunk = numpy.pad(chunk, (0, min_len - len(chunk)), "wrap")
    with torch.no_grad():
        emb = encoder.forward(torch.FloatTensor(chunk).unsqueeze(0).cuda(), aug=False)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy()


def extract_embeddings_for_segments(
    encoder: ECAPA_TDNN,
    audio: numpy.ndarray,
    segments: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], numpy.ndarray]:
    sub_segments = []
    embeddings = []
    window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    step_samples = int(STEP_SEC * SAMPLE_RATE)
    min_samples = int(MIN_SEG_SEC * SAMPLE_RATE)

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
            if pos < e and (e - pos) >= min_samples:
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
    merged_segs = [segments[0]]
    merged_labels = [labels[0]]
    for i in range(1, len(segments)):
        gap = segments[i][0] - merged_segs[-1][1]
        if labels[i] == merged_labels[-1] and gap < gap_threshold:
            merged_segs[-1] = (merged_segs[-1][0], segments[i][1])
        else:
            merged_segs.append(segments[i])
            merged_labels.append(labels[i])
    return merged_segs, numpy.array(merged_labels)


def diarize(
    audio_path: str,
    model_path: str,
    n_speakers: int = None,
    method: str = "ahc",
    threshold: float = 0.65,
    vad_threshold: float = 0.5,
    output_rttm: str = None,
) -> str:
    print(f"\n[diarize] === Шаг 1: Загрузка аудио ===")
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr
    print(f"[diarize] Файл: {audio_path}, длительность: {duration:.1f} сек")

    print(f"\n[diarize] === Шаг 2: VAD ===")
    speech_segments = get_speech_segments(audio_path, threshold=vad_threshold, return_seconds=True)
    speech_total = sum(e - s for s, e in speech_segments)
    print(f"[diarize] Речевых сегментов: {len(speech_segments)}, "
          f"суммарно: {speech_total:.1f} сек ({100*speech_total/duration:.0f}%)")
    if not speech_segments:
        print("[diarize] Речь не обнаружена.")
        return ""

    print(f"\n[diarize] === Шаг 3: Извлечение эмбеддингов ===")
    encoder = load_model(model_path)
    sub_segs, embeddings = extract_embeddings_for_segments(encoder, audio, speech_segments)
    if len(embeddings) == 0:
        print("[diarize] Не удалось извлечь эмбеддинги.")
        return ""

    print(f"\n[diarize] === Шаг 4: Кластеризация ===")
    labels = cluster_embeddings(embeddings, method=method, n_speakers=n_speakers,
                                threshold=threshold, max_speakers=10)

    print(f"\n[diarize] === Шаг 5: Постобработка ===")
    labels = smooth_labels(sub_segs, labels, window=3)
    sub_segs, labels = merge_short_segments(sub_segs, labels, min_duration=0.3)
    sub_segs, labels = merge_consecutive_segments(sub_segs, labels, gap_threshold=0.3)
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
    args = parser.parse_args()

    rttm = diarize(
        audio_path=args.audio,
        model_path=args.model,
        n_speakers=args.n_speakers,
        method=args.method,
        threshold=args.threshold,
        vad_threshold=args.vad_threshold,
        output_rttm=args.output,
    )
    print("\n=== RTTM ===")
    print(rttm)
