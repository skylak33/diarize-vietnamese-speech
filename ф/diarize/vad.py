"""
vad.py — Voice Activity Detection на основе silero-vad.
"""

import torch
import numpy
import soundfile

SAMPLE_RATE       = 16000   
THRESHOLD         = 0.5     # порог уверенности VAD (0.0–1.0)
MIN_SPEECH_MS     = 250     # минимальная длина речевого сегмента 
MIN_SILENCE_MS    = 100     # минимальная пауза между сегментами 
SPEECH_PAD_MS     = 30      # отступ вокруг каждого сегмента 


def load_audio(path: str) -> tuple[numpy.ndarray, int]:
    """Загрузить аудио и привести к моно 16kHz."""
    audio, sr = soundfile.read(path, dtype="float32")

    # стерео → моно
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # ресемплинг если нужно
    if sr != SAMPLE_RATE:
        import scipy.signal as ssig
        audio = ssig.resample_poly(audio, SAMPLE_RATE, sr).astype("float32")
        sr = SAMPLE_RATE

    return audio, sr


def get_speech_segments(
    path: str,
    threshold: float     = THRESHOLD,
    min_speech_ms: int   = MIN_SPEECH_MS,
    min_silence_ms: int  = MIN_SILENCE_MS,
    speech_pad_ms: int   = SPEECH_PAD_MS,
    return_seconds: bool = True,
) -> list[tuple[float, float]]:
    """
    Найти речевые сегменты в аудиофайле.
    """
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    audio, sr = load_audio(path)
    audio_tensor = torch.from_numpy(audio)

    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate    = sr,
        threshold        = threshold,
        min_speech_duration_ms  = min_speech_ms,
        min_silence_duration_ms = min_silence_ms,
        speech_pad_ms    = speech_pad_ms,
        return_seconds   = return_seconds,
    )

    segments = [(t["start"], t["end"]) for t in timestamps]
    return segments


def print_segments(segments: list[tuple[float, float]]) -> None:
    """Вывести сегменты в читаемом формате."""
    total = sum(e - s for s, e in segments)
    print(f"Найдено сегментов: {len(segments)}, суммарно речи: {total:.1f} сек")
    for i, (s, e) in enumerate(segments):
        print(f"  [{i+1:3d}] {s:7.3f}s — {e:7.3f}s  ({e-s:.3f}s)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Использование: python vad.py audio.wav")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Обрабатываем: {path}")
    segments = get_speech_segments(path)
    print_segments(segments)