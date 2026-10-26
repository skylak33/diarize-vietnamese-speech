"""Shared mono/16 kHz decoding without evaluation augmentation."""
import io
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def decode_audio(source) -> np.ndarray:
    if isinstance(source, dict):
        if source.get("array") is not None:
            audio, sr = np.asarray(source["array"]), int(source["sampling_rate"])
        else:
            source = io.BytesIO(source["bytes"]) if source.get("bytes") else source["path"]
            audio, sr = sf.read(source, dtype="float32")
    else:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        audio, sr = sf.read(source, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // divisor, sr // divisor)
    return np.asarray(audio, dtype=np.float32)
