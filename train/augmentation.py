"""Waveform augmentation for training only (MUSAN + RIR)."""
import random
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from utils.audio import decode_audio


class Augmenter:
    def __init__(self, mode='none', musan_path=None, rir_path=None):
        if mode not in {'none','noise','rir','full'}:
            raise ValueError('augmentation must be none, noise, rir, full')
        self.mode = mode
        self.noises = {kind: [] for kind in ('noise','music','speech')}
        if musan_path:
            for path in sorted(Path(musan_path).rglob('*.wav')):
                for kind in self.noises:
                    if kind in path.relative_to(musan_path).parts[:-1]:
                        self.noises[kind].append(path)
                        break
        self.rirs = sorted(Path(rir_path).rglob('*.wav')) if rir_path else []
        if mode in {'noise','full'} and not all(self.noises.values()):
            raise ValueError('MUSAN needs noise/, music/, speech/ WAV files for noise/full augmentation')
        if mode in {'rir','full'} and not self.rirs:
            raise ValueError('RIR augmentation requires WAV impulse responses under --rir_path')

    def add_noise(self, audio, kind):
        count = random.randint(3,8) if kind == 'speech' else 1
        noise = np.zeros_like(audio)
        for path in random.choices(self.noises[kind], k=count):
            signal = decode_audio(path)
            if len(signal) < len(audio):
                signal = np.pad(signal, (0,len(audio)-len(signal)), mode='wrap')
            start = random.randint(0, len(signal)-len(audio))
            noise += signal[start:start+len(audio)]
        snr = random.uniform(*{'noise':(0,15),'music':(5,15),'speech':(13,20)}[kind])
        scale = np.sqrt((np.mean(audio**2)+1e-8) / (np.mean(noise**2)+1e-8) / 10**(snr/10))
        return audio + noise * scale

    def add_rir(self, audio):
        rir = decode_audio(random.choice(self.rirs))
        norm = np.linalg.norm(rir)
        if norm <= 0:
            raise ValueError('RIR has zero energy')
        return fftconvolve(audio, rir / norm)[:len(audio)].astype(np.float32)

    def __call__(self, audio):
        if self.mode == 'none':
            return audio
        # Keep a clean branch, as in standard speaker-recognition augmentation.
        choices = ['clean']
        if self.mode in {'noise','full'}:
            choices += ['noise','music','speech']
        if self.mode in {'rir','full'}:
            choices += ['rir']
        choice = random.choice(choices)
        if choice == 'rir':
            return self.add_rir(audio)
        if choice != 'clean':
            return self.add_noise(audio, choice).astype(np.float32)
        return audio


def crop_training_audio(audio, num_frames, augmenter):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio):
        raise ValueError('Training audio must be nonempty mono')
    length = num_frames * 160 + 240
    if len(audio) < length:
        audio = np.pad(audio, (0,length-len(audio)), mode='wrap')
    start = random.randint(0,len(audio)-length)
    return augmenter(audio[start:start+length])
