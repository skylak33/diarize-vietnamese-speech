import os
import random
import numpy
import torch
import soundfile
from scipy import signal
from torch.utils.data import IterableDataset

METADATA_PATH = '/kaggle/input/datasets/dungnasa10/vietnamese-speaker-verification/new_metadata/metadata/train/training_metadata.txt'
AUDIO_BASE = '/kaggle/input/datasets/dungnasa10/vietnamese-speaker-verification/data'


class VLSPLoader(IterableDataset):
    def __init__(self, num_frames, n_class=1015, musan_path=None, rir_path=None, **kwargs):
        self.num_frames = num_frames
        self.n_class = n_class
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        self.rir_files = []

        import glob as _glob
        if musan_path and os.path.exists(musan_path):
            for f in _glob.glob(os.path.join(musan_path, '*/*/*/*.wav')):
                self.noiselist.setdefault(f.split('/')[-4], []).append(f)
        if rir_path and os.path.exists(rir_path):
            self.rir_files = _glob.glob(os.path.join(rir_path, '*/*/*.wav'))

        print('[loader_vlsp] Загружаем metadata...', flush=True)
        raw = []
        speaker_set = set()
        with open(METADATA_PATH) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    raw.append((parts[0], parts[1]))
                    speaker_set.add(parts[0])

        speaker_map = {s: i for i, s in enumerate(sorted(speaker_set))}
        self.data = [
            (os.path.join(AUDIO_BASE, path), speaker_map[spk])
            for spk, path in raw
            if speaker_map[spk] < self.n_class
        ]
        print(f'[loader_vlsp] Спикеров: {len(speaker_map)}, сэмплов: {len(self.data)}', flush=True)

    def _process(self, audio):
        audio = numpy.array(audio, dtype=numpy.float32)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            audio = numpy.pad(audio, (0, length - audio.shape[0]), 'wrap')
        start = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = numpy.stack([audio[start:start + length]])
        augtype = random.randint(0, 5)
        if augtype == 1 and self.rir_files:
            audio = self.add_rev(audio)
        elif augtype == 2 and 'speech' in self.noiselist:
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3 and 'music' in self.noiselist:
            audio = self.add_noise(audio, 'music')
        elif augtype == 4 and 'noise' in self.noiselist:
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5 and 'speech' in self.noiselist and 'music' in self.noiselist:
            audio = self.add_noise(self.add_noise(audio, 'speech'), 'music')
        return audio[0]

    def __iter__(self):
        data = self.data[:]
        random.shuffle(data)
        for path, label in data:
            try:
                audio, _ = soundfile.read(path)
                yield torch.FloatTensor(self._process(audio)), label
            except Exception:
                continue

    def add_rev(self, audio):
        rir, _ = soundfile.read(random.choice(self.rir_files))
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio**2) + 1e-4)
        length = self.num_frames * 160 + 240
        noises = []
        for noise in random.sample(self.noiselist[noisecat],
                                   random.randint(*self.numnoise[noisecat])):
            na, _ = soundfile.read(noise)
            if na.shape[0] <= length:
                na = numpy.pad(na, (0, length - na.shape[0]), 'wrap')
            start = numpy.int64(random.random() * (na.shape[0] - length))
            na = numpy.stack([na[start:start + length]])
            noise_db = 10 * numpy.log10(numpy.mean(na**2) + 1e-4)
            snr = random.uniform(*self.noisesnr[noisecat])
            noises.append(numpy.sqrt(10**((clean_db - noise_db - snr) / 10)) * na)
        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio


train_loader = VLSPLoader
