from pathlib import Path
import os
import random
import torch
from utils.audio import decode_audio
from train.augmentation import Augmenter, crop_training_audio
from torch.utils.data import IterableDataset

METADATA_PATH = '/kaggle/input/datasets/dungnasa10/vietnamese-speaker-verification/new_metadata/metadata/train/training_metadata.txt'
AUDIO_BASE = '/kaggle/input/datasets/dungnasa10/vietnamese-speaker-verification/data'


class VLSPLoader(IterableDataset):
    def __init__(self, num_frames, n_class=1015, musan_path=None, rir_path=None, augmentation=None, metadata_path=None, audio_base=None, exclude_speakers=None, **kwargs):
        self.num_frames = num_frames
        self.n_class = n_class
        mode = augmentation or ('full' if musan_path and rir_path else 'noise' if musan_path else 'rir' if rir_path else 'none')
        self.augmenter = Augmenter(mode, musan_path, rir_path)
        metadata_path = metadata_path or METADATA_PATH
        audio_base = audio_base or AUDIO_BASE
        excluded = set(Path(exclude_speakers).read_text().splitlines()) if exclude_speakers else set()
        print('[loader_vlsp] Загружаем metadata...', flush=True)
        raw = []
        speaker_set = set()
        with open(metadata_path, encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    raise ValueError(f'{metadata_path}:{number}: expected speaker<TAB>relative_path')
                if parts[0] not in excluded:
                    raw.append((parts[0], parts[1]))
                    speaker_set.add(parts[0])

        if not raw:
            raise ValueError("No training rows; expected speaker<TAB>relative_path")
        if len(speaker_set) > self.n_class:
            raise ValueError("n_class is smaller than the number of training speakers")
        speaker_map = {s: i for i, s in enumerate(sorted(speaker_set))}
        self.speaker_map = speaker_map
        self.data = [
            (os.path.join(audio_base, path), speaker_map[spk])
            for spk, path in raw
            if speaker_map[spk] < self.n_class
        ]
        print(f'[loader_vlsp] Спикеров: {len(speaker_map)}, сэмплов: {len(self.data)}', flush=True)

    def _process(self, audio):
        return crop_training_audio(audio, self.num_frames, self.augmenter)

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError('VLSPLoader requires num_workers=0 to preserve sample ordering')
        data = self.data[:]
        random.shuffle(data)
        for path, label in data:
            yield torch.from_numpy(self._process(decode_audio(path))), label


train_loader = VLSPLoader
