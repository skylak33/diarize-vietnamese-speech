import io
import os
import random
import numpy
import torch
import soundfile
import requests

for _var in ("all_proxy", "ALL_PROXY"):
    if os.environ.get(_var, "").startswith("socks://"):
        os.environ[_var] = os.environ[_var].replace("socks://", "socks5://")
_socks_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("ALL_PROXY")
if _socks_proxy and _socks_proxy.startswith("socks"):
    for _lc in ("https_proxy", "http_proxy"):
        os.environ[_lc] = _socks_proxy

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import signal
from torch.utils.data import IterableDataset

import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import HfFileSystem

DATASET_REPO = "dolly-vn/dolly-audio-1000h-vietnamese"
_HF_BASE = "https://huggingface.co/datasets"

_PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("ALL_PROXY") or None
_SESSION = requests.Session()
if _PROXY:
    _SESSION.proxies = {"http": _PROXY, "https": _PROXY}


def _get_parquet_urls() -> list[str]:
    print("[loader_hf] Получение списка шардов...", flush=True)
    fs = HfFileSystem()
    files = fs.glob(f"datasets/{DATASET_REPO}/data/train-*.parquet")
    urls = [f"hf://datasets/{DATASET_REPO}/data/" + p.split("/data/")[-1] for p in files]
    print(f"[loader_hf] Найдено шардов: {len(urls)}", flush=True)
    return urls


def _load_parquet_shard(url: str) -> pa.Table:
    import time
    path = url.replace("hf://datasets/", "")
    repo, rest = path.split("/data/", 1)
    direct_url = f"{_HF_BASE}/{repo}/resolve/main/data/{rest}"
    for attempt in range(6):
        try:
            resp = _SESSION.get(direct_url, timeout=300)
            resp.raise_for_status()
            return pq.read_table(io.BytesIO(resp.content))
        except Exception as e:
            if attempt == 5:
                raise
            wait = 2 ** attempt
            print(f"[loader_hf] Ошибка шарда (попытка {attempt+1}/6): {e} — повтор через {wait}с",
                  flush=True)
            time.sleep(wait)


def _decode_audio(row_audio) -> numpy.ndarray:
    if isinstance(row_audio, dict):
        if "bytes" in row_audio and row_audio["bytes"]:
            return soundfile.read(io.BytesIO(row_audio["bytes"]), dtype="float32")[0]
        elif "path" in row_audio:
            return soundfile.read(row_audio["path"], dtype="float32")[0]
    elif isinstance(row_audio, (bytes, bytearray)):
        return soundfile.read(io.BytesIO(row_audio), dtype="float32")[0]
    raise ValueError(f"Неизвестный формат аудио: {type(row_audio)}")


class TrainLoader(IterableDataset):
    def __init__(self, num_frames, musan_path=None, rir_path=None,
                 buffer_size=100, prefetch_shards=2, decode_workers=4,
                 n_class=None, max_shards=None, **kwargs):
        self.num_frames = num_frames
        self.buffer_size = buffer_size
        self.prefetch_shards = prefetch_shards
        self.decode_workers = decode_workers
        self.n_class = n_class
        self.max_shards = max_shards

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

        self.speaker_map = {}
        self.speaker_counter = 0

    def _get_label(self, speaker_id: str) -> int:
        if speaker_id not in self.speaker_map:
            if self.n_class is not None and self.speaker_counter >= self.n_class:
                return -1
            self.speaker_map[speaker_id] = self.speaker_counter
            self.speaker_counter += 1
        return self.speaker_map[speaker_id]

    def _process(self, audio_array: numpy.ndarray) -> numpy.ndarray:
        audio = numpy.array(audio_array, dtype=numpy.float32)
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

    def _iter_shards_async(self, shard_urls: list[str]):
        total = len(shard_urls)
        done = 0
        with ThreadPoolExecutor(max_workers=self.prefetch_shards) as pool:
            futures = {}
            url_iter = iter(shard_urls)
            for _ in range(self.prefetch_shards):
                url = next(url_iter, None)
                if url is None:
                    break
                futures[pool.submit(_load_parquet_shard, url)] = url

            while futures:
                done_future = next(as_completed(futures))
                url = futures.pop(done_future)
                try:
                    table = done_future.result()
                except Exception as e:
                    print(f"[loader_hf] Пропускаем шард {url.split('/')[-1]}: {e}", flush=True)
                    next_url = next(url_iter, None)
                    if next_url:
                        futures[pool.submit(_load_parquet_shard, next_url)] = next_url
                    continue
                done += 1
                print(f"[loader_hf] Шард {done}/{total}: {url.split('/')[-1]} ({len(table)} строк)",
                      flush=True)
                next_url = next(url_iter, None)
                if next_url:
                    futures[pool.submit(_load_parquet_shard, next_url)] = next_url
                yield table

    def _decode_row(self, row: dict):
        raw_audio = row.get("audio") or row.get("audio_bytes")
        speaker_id = str(row.get("voice_id") or row.get("speaker_id") or "spk_0")
        return _decode_audio(raw_audio), self._get_label(speaker_id)

    def __iter__(self):
        shard_urls = _get_parquet_urls()
        total_shards = len(shard_urls)
        random.shuffle(shard_urls)
        if self.max_shards is not None:
            shard_urls = shard_urls[:self.max_shards]
            print(f"[loader_hf] Используем {len(shard_urls)} шардов из {total_shards}", flush=True)

        buffer = []
        for table in self._iter_shards_async(shard_urls):
            rows = table.to_pylist()
            random.shuffle(rows)
            with ThreadPoolExecutor(max_workers=self.decode_workers) as pool:
                for f in [pool.submit(self._decode_row, row) for row in rows]:
                    try:
                        audio_arr, label = f.result()
                        if label == -1:
                            continue
                        buffer.append((torch.FloatTensor(self._process(audio_arr)), label))
                    except Exception as e:
                        print(f"[loader_hf] skip sample: {e}")

                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)
                        yield from buffer
                        buffer = []

        if buffer:
            random.shuffle(buffer)
            yield from buffer

    def add_rev(self, audio):
        rir, _ = soundfile.read(random.choice(self.rir_files))
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio**2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        length = self.num_frames * 160 + 240
        noises = []
        for noise in random.sample(self.noiselist[noisecat],
                                   random.randint(numnoise[0], numnoise[1])):
            na, _ = soundfile.read(noise)
            if na.shape[0] <= length:
                na = numpy.pad(na, (0, length - na.shape[0]), 'wrap')
            start = numpy.int64(random.random() * (na.shape[0] - length))
            na = numpy.stack([na[start:start + length]])
            noise_db = 10 * numpy.log10(numpy.mean(na**2) + 1e-4)
            snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10**((clean_db - noise_db - snr) / 10)) * na)
        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio


# backward-compat alias
train_loader = TrainLoader
