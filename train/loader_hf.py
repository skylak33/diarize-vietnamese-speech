import io
import os
import random
import torch
import requests

for _var in ("all_proxy", "ALL_PROXY"):
    if os.environ.get(_var, "").startswith("socks://"):
        os.environ[_var] = os.environ[_var].replace("socks://", "socks5://")
_socks_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("ALL_PROXY")
if _socks_proxy and _socks_proxy.startswith("socks"):
    for _lc in ("https_proxy", "http_proxy"):
        os.environ[_lc] = _socks_proxy

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from utils.audio import decode_audio
from train.augmentation import Augmenter, crop_training_audio
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
        except (requests.RequestException, OSError, pa.ArrowInvalid) as e:
            if attempt == 5:
                raise
            wait = 2 ** attempt
            print(f"[loader_hf] Ошибка шарда (попытка {attempt+1}/6): {e} — повтор через {wait}с",
                  flush=True)
            time.sleep(wait)


def _decode_audio(source):
    return decode_audio(source)


class TrainLoader(IterableDataset):
    def __init__(self, num_frames, musan_path=None, rir_path=None,
                 buffer_size=100, prefetch_shards=2, decode_workers=4,
                 n_class=None, max_shards=None, augmentation=None, exclude_speakers=None, **kwargs):
        self.num_frames = num_frames
        self.buffer_size = buffer_size
        self.prefetch_shards = prefetch_shards
        self.decode_workers = decode_workers
        self.n_class = n_class
        self.max_shards = max_shards

        mode = augmentation or ('full' if musan_path and rir_path else 'noise' if musan_path else 'rir' if rir_path else 'none')
        self.augmenter = Augmenter(mode, musan_path, rir_path)
        self.excluded = set(Path(exclude_speakers).read_text().splitlines()) if exclude_speakers else set()
        self.speaker_map = {}
        self.speaker_counter = 0

    def _get_label(self, speaker_id: str) -> int:
        if speaker_id not in self.speaker_map:
            if self.n_class is not None and self.speaker_counter >= self.n_class:
                raise ValueError("n_class is smaller than observed training speaker count")
            self.speaker_map[speaker_id] = self.speaker_counter
            self.speaker_counter += 1
        return self.speaker_map[speaker_id]

    def _process(self, audio_array):
        return crop_training_audio(audio_array, self.num_frames, self.augmenter)

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
                done_future = next(iter(futures))  # submission order, independent of network timing
                url = futures.pop(done_future)
                table = done_future.result()  # fail instead of changing the training corpus
                done += 1
                print(f"[loader_hf] Шард {done}/{total}: {url.split('/')[-1]} ({len(table)} строк)",
                      flush=True)
                next_url = next(url_iter, None)
                if next_url:
                    futures[pool.submit(_load_parquet_shard, next_url)] = next_url
                yield table

    def _decode_row(self, row: dict):
        raw_audio = row.get('audio') or row.get('audio_bytes')
        speaker = row.get('voice_id', row.get('speaker_id'))
        if speaker is None:
            raise ValueError('Missing voice_id/speaker_id')
        return _decode_audio(raw_audio), str(speaker)

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError('Dolly loader requires num_workers=0')
        shard_urls = sorted(_get_parquet_urls())
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
                    audio_arr, speaker = f.result()
                    if speaker in self.excluded:
                        continue
                    label = self._get_label(speaker)
                    buffer.append((torch.from_numpy(self._process(audio_arr)), label))

                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)
                        yield from buffer
                        buffer = []

        if buffer:
            random.shuffle(buffer)
            yield from buffer


# backward-compat alias
train_loader = TrainLoader
