import os
import sys
import random
import argparse
import io

import numpy
import torch
import soundfile
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ecapa_tdnn import ECAPA_TDNN
from utils.metrics import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf

for _v in ("all_proxy", "ALL_PROXY"):
    if os.environ.get(_v, "").startswith("socks://"):
        os.environ[_v] = os.environ[_v].replace("socks://", "socks5://")

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--n_pairs",    type=int, default=2000)
parser.add_argument("--n_shards",   type=int, default=10)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
numpy.random.seed(args.seed)

print(f"[eval] Загружаем модель: {args.model_path}")
encoder = ECAPA_TDNN(C=1024).cuda()
encoder.eval()
state = torch.load(args.model_path, map_location="cuda")
clean = {}
for k, v in state.items():
    k = k.replace("speaker_encoder.module.", "").replace("speaker_encoder.", "")
    if k in encoder.state_dict():
        clean[k] = v
encoder.load_state_dict(clean, strict=False)
print("[eval] Модель загружена.")

from datasets import load_dataset, concatenate_datasets

print(f"[eval] Загружаем данные ({args.n_shards} шардов)...")
ds_parts = []
for i in range(args.n_shards):
    try:
        part = load_dataset(
            "parquet",
            data_files={"test": f"hf://datasets/vuhoanhuy/dolly-audio-1000h-vietnamese/"
                                 f"data/train-{i:05d}-of-00332.parquet"},
            split="test",
        )
        ds_parts.append(part)
    except Exception as e:
        print(f"Шард {i}: {e}")
ds = concatenate_datasets(ds_parts)
print(f"[eval] Загружено {len(ds)} сэмплов")

speaker_field = "voice_id" if "voice_id" in ds.column_names else "speaker_id"
speaker2utts = {}
for sample in ds:
    speaker2utts.setdefault(sample[speaker_field], []).append(sample["audio"])
speakers = [s for s, v in speaker2utts.items() if len(v) >= 2]
print(f"[eval] Спикеров: {len(speakers)}")


def get_emb(audio_dict):
    if audio_dict.get("bytes") is not None:
        arr, _ = soundfile.read(io.BytesIO(audio_dict["bytes"]))
    elif audio_dict.get("array") is not None:
        arr = numpy.array(audio_dict["array"])
    else:
        raise ValueError(f"Нет аудио: {list(audio_dict.keys())}")
    arr = arr.astype(numpy.float32)
    if len(arr) < 8000:
        arr = numpy.pad(arr, (0, 8000 - len(arr)), "wrap")
    data = torch.FloatTensor(arr).unsqueeze(0).cuda()
    with torch.no_grad():
        emb = F.normalize(encoder.forward(data, aug=False), p=2, dim=1)
    return emb.squeeze(0).cpu()


half = args.n_pairs // 2
pos_pairs, neg_pairs = [], []
while len(pos_pairs) < half:
    spk = random.choice(speakers)
    a, b = random.sample(speaker2utts[spk], 2)
    pos_pairs.append((1, a, b))
while len(neg_pairs) < half:
    sa, sb = random.sample(speakers, 2)
    neg_pairs.append((0, random.choice(speaker2utts[sa]), random.choice(speaker2utts[sb])))

from tqdm import tqdm

scores, labels = [], []
for label, a, b in tqdm(pos_pairs + neg_pairs, desc="Pairs"):
    try:
        scores.append(float(torch.dot(get_emb(a), get_emb(b))))
        labels.append(label)
    except Exception:
        pass

EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

print(f"  Модель : {args.model_path}")
print(f"  Пар    : {len(scores)}")
print(f"  EER    : {EER:.2f}%")
print(f"  minDCF : {minDCF:.4f}")

