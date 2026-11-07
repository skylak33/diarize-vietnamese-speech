"""Generate leakage-safe clean Vietnamese conversations from audited local speech."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.audio import decode_audio


GENERATOR_VERSION = "clean-v1"
SPLITS = ("development", "validation", "holdout")


class UnusableSpeakerError(ValueError):
    def __init__(self, speaker_id):
        self.speaker_id = speaker_id
        super().__init__(f"No source with enough continuous speech for {speaker_id}")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _speaker_bucket(speaker_id, seed, modulus):
    value = hashlib.sha256(f"{seed}:{speaker_id}".encode()).digest()
    return int.from_bytes(value[:8], "big") % modulus


def assign_source_split(row, seed):
    """Map official/local source splits to the benchmark without speaker leakage."""
    dataset, source_split = row["dataset"], row["split"]
    if dataset == "dnu_vispeech":
        mapping = {"train": "development", "validation": "validation", "test": "holdout"}
        if source_split not in mapping:
            raise ValueError(f"Unknown DNU split: {source_split}")
        return mapping[source_split]
    if dataset == "vietnamese_asr":
        if source_split == "test":
            return "holdout"
        if source_split != "train":
            raise ValueError(f"Unknown VIVOS split: {source_split}")
        return "development" if _speaker_bucket(row["speaker_id"], seed, 3) < 2 else "validation"
    if dataset == "viespeaker":
        bucket = _speaker_bucket(row["speaker_id"], seed, 4)
        return "development" if bucket < 2 else "validation" if bucket == 2 else "holdout"
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"seed", "scenario", "target_counts", "speaker_counts",
                "duration_seconds", "turn_seconds", "silence_seconds",
                "sample_rate_hz", "noise", "overlap", "transformations", "vad"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Config is missing fields: {missing}")
    if (config["scenario"] != "clean" or config["noise"] or config["overlap"]
            or config["transformations"]):
        raise ValueError("The primary generator accepts only clean, non-overlap audio")
    if config["sample_rate_hz"] != 16000:
        raise ValueError("sample_rate_hz must be 16000")
    if set(config["target_counts"]) != set(SPLITS):
        raise ValueError(f"target_counts must contain {SPLITS}")
    if (not config["speaker_counts"] or min(config["speaker_counts"]) < 1
            or max(config["speaker_counts"]) > 5):
        raise ValueError("speaker_counts must be within 1..5")
    for field in ("duration_seconds", "turn_seconds", "silence_seconds"):
        if not 0 <= config[field]["min"] <= config[field]["max"]:
            raise ValueError(f"Invalid {field} range")
    if config["duration_seconds"]["min"] < 30:
        raise ValueError("Primary conversations must be at least 30 seconds")
    return config


class SourceReader:
    """Decode selected sources lazily and verify the bytes actually used."""

    def __init__(self, cache_size=32):
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.row_groups = OrderedDict()

    def _remember(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
        return value

    def read(self, row):
        key = row["utterance_id"] if "utterance_id" in row else row["source_path"]
        if key in self.cache:
            self.cache.move_to_end(key)
            result = self.cache[key]
            declared = row.get("content_sha256")
            if declared and declared.lower() != result[1]:
                raise ValueError(f"Source checksum mismatch: {row['source_path']}")
            return result
        if row["source_type"] == "audio_file":
            path = Path(row["source_path"])
            raw = path.read_bytes()
            digest = _sha256_bytes(raw)
            declared = row.get("content_sha256")
            if declared and declared.lower() != digest:
                raise ValueError(f"Source checksum mismatch: {path}")
            result = (decode_audio(path), digest)
        elif row["source_type"] == "parquet_row":
            group_key = (row["source_path"], int(row["row_group"]))
            if group_key in self.row_groups:
                table = self.row_groups[group_key]
                self.row_groups.move_to_end(group_key)
            else:
                table = pq.ParquetFile(group_key[0]).read_row_group(group_key[1], columns=["audio"])
                self._remember(self.row_groups, group_key, table)
            audio = table.column("audio")[int(row["row_index_in_group"])].as_py()
            raw = audio.get("bytes")
            if not raw:
                raise ValueError(f"Embedded audio bytes are missing: {row['source_locator']}")
            result = (decode_audio(audio), _sha256_bytes(raw))
        else:
            raise ValueError(f"Unsupported source_type: {row['source_type']}")
        return self._remember(self.cache, key, result)


class SileroVad:
    """One Silero model per generation process; returns sample-index regions."""

    def __init__(self, config):
        import torch
        from silero_vad import load_silero_vad

        self.torch = torch
        self.model = load_silero_vad()
        self.config = config

    def detect(self, audio):
        from silero_vad import get_speech_timestamps

        rows = get_speech_timestamps(
            self.torch.from_numpy(np.asarray(audio, dtype=np.float32)), self.model,
            sampling_rate=16000, threshold=self.config["threshold"],
            min_speech_duration_ms=self.config["min_speech_ms"],
            min_silence_duration_ms=self.config["min_silence_ms"],
            speech_pad_ms=self.config["speech_pad_ms"], return_seconds=False)
        return [(int(row["start"]), int(row["end"])) for row in rows]


def _load_pools(manifest_path, seed, maximum):
    pools = defaultdict(lambda: defaultdict(list))
    speaker_splits = {}
    seen_utterances = set()
    with Path(manifest_path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            utterance_id = row.get("utterance_id")
            speaker_id = row.get("speaker_id")
            if not utterance_id or not speaker_id:
                raise ValueError(f"Manifest row {number} lacks utterance_id/speaker_id")
            if utterance_id in seen_utterances:
                raise ValueError(f"Duplicate utterance_id: {utterance_id}")
            seen_utterances.add(utterance_id)
            split = assign_source_split(row, seed)
            previous = speaker_splits.setdefault(speaker_id, split)
            if previous != split:
                raise ValueError(f"Speaker crosses splits: {speaker_id}")
            candidates = pools[split][speaker_id]
            if len(candidates) < maximum:
                candidates.append(row)
    return {split: dict(pools[split]) for split in SPLITS}


def _pick_clip(speaker_rows, desired_samples, minimum_samples, reader, vad, cursor):
    for offset in range(len(speaker_rows)):
        index = (cursor + offset) % len(speaker_rows)
        row = speaker_rows[index]
        audio, digest = reader.read(row)
        regions = vad.detect(audio)
        eligible = [(start, end) for start, end in regions if end - start >= minimum_samples]
        if eligible:
            start, end = max(eligible, key=lambda region: region[1] - region[0])
            samples = min(desired_samples, end - start)
            return np.asarray(audio[start:start + samples], dtype=np.float32), row, digest, index + 1
    raise UnusableSpeakerError(speaker_rows[0]["speaker_id"])


def _select_speakers(pools, speaker_count, conversation_index, rng):
    by_dataset = defaultdict(list)
    for speaker, rows in pools.items():
        by_dataset[rows[0]["dataset"]].append(speaker)
    datasets = [name for name in ("dnu_vispeech", "vietnamese_asr", "viespeaker")
                if by_dataset[name]]
    if not datasets or len(pools) < speaker_count:
        raise ValueError(f"Only {len(pools)} usable speakers remain, needs {speaker_count}")
    selected = []
    for offset in range(speaker_count):
        dataset = datasets[(conversation_index - 1 + offset) % len(datasets)]
        candidates = sorted(set(by_dataset[dataset]) - set(selected))
        if not candidates:
            candidates = sorted(set(pools) - set(selected))
        selected.append(rng.choice(candidates))
    return selected


def _conversation(config, split, index, speaker_count, pools, rng, reader, vad):
    selected = _select_speakers(pools, speaker_count, index, rng)
    minimum_turns = max(4, speaker_count * 2)
    order = []
    while len(order) < minimum_turns:
        cycle = selected[:]
        rng.shuffle(cycle)
        order.extend(cycle)

    duration_min = float(config["duration_seconds"]["min"])
    duration_max = float(config["duration_seconds"]["max"])
    turn_min = float(config["turn_seconds"]["min"])
    turn_max = float(config["turn_seconds"]["max"])
    silence_min = float(config["silence_seconds"]["min"])
    silence_max = float(config["silence_seconds"]["max"])
    pieces, turns, inserted_silence = [], [], []
    cursors = Counter()
    current = 0
    turn_index = 0
    while True:
        if turn_index >= len(order):
            cycle = selected[:]
            rng.shuffle(cycle)
            order.extend(cycle)
        speaker = order[turn_index]
        silence = 0. if not pieces else rng.uniform(silence_min, silence_max)
        remaining = duration_max - current / 16000 - silence
        if remaining < turn_min:
            break
        mandatory_after = max(0, minimum_turns - turn_index - 1)
        reserved = mandatory_after * (turn_min + silence_max)
        upper = min(turn_max, remaining - reserved)
        if upper < turn_min:
            raise ValueError("Conversation bounds cannot fit two turns per speaker")
        desired = rng.uniform(turn_min, upper)
        desired_samples = int(round(desired * 16000))
        clip, source, digest, next_cursor = _pick_clip(
            pools[speaker], desired_samples, int(round(turn_min * 16000)),
            reader, vad, cursors[speaker])
        cursors[speaker] = next_cursor
        if silence:
            silence_samples = int(round(silence * 16000))
            pieces.append(np.zeros(silence_samples, dtype=np.float32))
            inserted_silence.append({"start": current / 16000,
                                     "end": (current + silence_samples) / 16000})
            current += silence_samples
        start = current / 16000
        pieces.append(clip)
        current += len(clip)
        end = current / 16000
        turns.append({"start": start, "end": end, "speaker_id": speaker,
                      "source_utterance_id": source["utterance_id"],
                      "source_recording_id": source["source_recording_id"],
                      "source_hash": digest, "source_dataset": source["dataset"]})
        turn_index += 1
        counts = Counter(turn["speaker_id"] for turn in turns)
        if (current / 16000 >= duration_min and len(turns) >= minimum_turns
                and all(counts[speaker] >= 2 for speaker in selected)):
            break
    duration = current / 16000
    counts = Counter(turn["speaker_id"] for turn in turns)
    if (not duration_min <= duration <= duration_max
            or any(counts[speaker] < 2 for speaker in selected)):
        raise ValueError(f"Could not satisfy duration bounds: {duration}")
    return np.concatenate(pieces), selected, turns, inserted_silence


def _write_conversation(output, split, index, audio, speakers, turns, silences,
                        config, seed):
    conversation_id = f"{split}_{index:06d}"
    directory = output / split
    directory.mkdir(parents=True, exist_ok=True)
    wav_path = directory / f"{conversation_id}.wav"
    rttm_path = directory / f"{conversation_id}.rttm"
    json_path = directory / f"{conversation_id}.json"
    sf.write(wav_path, np.clip(audio, -1., 1.), 16000, subtype="PCM_16")
    rttm_path.write_text("".join(
        f"SPEAKER {conversation_id} 1 {turn['start']:.6f} "
        f"{turn['end'] - turn['start']:.6f} <NA> <NA> {turn['speaker_id']} <NA> <NA>\n"
        for turn in turns), encoding="utf-8")
    payload = {
        "schema_version": 1, "conversation_id": conversation_id, "split": split,
        "scenario": "clean", "seed": seed, "speaker_count": len(speakers),
        "speaker_ids": speakers, "audio_path": wav_path.name, "rttm_path": rttm_path.name,
        "duration_seconds": len(audio) / 16000, "turns": turns,
        "inserted_silence": silences, "overlap_intervals": [], "audio_transformations": [],
        "source_utterance_ids": sorted({turn["source_utterance_id"] for turn in turns}),
        "source_hashes": sorted({turn["source_hash"] for turn in turns}),
        "source_datasets": sorted({turn["source_dataset"] for turn in turns}),
        "generator_version": GENERATOR_VERSION,
        "speech_annotation": {"kind": "Silero-VAD-assisted source regions",
                              "is_manual_ground_truth": False, **config["vad"]},
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _validate_generated(records):
    speakers, utterances, hashes = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in records:
        split = row["split"]
        speakers[split].update(row["speaker_ids"])
        utterances[split].update(row["source_utterance_ids"])
        hashes[split].update(row["source_hashes"])
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            for name, values in (("speaker", speakers), ("utterance", utterances),
                                 ("content hash", hashes)):
                overlap = values[left] & values[right]
                if overlap:
                    raise ValueError(f"Cross-split {name} leakage: {sorted(overlap)[:3]}")
    return {"speaker_disjoint": True, "source_utterance_disjoint": True,
            "source_content_disjoint": True}


def generate(config, manifest_path, output_dir, target_counts=None, speaker_counts=None, vad=None):
    output = Path(output_dir)
    if output.exists() and any(output.rglob("*")):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    target_counts = dict(target_counts or config["target_counts"])
    speaker_counts = list(speaker_counts or config["speaker_counts"])
    pools = _load_pools(manifest_path, config["seed"], config.get("max_sources_per_speaker", 12))
    reader = SourceReader()
    vad = vad or SileroVad(config["vad"])
    records = []
    for split_index, split in enumerate(SPLITS):
        rng = random.Random(config["seed"] + 1009 * split_index)
        for index in range(1, target_counts[split] + 1):
            count = speaker_counts[(index - 1) % len(speaker_counts)]
            while True:
                try:
                    audio, speakers, turns, silences = _conversation(
                        config, split, index, count, pools[split], rng, reader, vad)
                    break
                except UnusableSpeakerError as error:
                    pools[split].pop(error.speaker_id, None)
            records.append(_write_conversation(
                output, split, index, audio, speakers, turns, silences,
                config, config["seed"] + 1009 * split_index))
    checks = _validate_generated(records)
    counts = dict(Counter(row["split"] for row in records))
    split_root = output.parent / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        split_rows = [row for row in records if row["split"] == split]
        (split_root / f"{split}.json").write_text(json.dumps({
            "schema_version": 1, "seed": config["seed"], "split": split,
            "conversation_ids": [row["conversation_id"] for row in split_rows],
            "speaker_ids": sorted({speaker for row in split_rows for speaker in row["speaker_ids"]}),
            "source_utterance_ids": sorted({source for row in split_rows
                                             for source in row["source_utterance_ids"]}),
            "checks": checks,
        }, indent=2), encoding="utf-8")
    summary = {"schema_version": 1, "generator_version": GENERATOR_VERSION,
               "seed": config["seed"], "counts": counts, "checks": checks,
               "speaker_count_distribution": dict(Counter(row["speaker_count"] for row in records)),
               "total_duration_seconds": sum(row["duration_seconds"] for row in records)}
    (output / "generation_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic_vietnamese_benchmark.json")
    parser.add_argument("--manifest", default=(
        "results/synthetic_vietnamese_benchmark/dataset_audit/utterance_manifest.jsonl"))
    parser.add_argument("--output-dir", default="results/synthetic_vietnamese_benchmark/generated")
    parser.add_argument("--smoke", action="store_true", help="Generate one one-speaker item per split")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    target = {split: 1 for split in SPLITS} if args.smoke else config["target_counts"]
    counts = [1] if args.smoke else config["speaker_counts"]
    if args.dry_run:
        print(json.dumps({"config": config, "target_counts": target,
                          "speaker_counts": counts}, indent=2))
        return 0
    summary = generate(config, args.manifest, args.output_dir, target, counts)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
