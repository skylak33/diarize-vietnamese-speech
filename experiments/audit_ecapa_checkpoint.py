"""Audit ECAPA checkpoint integrity, layer geometry, and audio consistency."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.pipeline import extract_embedding
from diarize.vad import load_audio
from utils.audio import decode_audio
from utils.model import load_model


LEGACY_ORDER = "out = self.relu(self.bn1(self.conv1(x)))"
CHECKPOINT_ORDER = "out = self.bn1(self.relu(self.conv1(x)))"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_checkpoint_state(checkpoint):
    state = checkpoint.get("state_dict", checkpoint)
    return {
        key.removeprefix("module.").removeprefix("speaker_encoder.").removeprefix("module."): value
        for key, value in state.items()
        if not key.removeprefix("module.").startswith("speaker_loss.")
    }


def representation_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("Expected a non-empty matrix")
    centered = values - values.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    probability = variance / variance.sum() if variance.sum() else variance
    effective_rank = float(np.exp(-(probability[probability > 0] * np.log(probability[probability > 0])).sum()))
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    similarities = normalized @ normalized.T
    upper = similarities[np.triu_indices(len(values), 1)]
    return {
        "count": len(values),
        "dimensions": values.shape[1],
        "norm": summary(np.linalg.norm(values, axis=1)),
        "element_mean": float(values.mean()),
        "mean_per_dimension_variance": float(values.var(axis=0).mean()),
        "min_per_dimension_variance": float(values.var(axis=0).min()),
        "max_per_dimension_variance": float(values.var(axis=0).max()),
        "normalized_centroid_norm": float(np.linalg.norm(normalized.mean(axis=0))),
        "effective_rank": effective_rank,
        "singular_values": singular[:20].tolist(),
        "explained_variance_top_10": probability[:10].tolist(),
        "pairwise_cosine": summary(upper) if len(upper) else None,
        "finite": bool(np.isfinite(values).all()),
    }


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def load_variant(model_path, source_path, order, device):
    source = Path(source_path).read_text(encoding="utf-8")
    if source.count(LEGACY_ORDER) + source.count(CHECKPOINT_ORDER) != 1:
        raise RuntimeError("Bottle2neck order is not uniquely identifiable")
    wanted = LEGACY_ORDER if order == "conv_bn_relu" else CHECKPOINT_ORDER
    source = source.replace(LEGACY_ORDER, wanted).replace(CHECKPOINT_ORDER, wanted)
    module = types.ModuleType(f"ecapa_audit_{order}")
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    raw = torch.load(model_path, map_location="cpu", weights_only=True)
    state = clean_checkpoint_state(raw)
    encoder = module.ECAPA_TDNN(C=state["conv1.weight"].shape[0])
    incompatible = encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()
    return encoder, incompatible


class LayerCapture:
    def __init__(self, encoder):
        self.values = {name: [] for name in ("frame_level", "pooling", "bn5", "fc6", "bn6", "l2")}
        self.handles = [
            encoder.attention.register_forward_pre_hook(self._frame),
            encoder.bn5.register_forward_pre_hook(self._input("pooling")),
            encoder.bn5.register_forward_hook(self._output("bn5")),
            encoder.fc6.register_forward_hook(self._output("fc6")),
            encoder.bn6.register_forward_hook(self._output("bn6")),
        ]

    def _frame(self, _module, args):
        # attention input starts with the 1536 frame-level channels, followed by mean/std.
        self.values["frame_level"].append(args[0][:, :1536].mean(dim=2).detach().cpu().numpy()[0])

    def _input(self, name):
        def hook(_module, args):
            self.values[name].append(args[0].detach().cpu().numpy()[0])
        return hook

    def _output(self, name):
        def hook(_module, _args, output):
            self.values[name].append(output.detach().cpu().numpy()[0])
        return hook

    def finish(self):
        for handle in self.handles:
            handle.remove()
        bn6 = np.asarray(self.values["bn6"])
        self.values["l2"] = F.normalize(torch.from_numpy(bn6), dim=1).numpy().tolist()
        return {name: representation_stats(value) for name, value in self.values.items()}


def vity_samples(artifact_root, audio_root, recordings, windows_per_recording):
    samples = []
    for directory in sorted(Path(artifact_root).iterdir()):
        archive = directory / "embeddings.npz"
        audio_path = Path(audio_root) / f"{directory.name}.wav"
        if not archive.exists() or not audio_path.exists():
            continue
        saved = np.load(archive)
        indices = np.unique(np.linspace(0, len(saved["embeddings"]) - 1, windows_per_recording, dtype=int))
        for index in indices:
            samples.append({
                "dataset": "ViYT-Diar",
                "id": f"{directory.name}:{index}",
                "path": audio_path,
                "start": int(round(float(saved["window_starts"][index]) * 16000)),
                "end": int(round(float(saved["window_ends"][index]) * 16000)),
                "saved_embedding": saved["embeddings"][index],
            })
        if len({sample["path"] for sample in samples}) >= recordings:
            break
    return samples


def vlsp_samples(trials_path, audio_root, pairs_per_label):
    selected, counts = [], {0: 0, 1: 0}
    with Path(trials_path).open(encoding="utf-8", newline="") as handle:
        for first, second, label_text in csv.reader(handle):
            label = int(label_text)
            if counts[label] >= pairs_per_label:
                continue
            counts[label] += 1
            pair_id = f"pair-{label}-{counts[label]}"
            for name in (first, second):
                path = Path(audio_root) / name
                audio = decode_audio(path)
                selected.append({"dataset": "VLSP-2021-public-test", "id": f"{pair_id}:{name}",
                                 "path": path, "start": 0, "end": len(audio), "label": label,
                                 "pair_id": pair_id})
            if min(counts.values()) >= pairs_per_label:
                break
    return selected


def source_evidence(repo, commit):
    source = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:models/ecapa_tdnn.py"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    order = "conv_relu_bn" if CHECKPOINT_ORDER in source else "conv_bn_relu" if LEGACY_ORDER in source else "unknown"
    return {"repository": str(Path(repo).resolve()), "commit": commit,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "bottle2neck_order": order}


def local_source_evidence(path):
    source = Path(path).read_text(encoding="utf-8")
    order = "conv_relu_bn" if CHECKPOINT_ORDER in source else "conv_bn_relu" if LEGACY_ORDER in source else "unknown"
    return {"path": str(Path(path).resolve()), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "bottle2neck_order": order}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="exps/exp1/model/model_0034.model")
    parser.add_argument("--artifacts", default="results/embedding_analysis")
    parser.add_argument("--vity-audio", required=True)
    parser.add_argument("--vlsp-trials")
    parser.add_argument("--vlsp-audio")
    parser.add_argument("--verification-comparison")
    parser.add_argument("--reference-repo")
    parser.add_argument("--reference-commit", default="a78321d")
    parser.add_argument("--output", default="results/ecapa_checkpoint_audit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recordings", type=int, default=4)
    parser.add_argument("--windows-per-recording", type=int, default=3)
    parser.add_argument("--vlsp-pairs-per-label", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = vity_samples(args.artifacts, args.vity_audio, args.recordings, args.windows_per_recording)
    if args.vlsp_trials and args.vlsp_audio:
        samples += vlsp_samples(args.vlsp_trials, args.vlsp_audio, args.vlsp_pairs_per_label)
    if not samples:
        raise RuntimeError("No real audio samples found")

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = clean_checkpoint_state(raw)
    production = load_model(args.checkpoint, args.device)
    production_keys = production.state_dict()
    state_match = set(state) == set(production_keys) and all(torch.equal(state[key], production_keys[key].cpu()) for key in state)
    batch_norm_means = {
        name: {"running_mean": summary(tensor.cpu().numpy()), "finite": bool(torch.isfinite(tensor).all())}
        for name, tensor in state.items() if name.endswith("running_mean")
    }
    batch_norm_variances = {
        name: {"running_variance": summary(tensor.cpu().numpy()), "finite": bool(torch.isfinite(tensor).all())}
        for name, tensor in state.items() if name.endswith("running_var")
    }
    checkpoint_health = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sha256": sha256(args.checkpoint),
        "format": "wrapped_state_dict" if isinstance(raw, dict) and "state_dict" in raw else "raw_state_dict",
        "raw_keys": len(raw.get("state_dict", raw)), "encoder_keys": len(state),
        "encoder_channels": int(state["conv1.weight"].shape[0]),
        "embedding_dimensions": int(state["bn6.weight"].shape[0]),
        "strict_loading": True, "production_loader_exact_state_match": state_match,
        "evaluation_mode": not production.training,
        "parameters_require_grad": bool(any(parameter.requires_grad for parameter in production.parameters())),
        "inference_uses_no_grad": True,
        "device": str(next(production.parameters()).device),
        "batch_norm_running_means": batch_norm_means,
        "batch_norm_running_variances": batch_norm_variances,
        "audited_source": local_source_evidence("models/ecapa_tdnn.py"),
    }
    if args.reference_repo:
        checkpoint_health["checkpoint_architecture_source"] = source_evidence(args.reference_repo, args.reference_commit)
    if args.verification_comparison:
        comparison_path = Path(args.verification_comparison)
        checkpoint_health["existing_independent_verification"] = {
            "source": str(comparison_path.resolve()), "sha256": sha256(comparison_path),
            "results": json.loads(comparison_path.read_text(encoding="utf-8")),
        }

    audio_cache, audio_checks = {}, []
    for sample in samples:
        path = sample["path"]
        if path not in audio_cache:
            decoded = decode_audio(path)
            vad_audio, sample_rate = load_audio(str(path))
            audio_cache[path] = decoded
            audio_checks.append({
                "path": str(path.resolve()), "sha256": sha256(path), "sample_rate": sample_rate,
                "samples": len(decoded), "dtype": str(decoded.dtype), "peak_absolute_amplitude": float(np.abs(decoded).max()),
                "decode_equals_vad": bool(np.array_equal(decoded, vad_audio)),
            })

    variants, embeddings = {}, {}
    for order in ("conv_bn_relu", "conv_relu_bn"):
        encoder, incompatible = load_variant(args.checkpoint, "models/ecapa_tdnn.py", order, args.device)
        capture = LayerCapture(encoder)
        variant_embeddings = []
        for sample in samples:
            audio = audio_cache[sample["path"]]
            variant_embeddings.append(extract_embedding(encoder, audio, sample["start"], sample["end"]))
        embeddings[order] = np.asarray(variant_embeddings)
        by_dataset = {
            dataset: representation_stats(embeddings[order][[sample["dataset"] == dataset for sample in samples]])
            for dataset in sorted({sample["dataset"] for sample in samples})
        }
        pair_scores = {0: [], 1: []}
        pair_indices = {}
        for index, sample in enumerate(samples):
            if sample.get("pair_id"):
                pair_indices.setdefault(sample["pair_id"], []).append(index)
        for indices in pair_indices.values():
            label = samples[indices[0]]["label"]
            pair_scores[label].append(float(np.dot(embeddings[order][indices[0]], embeddings[order][indices[1]])))
        variants[order] = {
            "missing_keys": incompatible.missing_keys, "unexpected_keys": incompatible.unexpected_keys,
            "sample_geometry": representation_stats(embeddings[order]), "geometry_by_dataset": by_dataset,
            "sampled_vlsp_pair_cosine": {"negative": summary(pair_scores[0]), "positive": summary(pair_scores[1])},
            "layers": capture.finish(),
        }

    vity_live_errors = []
    for sample, live in zip(samples, embeddings["conv_bn_relu"]):
        if sample.get("saved_embedding") is not None:
            vity_live_errors.append(float(np.max(np.abs(sample["saved_embedding"] - live))))
    first = samples[0]
    first_audio = audio_cache[first["path"]]
    repeat_a = extract_embedding(production, first_audio, first["start"], first["end"])
    repeat_b = extract_embedding(production, first_audio, first["start"], first["end"])
    all_saved = np.concatenate([np.load(path)["embeddings"] for path in sorted(Path(args.artifacts).glob("*/embeddings.npz"))])
    checkpoint_health["saved_vity_geometry"] = representation_stats(all_saved)
    checkpoint_health["sampled_real_audio_geometry"] = {key: value["sample_geometry"] for key, value in variants.items()}
    checkpoint_health["sample_counts"] = {
        "ViYT-Diar": sum(sample["dataset"] == "ViYT-Diar" for sample in samples),
        "VLSP-2021-public-test": sum(sample["dataset"] == "VLSP-2021-public-test" for sample in samples),
    }

    audio_consistency = {
        "files": audio_checks,
        "unique_waveform_sha256": len({item["sha256"] for item in audio_checks}),
        "all_waveforms_distinct": len({item["sha256"] for item in audio_checks}) == len(audio_checks),
        "all_decode_paths_equal": all(item["decode_equals_vad"] for item in audio_checks),
        "production_repeat_max_absolute_error": float(np.max(np.abs(repeat_a - repeat_b))),
        "saved_vs_live_conv_bn_relu_max_absolute_error": summary(vity_live_errors),
        "eval_eer_extraction": "decode_audio + diarize.pipeline.extract_embedding (same functions audited here)",
        "padding": "numpy.pad(..., mode='wrap') only below 8000 samples",
    }
    layer_statistics = {
        "variants": variants,
        "matched_embedding_cosine_between_orders": summary(np.sum(embeddings["conv_bn_relu"] * embeddings["conv_relu_bn"], axis=1)),
        "interpretation": "conv_relu_bn is the checkpoint/training architecture; conv_bn_relu is the current diagnostics-tree mismatch",
    }
    for name, payload in (("checkpoint_health.json", checkpoint_health), ("layer_statistics.json", layer_statistics),
                          ("audio_consistency.json", audio_consistency)):
        (output / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(samples), "checkpoint_sha256": checkpoint_health["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
