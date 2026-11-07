"""Cache natural-benchmark embeddings once for CPU-only diagnostics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diarize.embedding_diagnostics import write_artifacts
from diarize.pipeline import extract_embeddings_for_segments
from diarize.vad import get_speech_segments, load_audio
from experiments.run_natural_vietnamese_benchmark import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_CONFIG,
    ROOT,
    _reference,
    load_config,
    load_recordings,
)
from experiments.run_synthetic_diarization_benchmark import (
    _embedding_digest,
    reference_speech_regions,
)
from utils.model import load_model
from utils.provenance import environment, sha256
from utils.seed import set_seed


DEFAULT_RESULTS = ROOT / "results/natural_vietnamese_benchmark/full_evaluation/recording_results.json"
DEFAULT_OUTPUT = ROOT / "results/partition_quality_stability/embedding_cache_verified"


def expected_digests(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    digests = {}
    for row in rows:
        key = row["mode"], row["recording_id"]
        digest = row["embedding_sha256"]
        if key in digests and digests[key] != digest:
            raise ValueError(f"Inconsistent benchmark embedding digest for {key}")
        digests[key] = digest
    return digests


def cached_digest(path):
    with np.load(path) as arrays:
        windows = list(zip(arrays["window_starts"].tolist(), arrays["window_ends"].tolist()))
        return _embedding_digest(windows, arrays["embeddings"]), len(windows)


def run(args):
    config_path = Path(args.config).resolve()
    annotations = Path(args.annotation_manifest).resolve()
    results_path = Path(args.recording_results).resolve()
    benchmark_manifest_path = results_path.parent / "manifest.json"
    output = Path(args.output_dir).resolve()
    config = load_config(config_path)
    records, exclusions = load_recordings(annotations, args.dataset)
    expected = expected_digests(results_path)
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    if benchmark_manifest.get("status") != "complete":
        raise ValueError("Natural benchmark manifest is not complete")
    modes = tuple(config["evaluation"]["modes"])
    missing = [(mode, row["recording_id"]) for row in records for mode in modes
               if (mode, row["recording_id"]) not in expected]
    if missing:
        raise ValueError(f"Benchmark results lack embedding digests for {missing}")

    checkpoint = Path(config["checkpoint"]["path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if sha256(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint SHA256 does not match the frozen config")

    output.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    encoder = None
    cached_rows = []
    for record in records:
        reference = _reference(record)
        audio = sample_rate = None
        for mode in modes:
            artifact_dir = output / mode / record["recording_id"]
            cache_path = artifact_dir / "embeddings.npz"
            if cache_path.is_file():
                digest, window_count = cached_digest(cache_path)
            else:
                if encoder is None:
                    encoder = load_model(str(checkpoint), args.device)
                if audio is None:
                    audio, sample_rate = load_audio(str(record["audio_path"]))
                regions = (reference_speech_regions(reference)
                           if mode == "clustering_only" else get_speech_segments(
                               str(record["audio_path"]), threshold=config["vad"]["threshold"],
                               min_speech_ms=config["vad"]["min_speech_ms"],
                               min_silence_ms=config["vad"]["min_silence_ms"],
                               speech_pad_ms=config["vad"]["speech_pad_ms"],
                               return_seconds=True))
                windows, embeddings = extract_embeddings_for_segments(
                    encoder, audio, regions, config["embedding"]["window_sec"],
                    config["embedding"]["step_sec"], config["embedding"]["min_segment_sec"])
                digest, window_count = _embedding_digest(windows, embeddings), len(windows)
                if digest != expected[(mode, record["recording_id"])]:
                    raise RuntimeError(f"{record['recording_id']}/{mode}: embedding digest mismatch")
                metadata = {
                    "schema_version": 1,
                    "recording_id": record["recording_id"],
                    "mode": mode,
                    "dataset_revision": record["dataset_revision"],
                    "audio_sha256": record["audio_sha256"],
                    "reference_sha256": record["rttm_sha256"],
                    "checkpoint_sha256": config["checkpoint"]["sha256"],
                    "encoder_source_sha256": sha256(ROOT / "models/ecapa_tdnn.py"),
                    "embedding_sha256": digest,
                    "seed": config["seed"],
                    "vad": config["vad"],
                    "embedding": config["embedding"],
                }
                write_artifacts(
                    artifact_dir, embeddings, windows,
                    {"schema_version": 1, "recording_id": record["recording_id"], "stages": {}},
                    metadata)
                if not cache_path.is_file():
                    raise RuntimeError(f"{record['recording_id']}/{mode}: cache was not written")
            if digest != expected[(mode, record["recording_id"])]:
                raise RuntimeError(f"{record['recording_id']}/{mode}: cached digest mismatch")
            cached_rows.append({"recording_id": record["recording_id"], "mode": mode,
                                "embedding_sha256": digest, "embedding_windows": window_count,
                                "cache_path": str(cache_path)})

    analysis_environment = environment()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "analysis_environment": analysis_environment,
        "source_benchmark_git_commit": benchmark_manifest["environment"]["git_commit"],
        "source_benchmark_manifest_sha256": sha256(benchmark_manifest_path),
        "config_sha256": sha256(config_path),
        "annotation_manifest_sha256": sha256(annotations),
        "recording_results_sha256": sha256(results_path),
        "checkpoint_sha256": sha256(checkpoint),
        "encoder_source_sha256": sha256(ROOT / "models/ecapa_tdnn.py"),
        "dataset_revisions": sorted({row["dataset_revision"] for row in records}),
        "audio_sha256": benchmark_manifest["audio_sha256"],
        "reference_sha256": benchmark_manifest["reference_sha256"],
        "effective_configurations": benchmark_manifest["plan"]["configurations"],
        "scoring_protocol": benchmark_manifest["evaluation"],
        "excluded_recordings": exclusions,
        "caches": cached_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--annotation-manifest", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--recording-results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dataset", default="datahive")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    manifest = run(parse_args(argv))
    print(json.dumps({"cached_recording_modes": len(manifest["caches"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
