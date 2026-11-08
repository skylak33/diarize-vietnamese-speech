"""Frozen compatibility and provenance invariants for partition research."""

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from diarize.cluster import cluster_embeddings
from diarize.config import DEFAULT_CONFIG, load_config
from experiments.analyze_ahc_merge_quality import trace_reference_merges
from experiments.analyze_partition_quality import partition_quality
from utils.provenance import sha256


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT_SHA256 = "58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325"


def _coassignment(labels):
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]


def test_production_defaults_and_frozen_benchmark_config_are_unchanged():
    expected = {
        "sample_rate": 16000,
        "embedding": {"window_sec": 1.5, "step_sec": 0.75, "min_segment_sec": 0.5},
        "vad": {"threshold": 0.5, "min_speech_ms": 250, "min_silence_ms": 100,
                "speech_pad_ms": 30},
        "clustering": {"method": "ahc", "count_estimator": "backend_default",
                       "threshold": 0.65, "max_speakers": 10},
        "postprocessing": {"centroid_merge": {"enabled": False,
                           "similarity_threshold": 0.80, "max_small_cluster_size": 4,
                           "similarity_margin": 0.05}},
        "evaluation": {"collar": 0.25, "skip_overlap": False},
        "seed": 42,
        "device": "auto",
    }
    configurations = [DEFAULT_CONFIG, load_config(), yaml.safe_load(
        (ROOT / "configs/diarization_default.yaml").read_text(encoding="utf-8"))]
    for configuration in configurations:
        unchanged = copy.deepcopy(configuration)
        experimental = unchanged['postprocessing'].pop('confidence_aware_merge')
        assert experimental['enabled'] is False
        assert unchanged == expected

    benchmark = json.loads(
        (ROOT / "configs/natural_vietnamese_benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert benchmark["vad"] == expected["vad"]
    assert benchmark["embedding"] == expected["embedding"]
    assert [(row["name"], row["method"], row["count_estimator"], row["threshold"])
            for row in benchmark["configurations"]] == [
        ("fixed_ahc_065", "ahc", None, 0.65),
        ("fixed_ahc_075", "ahc", None, 0.75),
        ("legacy_adaptive_ahc_075", "ahc", "adaptive_ahc", 0.75),
        ("robust_adaptive_ahc_075", "ahc", "adaptive_ahc_robust", 0.75),
        ("spectral", "spectral", None, 0.75),
        ("nme_sc", "nme_sc", None, 0.75),
    ]


def test_checkpoint_and_bottle2neck_operation_order_are_unchanged():
    checkpoint = ROOT / "exps/exp1/model/model_0034.model"
    assert checkpoint.is_file(), "validated ECAPA checkpoint is required for provenance tests"
    assert sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256

    source = (ROOT / "models/ecapa_tdnn.py").read_text(encoding="utf-8")
    expected = "out = self.bn1(self.relu(self.conv1(x)))"
    legacy = "out = self.relu(self.bn1(self.conv1(x)))"
    assert source.count(expected) == 1
    assert legacy not in source


def test_fixed_legacy_and_robust_ahc_are_deterministic_and_do_not_mutate_embeddings():
    embeddings = np.array([
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
    ])
    original = embeddings.copy()
    expected = np.equal.outer(np.repeat(np.arange(3), 3), np.repeat(np.arange(3), 3))

    for estimator in (None, "adaptive_ahc", "adaptive_ahc_robust"):
        first = cluster_embeddings(embeddings, method="ahc", threshold=0.75,
                                   max_speakers=10, count_estimator=estimator)
        second = cluster_embeddings(embeddings, method="ahc", threshold=0.75,
                                    max_speakers=10, count_estimator=estimator)
        np.testing.assert_array_equal(_coassignment(first), expected)
        np.testing.assert_array_equal(_coassignment(second), expected)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(embeddings, original)


def test_partition_and_merge_diagnostics_do_not_mutate_inputs():
    references = ["A", "A", "B", "B"]
    predicted = [0, 0, 1, 1]
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    snapshots = copy.deepcopy(references), copy.deepcopy(predicted), durations.copy()
    partition_quality(references, predicted, durations)
    assert references == snapshots[0] and predicted == snapshots[1]
    np.testing.assert_array_equal(durations, snapshots[2])

    children = [[0, 1], [2, 3], [4, 5]]
    distances = [0.1, 0.2, 0.9]
    merge_inputs = copy.deepcopy(children), copy.deepcopy(distances), copy.deepcopy(references)
    trace_reference_merges(children, distances, references, selected_n_clusters=2)
    assert (children, distances, references) == merge_inputs


def test_frozen_benchmark_json_is_valid_json_and_has_no_label_driven_tuning_fields():
    path = ROOT / "configs/natural_vietnamese_benchmark.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert {"seed", "checkpoint", "vad", "embedding", "evaluation", "configurations"} <= raw.keys()
    forbidden = {"oracle_threshold", "tuned_threshold", "best_der_threshold"}
    assert not forbidden.intersection(raw)
    assert not any(forbidden.intersection(row) for row in raw["configurations"])
