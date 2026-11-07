import json

import numpy as np
import pytest

from experiments import cache_natural_diagnostic_embeddings as cache
from experiments.run_synthetic_diarization_benchmark import _embedding_digest


def test_expected_digests_require_shared_frontend_consistency(tmp_path):
    path = tmp_path / "results.json"
    rows = [{"mode": "full_pipeline", "recording_id": "a",
             "embedding_sha256": "one"},
            {"mode": "full_pipeline", "recording_id": "a",
             "embedding_sha256": "two"}]
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent"):
        cache.expected_digests(path)


def test_cached_digest_matches_benchmark_digest(tmp_path):
    embeddings = np.array([[1., 0.], [0., 1.]])
    starts, ends = np.array([0., 1.]), np.array([1., 2.])
    path = tmp_path / "embeddings.npz"
    np.savez(path, embeddings=embeddings, window_starts=starts, window_ends=ends)
    digest, count = cache.cached_digest(path)
    assert count == 2
    assert digest == _embedding_digest([(0., 1.), (1., 2.)], embeddings)
