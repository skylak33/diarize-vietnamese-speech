import numpy as np
import pytest
import torch

from experiments.audit_ecapa_checkpoint import clean_checkpoint_state, representation_stats, summary
from experiments.audit_embedding_statistics import center_and_normalize


def test_clean_checkpoint_state_keeps_encoder_only():
    state = clean_checkpoint_state({"state_dict": {
        "module.speaker_encoder.conv1.weight": torch.ones(1),
        "module.speaker_loss.weight": torch.zeros(1),
    }})
    assert list(state) == ["conv1.weight"]


def test_representation_stats_reports_collapsed_rank():
    stats = representation_stats(np.ones((3, 2)))
    assert stats["effective_rank"] == 1.0
    assert stats["normalized_centroid_norm"] == pytest.approx(1.0)


def test_optional_sample_summary_accepts_no_vlsp_scores():
    assert summary([]) is None


def test_center_and_normalize_removes_fixed_common_direction():
    embeddings = np.array([[2., 1.], [2., -1.]])
    centered = center_and_normalize(embeddings, np.array([2., 0.]))
    assert centered.tolist() == [[0., 1.], [0., -1.]]


def test_center_and_normalize_rejects_zero_residual():
    with pytest.raises(ValueError, match="zero residual"):
        center_and_normalize(np.ones((2, 3)), np.ones(3))
