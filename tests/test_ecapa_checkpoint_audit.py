import numpy as np
import pytest
import torch

from experiments.audit_ecapa_checkpoint import clean_checkpoint_state, representation_stats


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
