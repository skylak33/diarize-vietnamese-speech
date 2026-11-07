"""Checkpoint shapes cannot detect a swapped ReLU/BatchNorm operation."""

import torch

from models.ecapa_tdnn import ECAPA_TDNN


def test_res2_blocks_feed_rectified_convolution_to_batchnorm():
    torch.manual_seed(42)
    model = ECAPA_TDNN(C=8).eval()
    x = torch.randn(2, 8, 31)
    for block in (model.layer1, model.layer2, model.layer3):
        observed = []
        hook = block.bn1.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                convolution = block.conv1(x)
                assert (convolution < 0).any() and (convolution > 0).any()
                output = block(x)
            torch.testing.assert_close(observed[0], convolution.relu())
            assert output.shape == x.shape and torch.isfinite(output).all()
        finally:
            hook.remove()
