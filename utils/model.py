"""One strict encoder loader for raw, trainer and DataParallel checkpoints."""
import torch
from models.ecapa_tdnn import ECAPA_TDNN
from utils.device import resolve_device


def load_model(model_path: str, device: str = "auto") -> ECAPA_TDNN:
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    clean = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("speaker_encoder.").removeprefix("module.")
        if not key.startswith("speaker_loss."):
            clean[key] = value
    if "conv1.weight" not in clean:
        raise ValueError(f"No ECAPA encoder weights in {model_path}")
    encoder = ECAPA_TDNN(C=clean["conv1.weight"].shape[0])
    encoder.load_state_dict(clean, strict=True)
    return encoder.to(resolve_device(device)).eval()
