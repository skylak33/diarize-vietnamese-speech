import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
