"""Reusable, CPU-only diagnostics for diarization embedding windows."""

import json
from pathlib import Path

import numpy as np


def validate_embeddings(embeddings, window_starts=None, window_ends=None):
    """Return validated embedding arrays and optional window timestamps."""
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be a finite 2-D array")
    if len(embeddings) and (np.linalg.norm(embeddings, axis=1) == 0).any():
        raise ValueError("embeddings must be nonzero vectors")
    if window_starts is None:
        return embeddings, None, None
    starts, ends = np.asarray(window_starts, dtype=float), np.asarray(window_ends, dtype=float)
    if starts.ndim != 1 or ends.ndim != 1 or len(starts) != len(embeddings) or len(ends) != len(embeddings):
        raise ValueError("window timestamps must match embeddings")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or (ends <= starts).any():
        raise ValueError("window timestamps must be finite increasing intervals")
    return embeddings, starts, ends


def write_artifacts(output_dir, embeddings, segments, trace, config):
    """Write the stable diagnostic artifact schema without rerunning inference."""
    embeddings, starts, ends = validate_embeddings(embeddings, [s for s, _ in segments], [e for _, e in segments])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "embeddings.npz", embeddings=embeddings, window_starts=starts, window_ends=ends)
    (output_dir / "pipeline_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
