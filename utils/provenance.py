"""Small reproducibility records; no audio or dataset downloads."""
import hashlib
import importlib.metadata
import platform
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    versions = {}
    for package in ("torch", "torchaudio", "numpy", "scipy", "scikit-learn", "pyannote.metrics", "pyannote.core", "silero-vad"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False)
        commit, dirty = result.stdout.strip(), bool(status.stdout.strip())
    except FileNotFoundError:
        commit, dirty = None, None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                git_commit=commit, git_dirty=dirty)
