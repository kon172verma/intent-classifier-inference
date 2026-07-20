"""Device resolution and synchronisation helpers."""

from __future__ import annotations

import torch


def resolve_device(device_arg: str) -> str:
    """Return the concrete device string for a given ``--device`` argument."""
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    """Block until all pending operations on *device* have completed."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
