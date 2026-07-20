"""Host system information helpers (memory, OS)."""

from __future__ import annotations

import platform
import resource
from typing import Any

import torch


def peak_ram_mb() -> float:
    """Return peak RSS in MB.

    Accounts for the platform difference:
    - macOS reports ``ru_maxrss`` in bytes.
    - Linux reports ``ru_maxrss`` in kilobytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def model_weights_mb(model: Any) -> float:
    """Return total bytes occupied by model parameters and buffers in MB.

    Device-agnostic: works for CPU, MPS, and CUDA by counting ``numel * element_size``
    across all parameters and persistent buffers.
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return round(total / (1024 * 1024), 2)


def reset_peak_gpu_memory(device: str) -> None:
    """Reset peak-memory tracking counters (CUDA only; no-op on MPS/CPU)."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb(device: str) -> float | None:
    """Return peak GPU memory allocated since the last reset.

    - **CUDA**: ``torch.cuda.max_memory_allocated()`` — true peak including weights,
      activations, and KV cache captured since the last ``reset_peak_gpu_memory()``
      call.
    - **MPS**: ``torch.mps.current_allocated_memory()`` — *current* allocated bytes
      (MPS has no peak-tracking API).  Measured after generate() completes, so
      activations are freed; reflects weights + remaining KV cache.
    - **CPU**: returns ``None``.
    """
    if device == "cuda":
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    if device == "mps":
        return round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
    return None
