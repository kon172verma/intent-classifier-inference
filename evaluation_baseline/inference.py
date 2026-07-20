"""Core inference pass with TTFT measurement for the HF Transformers baseline."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from evaluation_baseline.cache import kv_cache_bytes, kv_cache_tokens
from evaluation_lib.config import MAX_NEW_TOKENS
from evaluation_lib.device import synchronize
from evaluation_lib.system_info import peak_gpu_memory_mb, reset_peak_gpu_memory


class TTFTCapture(LogitsProcessor):
    """Capture time-to-first-token inside ``model.generate()``.

    ``LogitsProcessor.__call__`` is invoked once per decode step.  The first
    call occurs immediately after the prefill forward pass, so the elapsed
    time at that point equals the prefill latency / TTFT.
    """

    def __init__(self, device: str) -> None:
        self._device = device
        self._start: float | None = None
        self._fired: bool = False
        self.ttft_ms: float | None = None

    def arm(self) -> None:
        """Start the clock.  Call this immediately before ``model.generate()``."""
        synchronize(self._device)
        self._start = time.perf_counter()
        self._fired = False
        self.ttft_ms = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not self._fired and self._start is not None:
            synchronize(self._device)
            self.ttft_ms = (time.perf_counter() - self._start) * 1000
            self._fired = True
        return scores


def run_inference(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    mode: str,
    device: str,
    ttft_capture: TTFTCapture,
    past_key_values: Any = None,
    attention_mask: torch.Tensor | None = None,
) -> dict:
    """Run one inference pass and return per-example timing and output data.

    Parameters
    ----------
    input_ids:
        Tokenised prompt (or the dynamic suffix when *past_key_values* provided).
    mode:
        ``"no_cache"`` | ``"kv_cache"`` | ``"prefix_cache"``
    past_key_values:
        Pre-computed prefix KV cache (``prefix_cache`` mode only).
    attention_mask:
        Full mask covering both the cached prefix and the current tokens.
    """
    use_cache = mode != "no_cache"

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": use_cache,
        "do_sample": False,
        "logits_processor": LogitsProcessorList([ttft_capture]),
        "return_dict_in_generate": True,
    }
    if past_key_values is not None:
        generate_kwargs["past_key_values"] = past_key_values
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    # Snapshot the cached prefix length *before* generate() mutates the cache
    # in-place (DynamicLayer.update() appends new KV tensors as decoding
    # proceeds, so reading this after generate() would double-count tokens).
    cached_prefix_tokens = kv_cache_tokens(past_key_values)

    reset_peak_gpu_memory(device)
    ttft_capture.arm()
    synchronize(device)
    t_start = time.perf_counter()

    with torch.no_grad():
        result = model.generate(input_ids, **generate_kwargs)

    synchronize(device)
    t_end = time.perf_counter()
    gpu_peak_mb = peak_gpu_memory_mb(device)

    e2e_ms = (t_end - t_start) * 1000
    ttft_ms = ttft_capture.ttft_ms if ttft_capture.ttft_ms is not None else e2e_ms
    prefill_ms = ttft_ms
    decode_ms = max(0.0, e2e_ms - prefill_ms)

    # Effective context length includes cached prefix tokens.
    n_input = input_ids.shape[1] + cached_prefix_tokens
    n_generated = result.sequences.shape[1] - input_ids.shape[1]

    prefill_tok_per_sec = (n_input / prefill_ms * 1000) if prefill_ms > 0 else None
    decode_tok_per_sec = (
        ((n_generated - 1) / decode_ms * 1000)
        if decode_ms > 0 and n_generated > 1
        else None
    )

    final_cache = getattr(result, "past_key_values", None)
    kv_bytes = kv_cache_bytes(final_cache) if use_cache else 0

    new_ids = result.sequences[0, input_ids.shape[1] :].tolist()
    generated_text = tokenizer.decode(
        [int(i) for i in new_ids], skip_special_tokens=True
    )

    return {
        "generated_text": generated_text.strip(),
        "n_input_tokens": n_input,
        "n_generated_tokens": n_generated,
        "prefill_latency_ms": round(prefill_ms, 3),
        "decode_latency_ms": round(decode_ms, 3),
        "e2e_latency_ms": round(e2e_ms, 3),
        "ttft_ms": round(ttft_ms, 3),
        "prefill_tok_per_sec": (
            round(prefill_tok_per_sec, 2) if prefill_tok_per_sec is not None else None
        ),
        "decode_tok_per_sec": (
            round(decode_tok_per_sec, 2) if decode_tok_per_sec is not None else None
        ),
        "kv_cache_bytes": kv_bytes,
        "peak_gpu_mb": gpu_peak_mb,
    }
