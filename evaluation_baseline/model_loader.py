"""Load HF Transformers models for baseline evaluation."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation_lib.config import MODEL_DISPLAY_NAMES, MODEL_PATHS


def load_model_and_tokenizer(model_key: str, device: str) -> tuple[Any, Any]:
    """Load the tokenizer and model for *model_key*, placing the model on *device*.

    Notes
    -----
    The model is first loaded to CPU and then moved to *device*.  Loading
    directly with ``device_map="mps"`` causes a segfault on some transformers
    builds; the two-step approach is safe on all backends.
    """
    model_path = MODEL_PATHS[model_key]
    print(f"[model] Loading tokenizer from {model_path}")
    tokenizer: Any = AutoTokenizer.from_pretrained(str(model_path))

    dtype = torch.float16
    print(
        f"[model] Loading {MODEL_DISPLAY_NAMES[model_key]}"
        f" → device={device}, dtype={dtype}"
    )

    model: Any = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype)
    if device != "cpu":
        model = model.to(device)  # type: ignore[arg-type]
    model.eval()
    print("[model] Ready.\n")
    return model, tokenizer
