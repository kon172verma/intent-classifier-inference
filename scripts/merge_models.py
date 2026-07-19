#!/usr/bin/env python3
"""Download base models and LoRA adapters, then export merged checkpoints.

This script expects a Hugging Face token in the project .env file as one of:
- HF_TOKEN
- HUGGINGFACE_TOKEN
- HUGGINGFACEHUB_API_TOKEN
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MERGE_JOBS = [
    {
        "name": "intent-classifier-qwen3-0.6b_C_1k_merged",
        "base_model": "Qwen/Qwen3-0.6B",
        "adapter_repo": "kon172verma/intent-classifier",
        "adapter_subfolder": "LoRA/qwen3-0.6b_C_1k",
    },
    {
        "name": "intent-classifier-llama3.2-1b_C_1k_merged",
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "adapter_repo": "kon172verma/intent-classifier",
        "adapter_subfolder": "LoRA/llama3.2-1b_C_1k",
    },
]


def _read_hf_token() -> str:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Add HF_TOKEN (or HUGGINGFACE_TOKEN) to .env"
        )
    return token


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def merge_job(job: dict[str, str], models_root: Path, token: str) -> None:
    output_dir = models_root / job["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Loading tokenizer: {job['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        job["base_model"],
        token=token,
        trust_remote_code=True,
    )

    print(f"[+] Loading base model: {job['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        job["base_model"],
        token=token,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    print(f"[+] Loading adapter: {job['adapter_repo']} ({job['adapter_subfolder']})")
    peft_model = PeftModel.from_pretrained(
        base_model,
        job["adapter_repo"],
        subfolder=job["adapter_subfolder"],
        token=token,
    )

    print("[+] Merging adapter with base model via merge_and_unload()")
    merged_model = peft_model.merge_and_unload()

    print(f"[+] Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"[ok] Completed: {job['name']}")


def main() -> None:
    root = _project_root()
    load_dotenv(root / ".env")
    token = _read_hf_token()

    models_root = root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    print(f"[*] Models output directory: {models_root}")
    for job in MERGE_JOBS:
        merge_job(job, models_root, token)

    print("\n[done] All requested merged models are available under ./models")


if __name__ == "__main__":
    main()
