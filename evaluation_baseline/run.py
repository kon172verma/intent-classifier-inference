#!/usr/bin/env python3
"""
Baseline inference benchmark: HF Transformers + PyTorch FP16.

Modes
------
no_cache     — No KV cache.  Each decode step re-processes all tokens.
kv_cache     — Standard decode-time KV cache (default transformers behaviour).
prefix_cache — Static system-prompt prefix pre-computed once; only the dynamic
               per-example suffix (tools + user request) is processed at
               inference time.  Cache creation time is reported separately.

Usage
------
    # Activate the project venv first: source .venv/bin/activate
    python evaluation_baseline/run.py --model qwen3 --mode no_cache
    python evaluation_baseline/run.py --model llama3 --mode kv_cache
    python evaluation_baseline/run.py --model qwen3 --mode prefix_cache --device mps

Output
------
    evaluation_baseline/results/<model>_<device>_<mode>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers

# Ensure the repo root is on sys.path so that evaluation_lib and
# evaluation_baseline are importable regardless of invocation directory.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_baseline.cache import (  # noqa: E402
    clone_cache,
    compute_prefix_cache,
    ingest_tools_prefix,
    kv_cache_bytes,
)
from evaluation_baseline.inference import (  # noqa: E402
    TTFTCapture,
    find_tools_query_boundary,
    run_inference,
)
from evaluation_baseline.model_loader import load_model_and_tokenizer  # noqa: E402
from evaluation_lib.config import (  # noqa: E402
    DATASET_DEFAULT,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    WARMUP_EXAMPLES,
)
from evaluation_lib.device import resolve_device  # noqa: E402
from evaluation_lib.metrics import aggregate_metrics, compute_quality  # noqa: E402
from evaluation_lib.system_info import model_weights_mb  # noqa: E402
from evaluation_lib.output_parser import extract_predicted_tool  # noqa: E402
from evaluation_lib.prompt import (  # noqa: E402
    build_full_prompt,
    build_system_prefix_text,
    build_tools_only_prompt,
)

_RESULTS_DIR = _REPO_ROOT / "evaluation_baseline" / "results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline inference benchmark (HF Transformers, FP16)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_PATHS),
        required=True,
        help="Model to benchmark",
    )
    p.add_argument(
        "--mode",
        choices=["no_cache", "kv_cache", "prefix_cache"],
        default="no_cache",
        help="Caching mode",
    )
    p.add_argument(
        "--device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
        help="Compute device",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
        help="Path to dataset JSON file",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_RESULTS_DIR,
        help="Directory to write JSON results",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_EXAMPLES,
        help="Number of warmup examples excluded from measurements",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    mode = args.mode

    print("=== Baseline Benchmark ===")
    print(f"  model   : {args.model} ({MODEL_DISPLAY_NAMES[args.model]})")
    print(f"  mode    : {mode}")
    print(f"  device  : {device}")
    print(f"  dataset : {args.dataset}")
    print(f"  warmup  : {args.warmup} examples\n")

    with open(args.dataset) as f:
        dataset: list[dict] = json.load(f)
    print(f"[data] Loaded {len(dataset)} examples from {args.dataset.name}\n")

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    weights_mb = model_weights_mb(model)
    print(f"[model] Parameter + buffer size: {weights_mb:.1f} MB (FP16 on {device})\n")

    # ------------------------------------------------------------------
    # Prefix-cache setup (prefix_cache mode only)
    # ------------------------------------------------------------------
    prefix_past_kv = None
    prefix_len = 0
    prefix_creation_ms = 0.0
    prefix_cache_size_bytes = 0

    if mode == "prefix_cache":
        prefix_past_kv, prefix_len, prefix_creation_ms = compute_prefix_cache(
            model, tokenizer, device
        )
        prefix_cache_size_bytes = kv_cache_bytes(prefix_past_kv)

        # Verify prefix token alignment against the first dataset example.
        _sample_prompt = build_full_prompt(
            tokenizer,
            dataset[0]["user_request"],
            dataset[0]["available_tools"],
        )
        _full_ids = tokenizer(_sample_prompt, return_tensors="pt").input_ids
        _prefix_text = build_system_prefix_text(tokenizer)
        _prefix_ids = tokenizer(_prefix_text, return_tensors="pt").input_ids
        computed_prefix_len = _prefix_ids.shape[1]

        if not torch.equal(_full_ids[:, :computed_prefix_len], _prefix_ids):
            print(
                "[prefix_cache] WARNING: prefix tokens do not align with full "
                "prompt tokens. Falling back to full prompt (no prefix savings)."
            )
            prefix_past_kv = None
            prefix_len = 0
        else:
            prefix_len = computed_prefix_len
            print(
                f"[prefix_cache] Prefix alignment verified."
                f" prefix_len={prefix_len} tokens.\n"
            )

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    ttft_capture = TTFTCapture(device)
    per_example: list[dict] = []

    for idx, example in enumerate(dataset):
        user_request = example["user_request"]
        available_tools = example["available_tools"]
        expected = example["answer"]
        tool_names = {t["name"] for t in available_tools}

        full_prompt = build_full_prompt(tokenizer, user_request, available_tools)
        full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)

        is_warmup = idx < args.warmup
        tag = (
            "[warmup]"
            if is_warmup
            else f"[{idx - args.warmup + 1:3d}/{len(dataset) - args.warmup}]"
        )

        if mode == "no_cache":
            timing = run_inference(
                model, tokenizer, full_ids, mode, device, ttft_capture
            )
        else:
            # kv_cache & prefix_cache: split prefill into a tools-list phase
            # and a user-query phase, timed separately. In production the
            # tools list is static across many requests while only the
            # query changes per call, so this isolates the per-request cost.
            if mode == "prefix_cache" and prefix_past_kv is not None:
                base_cache = clone_cache(prefix_past_kv)
                base_len = prefix_len
            else:
                base_cache = None
                base_len = 0

            tools_only_prompt = build_tools_only_prompt(tokenizer, available_tools)
            tools_only_ids = tokenizer(
                tools_only_prompt, return_tensors="pt"
            ).input_ids.to(device)
            boundary = find_tools_query_boundary(full_ids, tools_only_ids)

            tools_len = max(0, boundary - base_len)
            tools_ids = full_ids[:, base_len : base_len + tools_len]
            query_ids = full_ids[:, base_len + tools_len :]
            total_len = full_ids.shape[1]
            attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)

            cache_after_tools, tools_prefill_ms = ingest_tools_prefix(
                model, tools_ids, device, past_key_values=base_cache
            )
            timing = run_inference(
                model,
                tokenizer,
                query_ids,
                mode,
                device,
                ttft_capture,
                past_key_values=cache_after_tools,
                attention_mask=attn_mask,
                tools_prefill_ms=tools_prefill_ms,
                tools_prefill_tokens=tools_len,
                report_prefill_split=True,
            )

        predicted = extract_predicted_tool(timing["generated_text"], tool_names)
        correct = predicted == expected

        if not is_warmup:
            print(
                f"{tag} e2e={timing['e2e_latency_ms']:.0f}ms"
                f"  ttft={timing['ttft_ms']:.0f}ms"
                f"  expected={expected!r}  predicted={predicted!r}"
                f"  {'OK' if correct else 'WRONG'}"
            )
            per_example.append(
                {
                    "id": idx - args.warmup,
                    "user_request": user_request,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                    **timing,
                }
            )
        else:
            print(f"{tag} e2e={timing['e2e_latency_ms']:.0f}ms  (warmup, not recorded)")

    print(f"\n[done] Measured {len(per_example)} examples.")

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    aggregate = aggregate_metrics(per_example)
    quality = compute_quality(per_example, dataset, args.warmup)

    print("\n--- Quality ---")
    print(f"  accuracy       : {quality.get('tool_accuracy', 0):.2%}")
    print(f"  invalid rate   : {quality.get('invalid_tool_rate', 0):.2%}")
    print("\n--- Latency (mean) ---")
    print(f"  TTFT           : {aggregate.get('mean_ttft_ms')} ms")
    print(f"  prefill        : {aggregate.get('mean_prefill_latency_ms')} ms")
    if aggregate.get("mean_tools_prefill_latency_ms") is not None:
        print(
            f"    tools list   : {aggregate.get('mean_tools_prefill_latency_ms')} ms"
            f" ({aggregate.get('mean_tools_prefill_tokens')} tok, amortisable in prod)"
        )
        print(
            f"    user query   : {aggregate.get('mean_query_prefill_latency_ms')} ms"
            f" ({aggregate.get('mean_query_prefill_tokens')} tok, true per-request cost)"
        )
    print(f"  decode         : {aggregate.get('mean_decode_latency_ms')} ms")
    print(f"  E2E            : {aggregate.get('mean_e2e_latency_ms')} ms")
    print("\n--- Throughput ---")
    print(f"  prefill tok/s  : {aggregate.get('mean_prefill_tok_per_sec')}")
    print(f"  decode tok/s   : {aggregate.get('mean_decode_tok_per_sec')}")
    print("\n--- Memory ---")
    print(f"  model weights  : {weights_mb:.1f} MB (static, FP16)")
    print(f"  peak RAM       : {aggregate.get('peak_ram_mb')} MB")
    print(f"  mean KV cache  : {aggregate.get('mean_kv_cache_kb')} KB")
    _gpu_note = (
        "(CUDA: weights+activations+KV peak)"
        if device == "cuda"
        else "(MPS: weights+KV post-inference; no peak tracking)"
    )
    print(
        f"  mean peak GPU  : {aggregate.get('mean_peak_gpu_mb')} MB {_gpu_note if aggregate.get('mean_peak_gpu_mb') is not None else '(CPU — N/A)'}"
    )

    # ------------------------------------------------------------------
    # Write JSON output
    # ------------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_config: dict[str, Any] = {
        "model_key": args.model,
        "model_name": MODEL_DISPLAY_NAMES[args.model],
        "model_path": str(MODEL_PATHS[args.model]),
        "mode": mode,
        "device": device,
        "dtype": "float16",
        "dataset": str(args.dataset),
        "n_dataset_examples": len(dataset),
        "n_measured_examples": len(per_example),
        "warmup_examples": args.warmup,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.system(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_weights_mb": weights_mb,
    }

    if mode in ("kv_cache", "prefix_cache"):
        run_config["prefill_split_info"] = {
            "enabled": True,
            "note": (
                "prefill is measured in two phases: tools_prefill_* covers "
                "ingesting the available-tools list (static/cacheable across "
                "requests in production), query_prefill_* covers ingesting "
                "the dynamic user query. prefill_latency_ms/ttft_ms remain "
                "the sum of both phases for backward compatibility."
            ),
        }

    if mode == "prefix_cache":
        run_config["prefix_cache_info"] = {
            "prefix_tokens": prefix_len,
            "cache_creation_ms": round(prefix_creation_ms, 3),
            "cache_size_bytes": prefix_cache_size_bytes,
            "cache_size_kb": round(prefix_cache_size_bytes / 1024, 2),
            "note": (
                "cache_creation_ms is a one-time cost amortised over all examples. "
                "prefill_latency_ms in per_example reflects only dynamic suffix tokens."
            ),
        }

    output_doc = {
        "run_config": run_config,
        "aggregate": aggregate,
        "quality": quality,
        "per_example": per_example,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.model}_{device}_{mode}_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(output_doc, f, indent=2)

    print(f"\n[output] Results written to {out_path}")


if __name__ == "__main__":
    main()
