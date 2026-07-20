# Target Devices

This project benchmarks inference on two target devices.

## Apple M4 MacBook Air

- SoC: Apple M4 Macbook Air
- CPU: 10 cores
- GPU: 10-core integrated Apple GPU (Metal 4 support)
- Memory: 16 GB unified memory
- Storage: 500 GB internal SSD
- Role in this project: primary development and baseline reference platform

Why this device matters:

- Fast iteration for model conversion and benchmark harness validation
- Strong local baseline for latency and quality comparisons
- Useful reference before running constrained edge tests

## Raspberry Pi 5 Model B Rev 1.0

- Board: Raspberry Pi 5 Model B Rev 1.0
- CPU: 4-core ARM CPU
- Memory: about 8 GB system memory (7937 MiB reported)
- Storage: 127 GB local disk (mmcblk0)
- Network: Ethernet available
- Role in this project: constrained edge inference benchmark target

Why this device matters:

- Represents practical CPU-only edge deployment conditions
- Useful for quantized runtime comparisons under memory constraints
- Critical for TTFT, prefill latency, and RAM-footprint benchmarking

## Baseline Stack

- Apple M4 MacBook Air: Transformers reference, then llama.cpp and ONNX Runtime
- Raspberry Pi 5: llama.cpp plus GGUF first, then ONNX Runtime CPU
