#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-ASR Transformers Inference Example

This script demonstrates how to use Qwen3-ASR with Transformers backend for:
- Single audio transcription
- Batch transcription
- Transcription with timestamps

Usage:
    python example_inference.py
"""

import os
import torch
from qwen_asr import Qwen3ASRModel

# Configuration
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_MODEL = os.environ.get("ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda:0")
DTYPE = os.environ.get("DTYPE", "bfloat16")

# Map string dtype to torch dtype
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Sample audio URLs
SAMPLE_ZH = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
SAMPLE_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"


def print_result(title: str, results) -> None:
    """Pretty print transcription results."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    for i, r in enumerate(results):
        print(f"\n[Sample {i + 1}]")
        print(f"  Language: {r.language}")
        print(f"  Text: {r.text}")

        if r.time_stamps and len(r.time_stamps) > 0:
            print(f"  Timestamps ({len(r.time_stamps)} segments):")
            # Show first 3 and last 3 timestamps
            ts_list = r.time_stamps
            to_show = (
                ts_list[:3] + (["..."] if len(ts_list) > 6 else []) + ts_list[-3:]
                if len(ts_list) > 6
                else ts_list
            )
            for ts in to_show:
                if ts == "...":
                    print("    ...")
                else:
                    print(f"    [{ts.start_time:.2f}s - {ts.end_time:.2f}s] {ts.text}")


def main():
    print("=" * 60)
    print("  Qwen3-ASR Transformers Inference Example")
    print("=" * 60)
    print(f"\nLoading models...")
    print(f"  ASR Model: {ASR_MODEL}")
    print(f"  Aligner Model: {ALIGNER_MODEL}")
    print(f"  Device Map: {DEVICE_MAP}")
    print(f"  Dtype: {DTYPE}")

    dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)

    # Initialize model with Transformers backend
    model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        dtype=dtype,
        device_map=DEVICE_MAP,
        # attn_implementation="flash_attention_2",  # Uncomment if flash-attn is installed
        max_inference_batch_size=32,
        max_new_tokens=256,
        forced_aligner=ALIGNER_MODEL,
        forced_aligner_kwargs=dict(
            dtype=dtype,
            device_map=DEVICE_MAP,
            # attn_implementation="flash_attention_2",  # Uncomment if flash-attn is installed
        ),
    )

    print("\nModels loaded successfully!")

    # Example 1: Single audio transcription (auto language detection)
    print("\n" + "=" * 60)
    print("  Example 1: Single Audio (Auto Language Detection)")
    print("=" * 60)

    results = model.transcribe(
        audio=SAMPLE_EN,
        language=None,  # Auto-detect language
        return_time_stamps=False,
    )
    print_result("Single Audio - English", results)

    # Example 2: Single audio with forced language
    print("\n" + "=" * 60)
    print("  Example 2: Single Audio (Forced Language)")
    print("=" * 60)

    results = model.transcribe(
        audio=SAMPLE_ZH,
        language="Chinese",
        return_time_stamps=False,
    )
    print_result("Single Audio - Chinese", results)

    # Example 3: Batch transcription
    print("\n" + "=" * 60)
    print("  Example 3: Batch Transcription")
    print("=" * 60)

    results = model.transcribe(
        audio=[SAMPLE_EN, SAMPLE_ZH],
        language=["English", "Chinese"],
        return_time_stamps=False,
    )
    print_result("Batch Transcription", results)

    # Example 4: Transcription with timestamps
    print("\n" + "=" * 60)
    print("  Example 4: Transcription with Timestamps")
    print("=" * 60)

    results = model.transcribe(
        audio=SAMPLE_EN,
        language="English",
        return_time_stamps=True,
    )
    print_result("With Timestamps - English", results)

    # Example 5: Batch with timestamps
    print("\n" + "=" * 60)
    print("  Example 5: Batch with Timestamps")
    print("=" * 60)

    results = model.transcribe(
        audio=[SAMPLE_EN, SAMPLE_ZH],
        language=["English", "Chinese"],
        return_time_stamps=True,
    )
    print_result("Batch with Timestamps", results)

    print("\n" + "=" * 60)
    print("  All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
