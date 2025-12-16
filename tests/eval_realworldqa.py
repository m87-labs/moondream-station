#!/usr/bin/env python3
"""
RealWorldQA evaluation for Moondream Station.

Evaluates visual question answering accuracy using the lmms-lab/RealWorldQA dataset
by hitting the API endpoint (like simple_test.py).

Usage:
    1. Start the server: moondream-station serve
    2. Run eval: python tests/eval_realworldqa.py [--debug] [--max-samples N]
"""

import argparse
import sys
import time

import moondream
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_ENDPOINT = "http://localhost:2020/v1"


def main():
    parser = argparse.ArgumentParser(description="RealWorldQA evaluation for Moondream Station")
    parser.add_argument(
        "--endpoint",
        "-e",
        type=str,
        default=DEFAULT_ENDPOINT,
        help="Server endpoint URL",
    )
    parser.add_argument("--debug", action="store_true", help="Show detailed output for each sample")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    print(f"Connecting to Moondream Station at {args.endpoint}")
    model = moondream.vl(endpoint=args.endpoint)

    print("Loading RealWorldQA dataset...")
    dataset = load_dataset("lmms-lab/RealWorldQA", split="test")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} samples")

    correct = 0
    total = 0
    total_time = 0.0

    for row in tqdm(dataset, disable=args.debug, desc="RealWorldQA"):
        image = row["image"]  # PIL Image from HuggingFace datasets
        question = row["question"]
        ground_truth = row["answer"]

        start = time.time()

        result = model.query(
            image,
            question,
            settings={"temperature": 0, "max_tokens": 128},
        )
        model_answer = str(result.get("answer", "")).strip()

        elapsed = time.time() - start
        total_time += elapsed

        is_correct = model_answer.strip().lower() == ground_truth.strip().lower()

        total += 1
        if is_correct:
            correct += 1

        if args.debug:
            status = "CORRECT" if is_correct else "WRONG"
            print(f"[{status}] Q: {question}")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Model Answer: {model_answer}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Running: {correct}/{total} ({correct * 100 / total:.2f}%)")
            print("-" * 40)

    print(f"\nRealWorldQA Results")
    print(f"Accuracy: {correct * 100 / total:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per sample: {total_time / total:.2f}s")


if __name__ == "__main__":
    main()
