#!/usr/bin/env python3
"""
CountBenchQA evaluation for Moondream Station.

Evaluates counting accuracy using the moondream/CountBenchQA dataset
by hitting the API endpoint (like simple_test.py).

Usage:
    1. Start the server: moondream-station serve
    2. Run eval: python tests/eval_countbenchqa.py [--point] [--debug]
"""

import argparse
import re
import sys
import time

import moondream
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_ENDPOINT = "http://localhost:2020/v1"


def extract_object_from_question(question: str) -> str:
    """Extract the object being counted from a 'how many X' question."""
    question = question.lower()
    if question.startswith("how many "):
        rest = question[9:]
        if " are " in rest:
            return rest.split(" are ")[0].strip()
        elif " is " in rest:
            return rest.split(" is ")[0].strip()
    return question


def normalize_number_answer(answer: str) -> str:
    """Normalize an answer to a digit string (0-10)."""
    answer = answer.strip().lower()
    if answer.isdigit() and 0 <= int(answer) <= 10:
        return answer

    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    if answer in word_to_num:
        return word_to_num[answer]

    digit_matches = re.findall(r"\b(\d+)\b", answer)
    for match in digit_matches:
        if match.isdigit() and 0 <= int(match) <= 10:
            return match

    for word, num in word_to_num.items():
        if re.search(r"\b" + re.escape(word) + r"\b", answer):
            return num

    return answer


def main():
    parser = argparse.ArgumentParser(description="CountBenchQA evaluation for Moondream Station")
    parser.add_argument(
        "--endpoint",
        "-e",
        type=str,
        default=DEFAULT_ENDPOINT,
        help="Server endpoint URL",
    )
    parser.add_argument("--debug", action="store_true", help="Show detailed output for each sample")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--point", action="store_true", help="Use point mode instead of query"
    )
    args = parser.parse_args()

    print(f"Connecting to Moondream Station at {args.endpoint}")
    model = moondream.vl(endpoint=args.endpoint)

    print("Loading CountBenchQA dataset...")
    dataset = load_dataset("moondream/CountBenchQA", split="test")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} samples")

    correct = 0
    total = 0
    total_time = 0.0

    mode_str = "point" if args.point else "query"
    desc = f"CountBenchQA ({mode_str})"

    for row in tqdm(dataset, disable=args.debug, desc=desc):
        image = row["image"]  # PIL Image from HuggingFace datasets
        question = row["question"]
        ground_truth = str(row["number"])

        start = time.time()

        if args.point:
            obj = extract_object_from_question(question)
            result = model.point(image, obj)
            points = result.get("points", [])
            model_answer = str(len(points))
        else:
            result = model.query(
                image,
                question,
                reasoning=True,
                settings={"temperature": 0, "max_tokens": 512},
            )
            model_answer = str(result.get("answer", "")).strip()

        elapsed = time.time() - start
        total_time += elapsed

        is_correct = normalize_number_answer(model_answer) == normalize_number_answer(
            ground_truth
        )

        total += 1
        if is_correct:
            correct += 1

        if args.debug:
            if args.point:
                obj = extract_object_from_question(question)
                print(f"Q: {question}")
                print(f"Object: {obj}")
                print(f"Points: {len(result.get('points', []))}")
            else:
                reasoning = result.get("reasoning", {}).get("text", "")
                print(f"Q: {question}")
                print(f"Reasoning: {reasoning}")
            print(f"Answer: {model_answer}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Correct: {is_correct}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Running: {correct}/{total} ({correct * 100 / total:.2f}%)")
            print("-" * 40)

    mode = "point" if args.point else "query"
    print(f"\nCountBenchQA Results ({mode})")
    print(f"Accuracy: {correct * 100 / total:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per sample: {total_time / total:.2f}s")


if __name__ == "__main__":
    main()
