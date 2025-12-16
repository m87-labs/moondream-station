"""
Comprehensive test script for Moondream Station MLX backend.

Tests all 4 capabilities (caption, query, detect, point), streaming,
reasoning, and sampling settings (temperature, top_p, max_tokens).

Usage:
    1. Start the server: moondream-station serve
    2. Run tests: python tests/simple_test.py [--image PATH]
"""

import argparse
import sys
import time
from pathlib import Path

import moondream
from PIL import Image

DEFAULT_ENDPOINT = "http://localhost:2020/v1"
DEFAULT_IMAGE = "/Users/ethanreid/Downloads/4N0A6482.jpeg"


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name: str, result, elapsed: float):
    print(f"\n[{name}] ({elapsed:.2f}s)")
    if isinstance(result, dict):
        for key, value in result.items():
            if key.startswith("_"):
                continue
            if hasattr(value, "__iter__") and hasattr(value, "__next__"):
                print(f"  {key}: <streaming generator>")
            elif isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            elif isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{len(value)} items]")
                for item in value[:3]:
                    print(f"    - {item}")
                print(f"    ... and {len(value) - 3} more")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  {result}")


def test_caption(model, image: Image.Image):
    print_header("CAPTION TESTS")

    # Test basic caption
    start = time.time()
    result = model.caption(image, length="short")
    print_result("Caption (short)", result, time.time() - start)

    start = time.time()
    result = model.caption(image, length="normal")
    print_result("Caption (normal)", result, time.time() - start)

    start = time.time()
    result = model.caption(image, length="long")
    print_result("Caption (long)", result, time.time() - start)

    # Test streaming caption
    print("\n[Caption (streaming)]")
    start = time.time()
    result = model.caption(image, length="normal", stream=True)
    print("  caption: ", end="", flush=True)
    for chunk in result["caption"]:
        print(chunk, end="", flush=True)
    print(f"\n  (elapsed: {time.time() - start:.2f}s)")

    # Test with temperature setting
    start = time.time()
    result = model.caption(image, length="short", settings={"temperature": 0.8})
    print_result("Caption (temp=0.8)", result, time.time() - start)

    # Test with max_tokens setting
    start = time.time()
    result = model.caption(image, length="normal", settings={"max_tokens": 50})
    print_result("Caption (max_tokens=50)", result, time.time() - start)


def test_query(model, image: Image.Image):
    print_header("QUERY TESTS")

    question = "What do you see in this image?"

    # Test basic query
    start = time.time()
    result = model.query(image, question)
    print_result(f"Query: '{question}'", result, time.time() - start)

    # Test query with reasoning
    start = time.time()
    result = model.query(image, question, reasoning=True)
    print_result(f"Query with reasoning", result, time.time() - start)

    # Test streaming query
    print(f"\n[Query (streaming): '{question}']")
    start = time.time()
    result = model.query(image, question, stream=True)
    print("  answer: ", end="", flush=True)
    for chunk in result["answer"]:
        print(chunk, end="", flush=True)
    print(f"\n  (elapsed: {time.time() - start:.2f}s)")

    # Test with temperature and top_p
    start = time.time()
    result = model.query(
        image, "Describe the colors.", settings={"temperature": 0.3, "top_p": 0.95}
    )
    print_result("Query (temp=0.3, top_p=0.95)", result, time.time() - start)

    # Test greedy decoding (temp=0)
    start = time.time()
    result = model.query(image, "What is this?", settings={"temperature": 0})
    print_result("Query (greedy, temp=0)", result, time.time() - start)


def test_detect(model, image: Image.Image):
    print_header("DETECT TESTS")

    # Test basic detection
    objects_to_detect = ["text", "logo", "shape"]

    for obj in objects_to_detect:
        start = time.time()
        result = model.detect(image, obj)
        print_result(f"Detect '{obj}'", result, time.time() - start)

    # Test with max_objects setting
    start = time.time()
    result = model.detect(image, "shape", settings={"max_objects": 5})
    print_result("Detect (max_objects=5)", result, time.time() - start)


def test_point(model, image: Image.Image):
    print_header("POINT TESTS")

    # Test basic pointing
    objects_to_point = ["text", "center", "logo"]

    for obj in objects_to_point:
        start = time.time()
        result = model.point(image, obj)
        print_result(f"Point '{obj}'", result, time.time() - start)

    # Test with max_objects setting
    start = time.time()
    result = model.point(image, "corner", settings={"max_objects": 4})
    print_result("Point (max_objects=4)", result, time.time() - start)


def test_settings_combinations(model, image: Image.Image):
    print_header("SETTINGS COMBINATION TESTS")

    # Test various setting combinations
    test_cases = [
        {"temperature": 0.1, "max_tokens": 100},
        {"temperature": 0.5, "top_p": 0.9},
        {"temperature": 0.9, "top_p": 0.5, "max_tokens": 200},
        {"max_tokens": 30},  # Only max_tokens
    ]

    question = "Briefly describe this image."
    for settings in test_cases:
        start = time.time()
        result = model.query(image, question, settings=settings)
        settings_str = ", ".join(f"{k}={v}" for k, v in settings.items())
        print_result(f"Query ({settings_str})", result, time.time() - start)


def main():
    parser = argparse.ArgumentParser(description="Test Moondream Station capabilities")
    parser.add_argument(
        "--image", "-i", type=str, default=str(DEFAULT_IMAGE), help="Path to test image"
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        type=str,
        default=DEFAULT_ENDPOINT,
        help="Server endpoint URL",
    )
    parser.add_argument(
        "--skip-caption", action="store_true", help="Skip caption tests"
    )
    parser.add_argument("--skip-query", action="store_true", help="Skip query tests")
    parser.add_argument("--skip-detect", action="store_true", help="Skip detect tests")
    parser.add_argument("--skip-point", action="store_true", help="Skip point tests")
    parser.add_argument(
        "--skip-settings", action="store_true", help="Skip settings tests"
    )
    args = parser.parse_args()

    # Check image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        sys.exit(1)

    print(f"Testing Moondream Station at {args.endpoint}")
    print(f"Using image: {args.image}")

    # Load image and create client
    image = Image.open(args.image)
    model = moondream.vl(endpoint=args.endpoint)

    total_start = time.time()

    try:
        if not args.skip_caption:
            test_caption(model, image)

        if not args.skip_query:
            test_query(model, image)

        if not args.skip_detect:
            test_detect(model, image)

        if not args.skip_point:
            test_point(model, image)

        if not args.skip_settings:
            test_settings_combinations(model, image)

        print_header("ALL TESTS COMPLETED")
        print(f"Total time: {time.time() - total_start:.2f}s")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
