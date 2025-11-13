"""
Usage:
    python prepare_pretrain_data.py --original <path> --author <name> [--reframe PATH] [--output PATH]
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_original_to_pretrain(entries: List[Dict[str, Any]], author: str) -> List[Dict[str, str]]:
    converted = []
    for entry in entries:
        title = entry.get("title", "")
        text = entry.get("text", "")
        metadata = f"[作者: {author}][标题: {title}]"
        full_text = f"{metadata}{text}"
        converted.append({"text": full_text})
    return converted


def convert_reframe_to_pretrain(entries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    converted = []
    for entry in entries:
        text = entry.get("text", "")
        converted.append({"text": text})
    return converted


def merge_json_files(original_path: str, reframe_path: str, output_path: str, author: str) -> bool:
    original_data = load_json_file(original_path)
    original_name = Path(original_path).name
    print(f"Loaded {original_name}: {len(original_data)} entries")

    reframe_data = load_json_file(reframe_path)
    reframe_name = Path(reframe_path).name
    print(f"Loaded {reframe_name}: {len(reframe_data)} entries")

    converted_original = convert_original_to_pretrain(original_data, author=author)
    converted_reframe = convert_reframe_to_pretrain(reframe_data)
    print(f"Converted original data: {len(converted_original)} entries")
    print(f"Converted reframed data: {len(converted_reframe)} entries")

    merged_data = converted_original + converted_reframe
    print(f"Total merged: {len(merged_data)} entries")

    # Save output
    if output_path is None:
        output_path = "pre_train.json"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved merged output to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original",
        required=True
    )
    parser.add_argument(
        "--reframe",
        default="json_output/reframe.json"
    )
    parser.add_argument(
        "--output",
        default="pre_train.json"
    )
    parser.add_argument(
        "--author",
        required=True
    )

    args = parser.parse_args()

    print(f"Original data file: {args.original}")
    print(f"Reframed data file: {args.reframe}")
    print(f"Output file: {args.output}")
    print(f"Author: {args.author}\n")

    # Run merge
    success = merge_json_files(args.original, args.reframe, args.output, author=args.author)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
    