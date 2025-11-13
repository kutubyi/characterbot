"""
Usage:
    python prepare_finetune_data.py [--mcq PATH] [--qa PATH] [--style PATH] [--output PATH]
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_mcq_to_alpaca(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    for entry in entries:
        qa_pairs = entry.get("qa_pairs", [])
        for qa in qa_pairs:
            question = qa.get("question", "")
            options = qa.get("option", {})
            answer = qa.get("answer", "")

            # Format input with options
            input_text = question + "\n"
            if isinstance(options, dict):
                for key, value in options.items():
                    input_text += f"{key}: {value}\n"
            elif isinstance(options, list):
                for option in options:
                    input_text += f"{option}\n"

            converted.append({
                "instruction": "请在以下四个选项中选择一个最合适的答案。",
                "input": input_text.strip(),
                "output": answer,
                "label": "1"
            })
    return converted


def convert_qa_to_alpaca(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    for entry in entries:
        qa_pairs = entry.get("qa_pairs", [])
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")

            converted.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "label": "2"
            })
    return converted


def convert_style_to_alpaca(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    for entry in entries:
        transformed = entry.get("transformed_sentences", [])
        for item in transformed:
            original = item.get("original", "")
            plain = item.get("plain", "")

            converted.append({
                "instruction": "请将以下文本改写为更平白的表达方式。",
                "input": original,
                "output": plain,
                "label": "3"
            })
    return converted


def merge_finetune_data(mcq_path: str, qa_path: str, style_path: str, output_path: str) -> bool:
    mcq_data = load_json_file(mcq_path)
    print(f"Loaded {Path(mcq_path).name}: {len(mcq_data)} entries")

    qa_data = load_json_file(qa_path)
    print(f"Loaded {Path(qa_path).name}: {len(qa_data)} entries")

    style_data = load_json_file(style_path)
    print(f"Loaded {Path(style_path).name}: {len(style_data)} entries")

    converted_mcq = convert_mcq_to_alpaca(mcq_data)
    converted_qa = convert_qa_to_alpaca(qa_data)
    converted_style = convert_style_to_alpaca(style_data)

    print(f"Converted MCQ data: {len(converted_mcq)} entries (label: 1)")
    print(f"Converted QA data: {len(converted_qa)} entries (label: 2)")
    print(f"Converted style transfer data: {len(converted_style)} entries (label: 3)")

    merged_data = converted_mcq + converted_qa + converted_style
    print(f"Total merged: {len(merged_data)} entries")

    # Save output
    if output_path is None:
        output_path = "fine_tune.json"

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
        "--mcq",
        default="json_output/multiple_choice_questions.json"
    )
    parser.add_argument(
        "--qa",
        default="json_output/generative_qa.json"
    )
    parser.add_argument(
        "--style",
        default="json_output/style_transfer.json"
    )
    parser.add_argument(
        "--output",
        default="train_with_charlora/fine_tune.json"
    )

    args = parser.parse_args()

    print(f"MCQ data file: {args.mcq}")
    print(f"QA data file: {args.qa}")
    print(f"Style transfer data file: {args.style}")
    print(f"Output file: {args.output}\n")

    # Run merge
    success = merge_finetune_data(args.mcq, args.qa, args.style, args.output)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
