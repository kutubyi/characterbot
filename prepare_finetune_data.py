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


def split_train_test(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
    labels_dict = {}
    for item in data:
        label = item.get("label", "unknown")
        if label not in labels_dict:
            labels_dict[label] = []
        labels_dict[label].append(item)

    train_data = []
    test_data = []

    for label, items in labels_dict.items():
        split_idx = int(len(items) * train_ratio)
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])

    return train_data, test_data


def merge_finetune_data(mcq_path: str, qa_path: str, style_path: str, output_dir: str) -> bool:
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

    train_data, test_data = split_train_test(merged_data, train_ratio=0.8)
    print(f"\nTrain/Test split (80% train / 20% test for each label):")
    print(f"  Training data: {len(train_data)} entries")
    print(f"  Test data: {len(test_data)} entries")

    if output_dir is None:
        output_dir = "train_with_charlora"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save training data
    train_output_path = os.path.join(output_dir, "fine_tune.json")
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved training data to: {train_output_path}")

    # Save test data
    test_output_path = os.path.join(output_dir, "test.json")
    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Saved test data to: {test_output_path}")

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
        default="train_with_charlora",
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
