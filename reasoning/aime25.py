#!/usr/bin/env python3
"""AIME 2025 inference and evaluation."""

import os
os.environ["HF_HOME"] = f"/home/rjie/orcd/scratch/.cache"

import argparse
import re
import torch

import hf_common as common

import numpy as np
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你有多个 GPU
    
    # 下面两行能保证卷积等操作的确定性，但会稍微降低运行速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PROMPT_TEMPLATE = """\
Solve the following math problem step by step. Put your final answer
in a \\boxed{{}} command.

{problem}

Remember to put your final answer in \\boxed{{}}.

Reasoning:"""


def build_prompt(item):
    problem = item.get("problem") or item.get("question") or item.get("prompt", "")
    return PROMPT_TEMPLATE.format(problem=problem)


def extract_answer(text, default=""):
    """Extract answer from \\boxed{}, with fallbacks."""
    boxed = common.extract_boxed(text)
    if boxed:
        return boxed
    match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = text.strip().split("\n")
    if lines and lines[-1].strip() and not lines[-1].strip().endswith(("?", ".")):
        return lines[-1].strip()
    return default


def evaluate(results_path):
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig

    results = common.load_results(results_path)
    total = correct = 0
    for res in results:
        if not res.get("success"):
            continue
        total += 1
        gold = parse(str(res.get("answer", "")).strip(),
                      extraction_config=[ExprExtractionConfig()])
        pred = parse(str(res.get("pred", "")).strip())
        if verify(gold, pred):
            correct += 1
    acc = f"{100 * correct / total:.1f}%" if total else "N/A"
    print(f"AIME 2025: {correct}/{total} = {acc}")


def main():
    ap = argparse.ArgumentParser(description="AIME 2025 Inference & Evaluation")
    common.add_common_args(ap)
    ap.add_argument(
        "--data_path",
        default="lchen001/AIME2025",
        help="Local JSONL path or HuggingFace dataset name (default: opencompass/AIME2025)",
    )
    ap.add_argument("--out", default="./results/aime25_preds.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    seed_everything(args.seed)

    if args.eval_only:
        evaluate(args.out)
        return

    # client, model = common.create_client(args.host, args.port, args.model_name)
    model, tokenizer = common.create_model_tokenizer(args.model_name)
    dataset = common.load_jsonl_or_hf(args.data_path, split="train")
    # if args.limit:
    #     dataset = dataset[:args.limit]

    # Calculate start and end indices
    start_idx = args.group_id * args.group_size
    end_idx = start_idx + args.group_size

    # Safety check for out-of-bounds
    if start_idx >= len(dataset):
        print(f"⚠️  Warning: group_id {args.group_id} is out of bounds for dataset size {len(dataset)}")
        dataset = []
    else:
        # Slice the dataset
        dataset = dataset[start_idx : end_idx]
        actual_count = len(dataset)
        print(f"   Group ID: {args.group_id}")
        print(f"   Range: [{start_idx} : {start_idx + actual_count}]")

    print(f"Loaded {len(dataset)} problems\n")

    def process(item):
        return common.process_question(
            model, tokenizer, item, model_name=args.model_name,
            n_samples=args.n, temperature=args.temperature,
            top_p=args.top_p, top_logprobs=args.top_logprobs,
            prompt_text=build_prompt(item),
            extract_fn=extract_answer,
            system_prompt="You are a helpful assistant that solves math problems.",
            max_new_tokens=args.max_new_tokens,
        )

    common.run_inference(dataset, process, args.out, args.max_workers, desc="AIME 2025")


if __name__ == "__main__":
    main()
