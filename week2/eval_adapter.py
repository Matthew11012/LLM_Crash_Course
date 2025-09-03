#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a LoRA adapter (perplexity on a validation split).

- Loads a base causal LM and the LoRA adapter.
- Rebuilds the same SFT prompt template used for training.
- Tokenizes with label masking (loss only on the response).
- Computes average loss and perplexity (exp(loss)) on the split.

Example:
python eval_adapter.py --base_model gpt2 --adapter_path outputs/lora_adapter --dataset_path hf_dataset --split validation --batch_size 8 --max_length 512 --device cuda
"""

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import default_data_collator


# ---------- Prompt template (same as you trained with) ----------
def build_prompt(instruction: str, inp: str | None) -> str:
    if inp and len(inp.strip()) > 0:
        return (
            "Below is an instruction that describes a task, and an input that provides additional context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )


# ---------- Tokenization function with label masking ----------
def make_tokenize_fn(tokenizer, max_length: int):
    eos_id = tokenizer.eos_token_id

    def _tok(example):
        instr = example.get("instruction", "")
        inp = example.get("input", "")
        out = example.get("output", "")

        prompt = build_prompt(instr, inp)
        # Encode prompt and response separately so we can mask the prompt.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # Append EOS to the response so the model knows where to stop.
        resp_ids = tokenizer(out, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            resp_ids = resp_ids + [eos_id]

        input_ids = prompt_ids + resp_ids
        # Labels: ignore prompt positions, supervise only response.
        labels = [-100] * len(prompt_ids) + resp_ids

        # Truncate to max_length
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        # Attention mask before padding
        attn_len = len(input_ids)

        # Pad to fixed length for efficient batching
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        attention_mask = [1] * attn_len + [0] * pad_len

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return _tok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="gpt2",
                   help="Base HF model id or local path (e.g., 'gpt2').")
    p.add_argument("--adapter_path", type=str, required=True,
                   help="Path to the saved LoRA adapter (folder).")
    p.add_argument("--dataset_path", type=str, default="hf_dataset",
                   help="Path to the saved HF dataset (load_from_disk).")
    p.add_argument("--split", type=str, default="validation",
                   choices=["train", "validation", "test"],
                   help="Which split to evaluate.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512,
                   help="Max sequence length for evaluation.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optionally cap number of samples for quick eval.")
    p.add_argument("--device", type=str, default=None,
                   help="cpu | cuda | auto (None uses CUDA if available).")
    return p.parse_args()


def main():
    args = parse_args()

    # ---------- Device ----------
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # Most causal LMs (e.g., GPT-2) lack a pad token. Use EOS as PAD.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocab size: {len(tokenizer)} | pad_token_id: {tokenizer.pad_token_id} | eos_token_id: {tokenizer.eos_token_id}")

    # ---------- Model + LoRA adapter ----------
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("adapter_path:", args.adapter_path)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    # print("Loaded adapters:", model.peft_config)
    model.eval().to(device)

    # only LoRA params should be trainable (we're evaluating, but check).
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params (should be just LoRA adapters): {len(trainable)} tensors")
    # PEFT sets inference_mode=True, when saving model after LoRA finetune. All params are frozen when model loaded, so 0 trainable tensors

    for name, module in model.named_modules():
        if "lora" in name.lower():
            print("LoRA module found:", name)

    # ---------- Data ----------
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
    dsd = load_from_disk(args.dataset_path)
    if args.split not in dsd:
        raise ValueError(f"Split '{args.split}' not found in dataset: {list(dsd.keys())}")
    ds = dsd[args.split]
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Eval split size: {len(ds)}")

    # Map -> tokenized
    tok_fn = make_tokenize_fn(tokenizer, args.max_length)
    ds_tok = ds.map(tok_fn, remove_columns=ds.column_names, desc="Tokenizing", num_proc=None)

    # Fixed-length batches, so default collate is fine
    dl = DataLoader(
        ds_tok,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator, # Use default_data_collator so we get proper tensors, not lists
    )

    # ---------- Evaluation loop ----------
    total_loss = 0.0
    total_tokens = 0  # count of supervised tokens (labels != -100)
    total_batches = 0

    with torch.inference_mode():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # averaged over non-ignored tokens per batch (HF handles -100 masking)

            if loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()
            total_batches += 1

            # Count supervised tokens (for info)
            total_tokens += (labels != -100).sum().item()

    mean_loss = total_loss / max(1, total_batches)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")  # guard overflow

    print({
        "eval_loss": mean_loss,
        "perplexity": ppl,
        "num_batches": total_batches,
        "supervised_tokens": total_tokens,
        "split": args.split,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    })


if __name__ == "__main__":
    main()
