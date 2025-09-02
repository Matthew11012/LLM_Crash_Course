import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

# -------------- Helpers & prompt templates ---------------

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, and an input that provides additional context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

def make_prompt(instruction: str, inp: str) -> str:
    if inp is None or str(inp).strip() == "":
        return PROMPT_NO_INPUT.format(instruction=instruction)
    else:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
    

def prepare_example_for_training(tokenizer, prompt: str, response: str, max_length: int) -> Tuple[List[int], List[int]]:
    """
    Tokenize prompt and response separately, preserve prompt, truncate response if needed.
    Returns:
      input_ids: list[int]  (prompt_ids + response_ids_truncated)
      labels: list[int]     ([-100]*len(prompt_ids) + response_ids_truncated)
    If the prompt alone exceeds max_length -> returns (None, None) and caller should skip the example.
    """
    # encode without special tokens to avoid model-specific tokenization surprises
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # if prompt too long, skip
    if len(prompt_ids) >= max_length:
        return None, None
    
    remaining = max_length - len(prompt_ids)
    if len(response_ids) > remaining:
        response_ids = response_ids[:remaining]
    
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    return input_ids, labels


def dedupe_keep_first(examples: List[Dict]) -> List[Dict]:
    """
    Simple dedupe on (instruction, input, output) triplet, keep first occurence.
    """
    seen = set()
    kept = []
    for ex in examples:
        key = (ex["instruction"].strip(), (ex.get("input") or "").strip(), ex["output"].strip())
        if key not in seen:
            seen.add(key)
            kept.append(ex)
    return kept



# --------------- Main processing function ---------------

def build_and_save(
    dataset_name: str,
    split: str,
    out_json: str,
    out_hf_dir: str,
    tokenizer_name: str,
    max_length: int,
    num_examples: int,
    val_frac: float,
    seed: int
):
    random.seed(seed)

    print(f"Loading dataset: {dataset_name} split={split} ...")
    ds = load_dataset(dataset_name, split=split) # returns a Dataset

    # prefer these fields if present; adjust if dataset uses different field names
    # common Alpaca-like entries: {instruction, input, output}
    possible_fields = ds.column_names
    print("Dataset columns:", possible_fields)

    # check presence of expected fields
    if "instruction" not in possible_fields or "output" not in possible_fields:
            raise ValueError("Dataset does not contain 'instruction' and 'output' fields; please adjust code to match the dataset structure.")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # safety: ensure pad tokens exists
    if tokenizer.pad_token_id is None:
        print("Tokenizer has no pad token; setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    out_json_path = Path(out_json)
    out_hf_dir_path = Path(out_hf_dir)

    # loop and format
    raw_examples = []
    taken = 0
    for i, item in enumerate(ds):
        if taken >= num_examples * 3: # collect some margin to allow filtering/deduping
            break
        instruction = str(item.get("instruction", "")).strip()
        inp = item.get("input", "") or ""
        output = str(item.get("output", "")).strip()
        if instruction == "" or output == "":
            continue
        raw_examples.append({"instruction": instruction, "input": inp, "output": output})
        taken += 1
    
    print(f"Loaded {len(raw_examples)} raw examples (before dedupe/filter).")


    # dedupe
    raw_examples = dedupe_keep_first(raw_examples)
    print(f"{len(raw_examples)} after deduplication.")


    # build prompt + tokenizatino check
    kept = []
    skipped_prompt_too_long = 0
    skipped_zero_response = 0
    for ex in raw_examples:
        prompt = make_prompt(ex["instruction"], ex["input"])
        if ex["output"].strip() == "":
            skipped_zero_response += 1
            continue
        prepared = prepare_example_for_training(tokenizer, prompt, ex["output"], max_length)
        if prepared[0] is None:
            skipped_prompt_too_long += 1
            continue
        input_ids, labels = prepared
        # we keep the original triplet + lengths for debugging
        kept.append({
            "instruction": ex["instruction"],
            "input": ex["input"],
            "output": ex["output"],
            "prompt": prompt,
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": len(tokenizer.encode(prompt, add_special_tokens=False)),
            "response_len": len([x for x in labels if x != 100])
        })
        if len(kept) >= num_examples:
            break
    

    print(f"Kept {len(kept)} examples (skipped {skipped_prompt_too_long} prompt-too-long, {skipped_zero_response} empty-response).")

    # Shuffle determinnistically and split into train/val
    random.Random(seed).shuffle(kept)
    n = len(kept)
    n_val = max(1, int(val_frac * n))
    n_train = n - n_val
    train_examples = kept[:n_train]
    val_examples = kept[n_train:]

    # write datase.jsonl (original text triplets)
    with open(out_json_path, "w", encoding="utf-8") as f:
        for ex in train_examples + val_examples:
            json_line = {"instruction": ex["instruction"], "input": ex["input"], "output": ex["output"]}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL: {out_json_path} (train={len(train_examples)}, val={len(val_examples)})")

    # Build HF DatasetDict with tokenized fields (input_ids, labels)
    def to_simple_record(ex_list):
        return {
            "input_ids": [ex["input_ids"] for ex in ex_list],
            "labels": [ex["labels"] for ex in ex_list],
            "prompt_len": [ex["prompt_len"] for ex in ex_list],
            "response_len": [ex["response_len"] for ex in ex_list],
            # optionally keep the raw text too:
            "instruction": [ex["instruction"] for ex in ex_list],
            "input": [ex["input"] for ex in ex_list],
            "output": [ex["output"] for ex in ex_list],
        }
    
    train_dict = to_simple_record(train_examples)
    val_dict = to_simple_record(val_examples)

    hf_train = Dataset.from_dict(train_dict)
    hf_val = Dataset.from_dict(val_dict)
    hf_dataset = DatasetDict({"train": hf_train, "validation": hf_val})

    # Save HF dataset to disk
    hf_dataset.save_to_disk(str(out_hf_dir_path))
    print(f"Saved HF Dataset to {out_hf_dir_path}")

    # ----------------- Sanity checks -----------------
    print("\nSanity checks:")
    print(" - sample 3 examples (decoded)")
    for ex in (train_examples + val_examples)[:3]:
        # decode a portion to verify text matches tokens
        ids = ex["input_ids"]
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        print("PROMPT (trim):", ex["prompt"][:200].replace("\n", " "))
        print("DECODED TOKENS =>", text[:200].replace("\n", " "))
        print("---")

    # token length stats
    prompt_lens = [e["prompt_len"] for e in kept]
    resp_lens = [e["response_len"] for e in kept]
    print(f"Total kept: {len(kept)}")
    print(f"Prompt len: mean={sum(prompt_lens)/len(prompt_lens):.1f} median={sorted(prompt_lens)[len(prompt_lens)//2]}")
    print(f"Response len: mean={sum(resp_lens)/len(resp_lens):.1f} median={sorted(resp_lens)[len(resp_lens)//2]}")

    # basic leakage check (exact text overlap)
    train_texts = set((i["instruction"].strip(), (i.get("input") or "").strip(), i["output"].strip()) for i in train_examples)
    val_texts = set((i["instruction"].strip(), (i.get("input") or "").strip(), i["output"].strip()) for i in val_examples)
    overlap = train_texts.intersection(val_texts)
    print(f"Train/Val overlap count (exact triple match): {len(overlap)} (should be 0)")

    # quick DataLoader collate test
    def collate_batch(batch):
        # batch : list of {"input_ids": [...], "labels":[...]}
        input_ids = [torch.tensor(x, dtype=torch.long) for x in batch["input_ids"]]
        labels = [torch.tensor(x, dtype=torch.long) for x in batch["labels"]]
        # pad to max length in this batch
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "labels": labels}
    
    # sample a tiny batch
    sample_batch = {
        "input_ids": [train_examples[i]["input_ids"] for i in range(min(4, len(train_examples)))],
        "labels": [train_examples[i]["labels"] for i in range(min(4, len(train_examples)))]
    }
    batch_t = collate_batch(sample_batch)
    print("Collated batch shapes: input_ids", batch_t["input_ids"].shape, "labels", batch_t["labels"].shape)

    print("\nDone. If everything above looks good you can proceed to training.")
    return

# --------------- CLI ---------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned", help="HF dataset to load")
    parser.add_argument("--split", type=str, default="train", help="split name to read from the dataset")
    parser.add_argument("--out_json", type=str, default="dataset.jsonl", help="output jsonl path")
    parser.add_argument("--out_hf_dir", type=str, default="hf_dataset", help="output HF dataset dir")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="tokenizer model name")
    parser.add_argument("--max_length", type=int, default=512, help="max tokens (prompt+response)")
    parser.add_argument("--num_examples", type=int, default=1000, help="how many final examples to keep")
    parser.add_argument("--val_frac", type=float, default=0.1, help="validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_save(
        dataset_name=args.dataset,
        split=args.split,
        out_json=args.out_json,
        out_hf_dir=args.out_hf_dir,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_examples=args.num_examples,
        val_frac=args.val_frac,
        seed=args.seed,
    )

"""
To run 
python prepare_dataset.py --dataset yahma/alpaca-cleaned --split train --out_json dataset.jsonl --out_hf_dir hf_dataset --tokenizer gpt2 --max_length 512 --num_examples 1000 --val_frac 0.1 --seed 42

"""

"""
# To use the output for training
from datasets import load_from_disk
ds = load_from_disk("hf_dataset")
# ds["train"][0] has fields input_ids, labels

Create DataLoader with a collate function that pads input_ids and labels (labels pad value = -100)
In training loop, compute logits: logits = model(input_ids).logits and loss via loss_fn = CrossEntropy(ignore_index=-100) or use HF Trainer (it expects labels)
"""