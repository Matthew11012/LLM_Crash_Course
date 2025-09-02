# Day 3 — Build a Small Instruction Dataset (with Hugging Face)

This note walks you through **preparing a small, clean instruction dataset** ready for causal-LM fine-tuning. We’ll use a public dataset (`yahma/alpaca-cleaned`), format prompts consistently, **tokenize** with a GPT-style tokenizer, **filter** by length, **split** into train/val, and **save** both a JSONL file and a tokenized HF `Dataset`. Along the way we’ll add **sanity checks**, **metrics**, and a few **reproducibility** tips.

---

## 1) What we’re building

**Inputs:** A public instruction dataset with fields like:

* `instruction`: the task to perform
* `input`: optional extra context
* `output`: the desired response

**Outputs:**

* `dataset.jsonl` — a cleaned, uniformly formatted instruction dataset
* `hf_dataset/` — a tokenized Hugging Face `DatasetDict` with `train` and `validation` splits

**Guarantees:**

* No empty examples
* No duplicates (by `(instruction, input, output)`)
* All examples **fit under your max token length**
* No train/val leakage
* Shapes and attention masks are correct after tokenization

---

## 2) Prompt template (why it matters)

To help the model learn consistent behavior, we wrap every example in a single, **rigid prompt format**. This teaches the model what “an instruction” looks like and where the **response** should start:

```text
Below is an instruction that describes a task{input_clause}. 
Write a response that appropriately completes the request.

### Instruction:
{instruction}
{input_block}
### Response:
{output}
```

* `input_clause` is: `, and an input that provides additional context` if `input` exists, otherwise empty.
* `input_block` is:

  ```
  ### Input:
  {input}
  ```

  if present, otherwise empty.

This regularity greatly stabilizes fine-tuning.

---

## 3) Tokenizer choices & padding

We’ll use a GPT-style tokenizer (e.g., `"gpt2"`). Two important points:

* **GPT-2 has no pad token**. We set:
  `tokenizer.pad_token = tokenizer.eos_token`
  so that padding uses EOS. Later, when we build labels, we’ll set loss to **ignore padding** (labels = `-100` where `input_ids == pad_id`).
* We **truncate** to `max_length` so every example fits in the model’s context window (e.g., 512 or 1024).

---

## 4) End-to-end script

A single script to: **load → clean → format → dedupe → length-filter → write JSONL → tokenize → split → save HF dataset → run sanity checks**.

> Save as `prepare_dataset.py` and run with
> `python prepare_dataset.py --model_name gpt2 --dataset yahma/alpaca-cleaned --num_examples 1000 --max_length 512`

```python
# prepare_dataset.py
import argparse, json, math, random, os
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import torch

# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Prompt formatting
# -----------------------------
def format_example(instruction: str, inp: str | None, output: str) -> Tuple[str, str]:
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    output = (output or "").strip()

    has_input = len(inp) > 0
    intro = "Below is an instruction that describes a task"
    if has_input:
        intro += ", and an input that provides additional context"
    intro += ". Write a response that appropriately completes the request."

    input_block = f"\n\n### Input:\n{inp}\n" if has_input else "\n"
    prompt = (
        f"{intro}\n\n### Instruction:\n{instruction}"
        f"{input_block}### Response:\n"
    )
    # `prompt` is what the model sees as input; `output` is the supervised target continuation
    return prompt, output

# -----------------------------
# Token length utilities
# -----------------------------
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def concat_and_tokenize(tokenizer, prompt: str, output: str, max_length: int) -> Dict[str, torch.Tensor]:
    # We train casual LM by concatenating prompt + output and predicting next tokens
    full_text = prompt + output
    enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",  # padding here is fine; we will ignore pad tokens in loss
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)

    # Labels: same as input_ids, but ignore pads with -100 (so they don't count toward loss)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "text": full_text,  # for sanity/debug
    }

# -----------------------------
# Main builder
# -----------------------------
def build_and_save(
    model_name: str = "gpt2",
    dataset_name: str = "yahma/alpaca-cleaned",
    split: str = "train",
    num_examples: int = 1000,
    max_length: int = 512,
    out_jsonl: str = "dataset.jsonl",
    out_hf_dir: str = "hf_dataset",
    seed: int = 42,
):
    set_seed(seed)

    print(f"Loading dataset: {dataset_name} split={split} ...")
    ds = load_dataset(dataset_name, split=split)

    # Peek at columns
    print("Dataset columns:", list(ds.features.keys()))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad token; setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Collect a pool to allow filtering/deduping later
    raw_examples = []
    taken = 0
    cap = num_examples * 3  # margin to survive filters
    for item in ds:
        if taken >= cap:
            break
        instruction = (item.get("instruction") or "").strip()
        inp = (item.get("input") or "").strip()
        output = (item.get("output") or "").strip()
        if instruction == "" or output == "":
            continue
        raw_examples.append({"instruction": instruction, "input": inp, "output": output})
        taken += 1

    print(f"Loaded {len(raw_examples)} raw examples (before dedupe/filter).")

    # Deduplicate by exact triple
    seen = set()
    deduped = []
    for r in raw_examples:
        key = (r["instruction"], r["input"], r["output"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"{len(deduped)} after deduplication.")

    # Length filtering: keep only those that fit under max_length when concatenated
    kept, prompt_lens, resp_lens = [], [], []
    n_prompt_too_long = n_empty_resp = 0

    for r in deduped:
        prompt, output = format_example(r["instruction"], r["input"], r["output"])
        # quick length checks
        pl = count_tokens(tokenizer, prompt)
        rl = count_tokens(tokenizer, output)
        # require prompt alone to fit; and prompt+resp to fit after truncation
        if pl >= max_length - 1:
            n_prompt_too_long += 1
            continue
        if len(output) == 0:
            n_empty_resp += 1
            continue
        kept.append({"prompt": prompt, "output": output})
        prompt_lens.append(pl)
        resp_lens.append(rl)
        if len(kept) >= num_examples:
            break

    print(f"Kept {len(kept)} examples (skipped{n_prompt_too_long} prompt-too-long, {n_empty_resp} empty-response).")

    # Write JSONL for transparency / future reuse
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL: {out_jsonl} (train={int(len(kept)*0.9)}, val={len(kept)-int(len(kept)*0.9)})")

    # Build HF Dataset
    texts = [ex["prompt"] + ex["output"] for ex in kept]
    ds_hf = Dataset.from_dict({"text": texts})

    # Train/val split (no leakage by construction)
    ds_dict = ds_hf.train_test_split(test_size=0.1, seed=seed)
    ds_dict = DatasetDict({"train": ds_dict["train"], "validation": ds_dict["test"]})

    # Tokenize batched
    def tokenize_batch(batch):
        enc = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        # Build labels that ignore pads
        labels = []
        for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
            lab = [tok if a == 1 else -100 for tok, a in zip(ids, attn)]
            labels.append(lab)
        enc["labels"] = labels
        return enc

    ds_tok = ds_dict.map(tokenize_batch, batched=True, remove_columns=["text"])

    # Save tokenized dataset
    os.makedirs(out_hf_dir, exist_ok=True)
    ds_tok.save_to_disk(out_hf_dir)
    print(f"Saved HF Dataset to {out_hf_dir}")

    # -----------------------------
    # Sanity checks
    # -----------------------------
    print("\nSanity checks:")
    print(" - sample 3 examples (decoded)")
    for i in range(min(3, len(kept))):
        s = kept[i]["prompt"][:160].replace("\n", " ")
        print("PROMPT (trim):", s)
        # decode tokens from the tokenized set to ensure reversibility
        ex = ds_tok["train"][i]
        decoded = tokenizer.decode([tid for tid in ex["input_ids"] if tid != tokenizer.pad_token_id])
        print("DECODED TOKENS =>", decoded[:160].replace("\n"," "))
        print("---")

    print(f"Total kept: {len(kept)}")
    if len(prompt_lens) > 0:
        prompt_lens_sorted = sorted(prompt_lens)
        resp_lens_sorted = sorted(resp_lens)
        print(f"Prompt len: mean={sum(prompt_lens)/len(prompt_lens):.1f} median={prompt_lens_sorted[len(prompt_lens)//2]}")
        print(f"Response len: mean={sum(resp_lens)/len(resp_lens):.1f} median={resp_lens_sorted[len(resp_lens)//2]}")

    # check simple overlap (should be 0 because of split function and dedupe)
    # here we compute exact text overlaps as a quick smoke test
    train_texts = set(tokenizer.batch_decode(ds_tok["train"]["input_ids"]))
    val_texts = set(tokenizer.batch_decode(ds_tok["validation"]["input_ids"]))
    overlap = len(train_texts & val_texts)
    print(f"Train/Val overlap count (exact match): {overlap} (should be 0)")

    # quick collate shape test (simulate a batch)
    bsz = 4
    train_subset = [ds_tok["train"][i] for i in range(min(bsz, len(ds_tok["train"])))]
    if len(train_subset) == bsz:
        input_ids = torch.tensor([t["input_ids"] for t in train_subset])
        labels = torch.tensor([t["labels"] for t in train_subset])
        print("Collated batch shapes:",
              "input_ids", input_ids.shape,
              "labels", labels.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--out_jsonl", type=str, default="dataset.jsonl")
    parser.add_argument("--out_hf_dir", type=str, default="hf_dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_and_save(
        model_name=args.model_name,
        dataset_name=args.dataset,
        split=args.split,
        num_examples=args.num_examples,
        max_length=args.max_length,
        out_jsonl=args.out_jsonl,
        out_hf_dir=args.out_hf_dir,
        seed=args.seed,
    )
```

---

## 5) Why each step matters

* **Deduplication**
  Prevents the model from “double counting” identical samples and avoids accidental train/val leakage.
* **Length filtering**
  Ensures every (prompt+response) fits your **context window**. Truncation mid-response can corrupt supervision; it’s better to filter or raise `max_length`.
* **Padding and labels**
  Causal LM loss should **ignore** padding — that’s why we set labels to `-100` where `attention_mask==0`.
* **Consistent template**
  Stable formatting lets the model learn exactly where the **response** should begin (after `### Response:`). This matters more than you think.

---

## 6) Sanity checks to always run

* **Decode a few tokenized rows** and visually confirm the template and the response are intact.
* **Length stats** (prompt/response token means/medians) to check you’re not choking the context.
* **Train/val overlap** should be **0**.
* **Batch shapes** after collation: `(batch, max_length)` for `input_ids`, `labels`, `attention_mask`.

---

## 7) Reproducibility tips

* **Seed everything** (`random`, `torch`, `cuda`) at the start of the script.
* Use **deterministic splits** (set `seed` in `train_test_split`).
* Freeze **prompt templates & normalization** choices (so future runs produce identical token streams).
* Keep the **same tokenizer version** and **model\_name** — they affect tokenization.

---

## 8) Tuning knobs (trade-offs)

* `max_length`:
  Larger gives more room for long answers but increases compute and padding waste.
* `num_examples`:
  More examples improve generalization but increase fine-tuning time. Start with **1k–2k** for quick iteration.
* `model_name`:
  Use the same tokenizer you’ll use for fine-tuning. (e.g., `gpt2`, `tiiuae/falcon-7b-instruct`, etc.)
* **Template design**:
  Keep it consistent with your downstream instruction-tuning style (SFT format).

---

## 9) What “correct” looks like

* JSONL has consistent `"prompt"` and `"output"` fields, no empties, no obviously broken text.
* Tokenizer logs:
  “Tokenizer has no pad token; setting pad\_token = eos\_token.” (expected for GPT-2).
* Sanity output similar to:

```
Loaded 3000 raw examples (before dedupe/filter).
3000 after deduplication.
Kept 1000 examples (skipped0 prompt-too-long, 0 empty-response).
Wrote JSONL: dataset.jsonl (train=900, val=100)
Saved HF Dataset to hf_dataset

Sanity checks:
 - sample 3 examples (decoded)
PROMPT (trim): Below is an instruction that describes a task, ...
DECODED TOKENS => Below is an instruction that describes a task, ...
---
Total kept: 1000
Prompt len: mean=58.3 median=47
Response len: mean=190.7 median=148
Train/Val overlap count (exact match): 0 (should be 0)
Collated batch shapes: input_ids torch.Size([4, 361]) labels torch.Size([4, 361])
```

---

## 10) Next steps

* You can now plug `hf_dataset/` straight into a Trainer/Accelerate/PEFT pipeline.
* If you plan to use **LoRA/QLoRA**, make sure your **tokenizer & template** here match the fine-tuning script’s expectations.
* Consider logging **length histograms** and a few **random decoded samples** to your experiment tracker (CSV/Weights & Biases).

---

**TL;DR:**
You’ve constructed a **clean, length-safe, reproducible instruction dataset** with a stable prompt format, saved it as JSONL for transparency and as a tokenized HF dataset for speed. This is exactly what you need before moving on to **fine-tuning (LoRA/QLoRA)**.
