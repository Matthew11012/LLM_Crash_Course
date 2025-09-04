# LoRA fine-tune & evaluation — Notes, walkthrough, and debugging log

*Comprehensive reference for the LoRA (PEFT) fine-tune of GPT‑2 on Alpaca-style data on 1000 example data*

---

## Summary

### 1000 Examples of Alpaca dataset
**Hyperparameters used for fine-tune:**

```
--model_name_or_path gpt2 \
--output_dir outputs/lora_adapter \
--r 8 --alpha 16 --dropout 0.1 \
--num_train_epochs 3 --per_device_train_batch_size 4 --fp16
```

**Evaluation result on:**

```json
{  
    "eval_loss": 2.157951602568993,
   "perplexity": 8.65339390169714,
   "num_batches": 13,
   "supervised_tokens": 11985,
   "split": "validation",
   "batch_size": 8,
   "max_length": 512 
}
```

### 10000 Examples of Alpaca dataset
**Hyperparameters used for fine-tune:**

```
--model_name_or_path gpt2 \
--output_dir outputs/lora_adapter \
--r 16 --alpha 32 --dropout 0.1 \
--num_train_epochs 3 --per_device_train_batch_size 4 --fp16
```

**Evaluation result on:**

```json
{  
    "eval_loss": 2.0961728372573853,
   "perplexity": 8.1349763807645240169714,
   "num_batches": 125,
   "supervised_tokens": 143095,
   "split": "validation",
   "batch_size": 8,
   "max_length": 512 
}
```

Perplexity ≈ **8.13** — means model assigns on average probability mass equivalent to choosing among \~8 tokens per position. Reasonable result for an instruction-tuned tiny model on a small dataset.

---

## Model Response Comparison
### Base vs 1000 Example LoRA Finetuned
```
Prompt: Explain why the sky is blue

KL divergence: nan
L2 distance:   1825.000000

=== Base model output ===
Explain why the sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

=== LoRA model output ===

Explain why the sky is blue and why it's blue.

The sky is blue because it is blue. It is blue because it is blue because it is blue because it is blue because it is blue because it is blue because it is blue because it is blue because it


Prompt: List the steps for making a peanut butter and jelly sandwich

KL divergence: nan
L2 distance:   3860.000000

=== Base model output ===
List the steps for making a peanut butter and jelly sandwich.

Step 1:

1. In a large bowl, combine the butter, sugar, and salt.

2. Add the eggs, milk, and vanilla.

3. Add the flour and mix well.

=== LoRA model output ===
List the steps for making a peanut butter and jelly sandwich.

Step 1: Preheat oven to 350 degrees F.

Step 2: In a large bowl, whisk together the butter, sugar, and vanilla.

Step 3: Add the eggs, vanilla, and salt.



Prompt: Write a short story about a robot who learns to paint.
KL divergence: nan
L2 distance:   5740.000000

=== Base model output ===
Write a short story about a robot who learns to paint.

The story is about a robot who learns to paint.

The story is about a robot who learns to paint.

The story is about a robot who learns to paint.

The story is about a robot who learns to

=== LoRA model output ===
Write a short story about a robot who learns to paint.

"I'm a robot," says the robot, "and I'm learning to paint."

"I'm learning to paint," says the robot, "and I'm learning to paint."

"I'm learning to paint,"


Prompt: Compare cats and dogs as pets in a few sentences.
KL divergence: nan
L2 distance:   487.500000

=== Base model output ===
Compare cats and dogs as pets in a few sentences.

"I'm not going to say that I'm a cat lover, but I'm not going to say that I'm a dog lover either," he said. "I'm not going to say that I'm a dog lover either."

=== LoRA model output ===
Compare cats and dogs as pets in a few sentences.

"I'm not sure if it's a good idea to have cats or dogs in your home, but I think it's a good idea to have them in your home," she said.

"I think it's important to have
```

### Base vs 10000 examples LoRA Finetuned
```
Prompt: Explain why the sky is blue
KL divergence: 0.228882
L2 distance:   3722.000000

=== Base model output ===
Explain why the sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

The sky is blue.

=== LoRA model output ===
Explain why the sky is blue and why it's blue.

The sky is blue because it is a complex system of colors, which means that it is composed of many different shades of blue, including red, green, and blue. The colors of the sky are also complex


Prompt: List the steps for making a peanut butter and jelly sandwich
KL divergence: 0.120178
L2 distance:   5072.000000

=== Base model output ===
List the steps for making a peanut butter and jelly sandwich.

Step 1:

1. In a large bowl, combine the butter, sugar, and salt.

2. Add the eggs, milk, and vanilla.

3. Add the flour and mix well.

=== LoRA model output ===
List the steps for making a peanut butter and jelly sandwich.

1. Preheat oven to 350 degrees F.

2. In a large bowl, whisk together the butter, sugar, and eggs.

3. Pour the mixture into a large bowl and whisk until smooth.


Prompt: Write a short story about a robot who learns to paint.
KL divergence: 0.407227
L2 distance:   8680.000000

=== Base model output ===
Write a short story about a robot who learns to paint.

The story is about a robot who learns to paint.

The story is about a robot who learns to paint.

The story is about a robot who learns to paint.

The story is about a robot who learns to

=== LoRA model output ===
Write a short story about a robot who learns to paint.

"I'm a robot, and I'm not a painter," says the robot, who is wearing a white suit and a blue tie. "I'm a robot, and I'm not a painter."

The robot is a robot


Prompt: Compare cats and dogs as pets in a few sentences.
KL divergence: 0.207520
L2 distance:   8080.000000

=== Base model output ===
Compare cats and dogs as pets in a few sentences.

"I'm not going to say that I'm a cat lover, but I'm not going to say that I'm a dog lover either," he said. "I'm not going to say that I'm a dog lover either."

=== LoRA model output ===
Compare cats and dogs as pets in a few sentences.

"I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but I'm


Prompt: Create a story about a robot that falls in love with a human.
KL divergence: 0.347168
L2 distance:   11000.000000

=== Base model output ===
Create a story about a robot that falls in love with a human.

The story is about a robot that falls in love with a human.

The story is about a robot that falls in love with a human.

The story is about a robot that falls in love with a human.

=== LoRA model output ===
Create a story about a robot that falls in love with a human.

The robot, named "Bumblebee," is a robotic robot that has been programmed to perform tasks such as picking up objects and moving them around. It's also capable of making complex calculations and making complex calculations on its own.
```
## 1 — What I trained and why

* I fine-tuned a **pretrained GPT‑2** model using **LoRA** (PEFT). LoRA injects *small low-rank adapters* into selected weight matrices (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj` or GPT‑2's `c_attn`, `c_proj`). In this case, I selected the `c_attn`, `c_proj` as the target modules.
* Advantages: tiny adapter file (MBs), freeze base model (no full-weight updates), cheap fine-tuning.
* I used Alpaca-style SFT data (instruction → response). The training objective is causal language modeling on the concatenated prompt+response, but the loss is masked to only penalize response tokens (prompt tokens get label `-100`).

---

## 2 — Files & artifacts

* `hf_dataset/` — tokenized dataset produced by Day 3 pipeline (train/validation splits). Each example contains `input_ids`, `attention_mask`, and `labels` where prompt tokens = `-100`.
* `train_lora.py` — training script that wraps base model with PEFT `get_peft_model(...)` and uses `Trainer`.
* `outputs/lora_adapter/` — saved adapter directory. Contains at least:

  * `adapter_config.json`
  * `adapter_model.safetensors` 
  * tokenizer files (optional if saved alongside)
* `eval_adapter.py` — evaluation script (loads base model + adapter with `PeftModel.from_pretrained(...)` and computes eval loss & perplexity).

---

## 3 — Key implementation details & rationale

### Tokenization & labels

* **Tokenizer pad handling:** GPT‑2 tokenizer has no PAD token by default. I set:

  ```py
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
  ```

  This allows padding during batching.

* **Supervised labels:** I concatenate `prompt + response` for each example and create labels as `[-100]*len(prompt_tokens) + response_tokens`. This means loss is computed only on the response portion and ignored on prompt tokens.

* **Truncation & padding:** Truncate `prompt+response` to `max_length` and then pad to `max_length`; label positions corresponding to pads are `-100` so they do not contribute to loss.

### Prompt template (Alpaca-style)

Use a consistent template to teach instruction format. Example template:

```
Below is an instruction that describes a task{', and an input that provides additional context' if input exists}.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}  # optional

### Response:
```

Keeping the exact template during train & eval is crucial — the adapter learns to respond in that style.

### Trainer & collator

*I used `transformers.Trainer` with `TrainingArguments`. `Trainer` handles optimizer, gradient accumulation, checkpointing and evaluation steps.
* Important: your `data_collator` must return properly padded tensors. Example collator (used in training):

```py
def data_collator(features):
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    labels = [torch.tensor(f["labels"]) for f in features]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
```

`Trainer` + `data_collator` produce batches that are correct shapes for the model.

---

## 4 — Why `Trainable params: 0 tensors` showed up (and why that was OK)

I observed `Trainable params (should be just LoRA adapters): 0 tensors` in the eval script. Here’s what I found:

* The adapter was saved with `inference_mode: true` in `adapter_config.json` (PEFT writes this flag when saving for inference). When I load via `PeftModel.from_pretrained(base_model, adapter_dir)`, PEFT respects `inference_mode=True` and will **freeze** adapter parameters (they do not `require_grad`). That leads to **zero trainable parameters** when you inspect `p.requires_grad` — because in evaluation mode nothing is trainable (expected for eval).

* **Important:** `0 tensors` does **not** mean the adapter is missing. The `Loaded adapters: {...}` print confirmed the adapter config was found. The adapter weights are present in `adapter_model.safetensors` and are applied during forward passes. They are just not trainable (no grads) during evaluation — which is the correct behaviour.

* If I wanted adapter params to be trainable (for continued finetuning / debugging), I could override that by re-configuring `is_trainable=True` or loading with `PeftModel.from_pretrained(..., is_trainable=True)` (but you rarely need this for eval).

---

## 5 — How to confirm the adapter **actually changes model outputs**

Run a before/after generation with the same prompt using (a) base model (no adapter) and (b) base + adapter. Example:

```py
# base
base = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
# with adapter
adapter_model = AutoModelForCausalLM.from_pretrained("gpt2")
adapter_model = PeftModel.from_pretrained(adapter_model, "outputs/lora_adapter").to(device)

prompt = "### Instruction:\nExplain why the sky is blue.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors='pt').to(device)

base_out = base.generate(**inputs, max_new_tokens=50)
adapter_out = adapter_model.generate(**inputs, max_new_tokens=50)

print("BASE:\n", tokenizer.decode(base_out[0], skip_special_tokens=True))
print("ADAPTER:\n", tokenizer.decode(adapter_out[0], skip_special_tokens=True))
```

The text should differ: the adapter-augmented model should display more instruction-like, helpful answers consistent with the Alpaca dataset.

If you want numerical confirmation, compare logits at the last step with `model(input_ids).logits[:, -1, :]` and see the KL or L2 differences between base and adapter logits.

---

## 6 — Common bugs I encountered (and solutions)

### 1) `accelerate` config file not found

**Symptom:** `FileNotFoundError: The passed configuration file accelerate_config.yaml does not exist.`

**Fix:** supply the absolute path to your `accelerate_config.yaml` or run `accelerate config` to generate the default file and then call `accelerate launch train_lora.py` without `--config_file`.

**Tip:** confirm the file name and working directory before invoking `accelerate`.

---

### 2) PyTorch CUDA capability mismatch warning

**Symptom:** `NVIDIA GeForce RTX 5060 ... not compatible with current PyTorch installation` and fallback to CPU.

**Cause:** PyTorch binary was built for different CUDA/compute capabilities.

**Fix:** install a PyTorch build that matches your CUDA driver/toolkit version (use the official instructions at pytorch.org) or run on CPU / move to a GPU instance that matches the binary. Alternatively, use Colab with a compatible runtime.

---

### 3) `TypeError: only integer tensors of a single element can be converted to an index` when calling `loss.item()`

**Symptom:** `loss` is not a scalar — sometimes comes back as a vector.

**Fix:** make sure to reduce to a scalar before `.item()`:

```py
loss = outputs.loss
if loss.dim() > 0:
    loss = loss.mean()
total_loss += loss.item()
```

---

### 4) `TypeError: 'tokenizers.Encoding' object cannot be interpreted as an integer` when using HuggingFace `tokenizers` fast API

**Symptom:** calling `torch.tensor([input_ids])` where `input_ids` was the return of `.encode(...)` rather than `.encode(...).ids`.

**Fix:** when using the `tokenizers` library directly, extract `ids`: `tokenizer.encode(text).ids`. If using `transformers` tokenizer, prefer `tokenizer(text)['input_ids']` or `tokenizer(text, return_tensors='pt')`.

---

### 5) `AttributeError: 'list' object has no attribute 'to'` in evaluation loop

**Symptom:** `batch['input_ids']` was a list rather than a tensor.

**Cause:** DataLoader returned Python lists because no collate function was provided.

**Fix:** use a collator that returns tensors, e.g. `transformers.data.data_collator.default_data_collator` or implement and pass a `collate_fn` that stacks/pads and returns tensors.

```py
from transformers import default_data_collator
dl = DataLoader(ds_tok, batch_size=8, collate_fn=default_data_collator)
for batch in dl:
    input_ids = batch['input_ids'].to(device)
```

Also ensure you spelled keys correctly (`labels`, not `lables`).

---

### 6) `adapter_model.safetensors` present instead of `.bin`

**Symptom:** you find `adapter_model.safetensors` in the adapter folder.

**Note:** This is expected and fine. `safetensors` is the recommended format — faster and safer. `PeftModel.from_pretrained(...)` will load it automatically.

---

## 7 — Evaluation script checklist (what it must do)

1. Load tokenizer and set `pad_token` if missing.
2. Load base model with `AutoModelForCausalLM.from_pretrained(base_model)`.
3. Attach adapter with `PeftModel.from_pretrained(base_model, adapter_dir)`.
4. Tokenize evaluation split using the *same prompt template* used for training and create labels masked with `-100` for prompts/padding.
5. Use `default_data_collator` or a custom collator to produce tensor batches.
6. Run forward passes and collect `outputs.loss` (mean over batch if necessary).
7. Compute `mean_loss = sum(batch_losses) / num_batches` and `ppl = exp(mean_loss)`.

A minimal example to run:

```bash
python eval_adapter.py \
  --base_model gpt2 \
  --adapter_path outputs/lora_adapter \
  --dataset_path hf_dataset \
  --split validation \
  --batch_size 8 \
  --max_length 512 \
  --device cpu
```

---

## 8 — Debugging tips & quick tests

* **Quick adapter presence test:** print `model.peft_config` and list modules containing `lora`:

```py
for name, module in model.named_modules():
    if "lora" in name.lower():
        print(name)
```

If this prints LoRA modules, the adapter is attached.

* **Quick output-difference test:** run `generate()` on base and adapter models for the same prompt and compare strings and/or logits.

* **Save reproducibility:** set a seed at the start of both training and evaluation:

```py
import random, torch, numpy as np
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
```

---

## 9 — Next steps & recommendations

* If you want to deploy the adapter in production: either load base+adapter with `PeftModel` at inference time or merge LoRA into base weights using PEFT merging utilities (`merge_and_unload()` or `PeftModel.merge_and_unload()`), then save a full model for efficient serving.

* For better evaluation, run a **pairwise human preference test** on 20–50 prompts: show base vs adapter outputs and record preferences. Aim for ≥60% adapter preference.

* If you plan to scale beyond GPT‑2, consider QLoRA (quantize base model to 4‑bit + LoRA) to fit much larger bases on limited GPU memory.

---

## 10 — Compact checklist to re-run / reproduce

1. Prepare tokenized dataset (`hf_dataset`) with same prompt template and labels masked to `-100` for prompts.
2. Train: `accelerate launch train_lora.py --hf_dataset_dir hf_dataset --model_name_or_path gpt2 --output_dir outputs/lora_adapter --r 8 --alpha 16 --dropout 0.1 --num_train_epochs 3 --per_device_train_batch_size 4 --fp16`
3. Validate adapter existence: `ls outputs/lora_adapter` (should show `adapter_config.json` and `adapter_model.safetensors`).
4. Evaluate: `python eval_adapter.py --base_model gpt2 --adapter_path outputs/lora_adapter --dataset_path hf_dataset --split validation --batch_size 8 --max_length 512 --device cpu`
5. Quick compare: run generation with/without adapter using the same prompt and compare outputs.

---
