# LoRA & PEFT — Deep-Dive Study Notes

*A practical, detailed guide you can keep as a reference. Covers the math, hyperparameters, parameter counts, QLoRA, and how it all fits into transformers.*

---

## 0) Quick Glossary

* **PEFT**: *Parameter-Efficient Fine-Tuning* — methods that adapt big models by training a tiny subset of parameters.
* **LoRA**: *Low-Rank Adaptation* — add small trainable matrices $A,B$ to selected weight matrices; freeze the base model.
* **QLoRA**: LoRA on a **quantized** (e.g., 4-bit) base model to further cut memory.
* **Target modules**: Which layers (e.g., attention projections) receive LoRA adapters.
* **Rank $r$**: The “width” (capacity) of the low-rank update. Higher $r$ → more capacity → more parameters.

---

## 1) Why LoRA?

Full fine-tuning retrains **all** weights $W_0$ of a large model — very expensive in compute and memory.
**LoRA freezes $W_0$** and learns a **small** additive update $\Delta W$ **per target layer**:

$$
W \;=\; W_0 \;+\; \underbrace{\frac{\alpha}{r} \, B A}_{\text{tiny, trainable update}}
$$

* You only train $A,B$ (millions → **hundreds of thousands** or less).
* Base model remains intact.
* You can swap different adapters for different tasks on the same base.

**Intuition:** We don’t need to change the whole mountain; we just “nudge” a few important slopes.

---

## 2) Where LoRA is inserted in Transformers

Transformers use linear projections heavily:

* **Attention**: $Q = X W_Q,\; K = X W_K,\; V = X W_V,\; \text{Out} = \text{Attn}(Q,K,V) W_O$
* **Feed-Forward**: $X \to X W_1 \to \text{act} \to X W_2$

LoRA is typically applied to the **attention projections** (`q_proj`, `k_proj`, `v_proj`, sometimes `o_proj`) and optionally FFN layers (`up_proj`, `down_proj`, `gate_proj` depending on architecture).

Why attention first? Small changes in the attention projections often yield **big behavioral gains**.

---

## 3) The LoRA math

Let a target linear layer have frozen weight $W_0 \in \mathbb{R}^{d \times k}$.
LoRA adds a low-rank correction:

$$
\Delta W \;=\; B A,\quad A \in \mathbb{R}^{r \times k},\; B \in \mathbb{R}^{d \times r}
$$

Forward pass becomes:

$$
Y \;=\; X W_0^\top \;+\; \underbrace{\frac{\alpha}{r} \, X (BA)^\top}_{\text{LoRA path}}
$$

* **Only $A,B$** have `requires_grad=True`.
* $W_0$ is **frozen** (no gradients or updates).
* $\alpha$ (*lora\_alpha*) scales the update; division by $r$ normalizes magnitude across different ranks.

**Variance intuition:** As $r$ increases, the sum of contributions grows. Scaling by $\alpha/r$ keeps the update’s **expected scale** stable so training behaves similarly across ranks.

---

## 4) Hyperparameters 

### 4.1 `r` (rank)

* Controls capacity and parameter count of each adapter.
* Params per adapter (ignoring bias): $\text{params} = r \cdot (d + k)$.
* Typical: **4, 8, 16, 32**.
  Start with **8 or 16** for medium models; go higher only if underfitting.

### 4.2 `lora_alpha` (scaling)

* Multiplier for the LoRA update (used as $\alpha/r$).
* Often set to **r** or **2r** (e.g., if $r=8$, $\alpha = 8$ or $16$).
* If updates feel too weak/strong, nudge `lora_alpha` rather than `r`.

### 4.3 `lora_dropout`

* Dropout applied on the **LoRA path** (not the base layer).
* Helps regularize and avoid overfitting on small datasets.
* Common values: **0.0–0.1**.

### 4.4 `target_modules`

* Which submodules get LoRA (strings matching module names).
* For LLaMA-like: `["q_proj","k_proj","v_proj","o_proj"]`.
  For GPT-2-like: `["c_attn","c_proj"]` (note: GPT-2 fuses qkv into one `c_attn`).
* **Important**: Must match your model’s actual module names.

### 4.5 `bias`

* `"none"` (default): no extra bias trained — simplest & most memory-efficient.
* `"lora_only"`: train bias only for LoRA layers.
* `"all"`: train all biases — defeats some efficiency; rarely needed.

### 4.6 `task_type`

* Tells PEFT how to wrap the base model (e.g., `TaskType.CAUSAL_LM`).
* Ensures correct integration into forward pass & saving/loading.

### 4.7 Other (occasionally relevant)

* `modules_to_save`: save extra modules (e.g., classification head) along with adapters.
* `fan_in_fan_out`: for unusual layer definitions (rare in modern LLMs).
* `inference_mode`: whether to freeze adapter grads; for serving.

---

## 5) Parameter Counting 

Assume a block has a projection $W \in \mathbb{R}^{d \times k}$.
LoRA params per such **one** projection:

$$
\#\text{params} = r\cdot k \;+\; d\cdot r = r\,(k+d)
$$

### Example A — GPT-2 Small–ish dims

* $d = k = 768$
* $r = 8$

Per projection:

$$
8 \cdot (768+768) = 8 \cdot 1536 = 12{,}288
$$

Per **attention** (Q, K, V, O = 4 projections):

$$
4 \cdot 12{,}288 = 49{,}152
$$

Per **layer** (just attention): **49,152** params.

For **12 layers**:

$$
12 \cdot 49{,}152 = 589{,}824 \approx 0.59\text{M params}
$$

**Compare**: GPT-2 small has \~**124M** parameters.
Training **0.6M** instead of **124M** ⇒ \~**200× fewer** trainable params.

If you also LoRA the FFN (2 more large linears), params go up proportionally — still tiny vs full fine-tuning.

### Example B — LLaMA-7B attention projection (illustrative)

Rough dims: $d=k=4096$, $r=16$.
Per projection:

$$
16 \cdot (4096+4096) = 16 \cdot 8192 = 131{,}072
$$

Four projections → **524,288** per layer.
32 layers → **\~16.8M** (still small compared to **7B**!).

> **Takeaway:** LoRA scales with $r$ and dims of **targeted** layers only. You can choose fewer targets for even leaner adapters.

---

## 6) Memory & Optimizer State Savings

With Adam, each trainable parameter usually carries **m** and **v** FP32 states (2×).

* **Full FT**: params + grads + Adam states for **all** weights → huge.
* **LoRA**: **only $A,B$** have grads & Adam states.
  Base $W_0$ stays frozen (no optimizer states there).

This is why LoRA fits on modest GPUs and trains **fast**.

---

## 7) How training actually flows

* **Forward**:
  $Y = X W_0^\top + \frac{\alpha}{r} X (BA)^\top$
  (base path + LoRA path)

* **Backward**:
  Gradients flow **only** into $A,B$.
  $W_0$ has `requires_grad=False`.

* **Initialization**:
  Typically $B$ is zeros and $A$ is small random (or vice versa), so the model **starts as the base** and gradually learns the delta.

---

## 8) Inference & Serving

At inference you need:

* The **base model** (frozen weights $W_0$).
* The **LoRA adapter weights** (small file with $A,B$).

Serving applies $W_0 + \frac{\alpha}{r} BA$ on the fly.
You can also **merge** the adapter into the base temporarily for speed:

```python
# Pseudocode idea (libraries differ)
model = PeftModel.from_pretrained(base, adapter_id)
model = model.merge_and_unload()  # folds LoRA into base weights for inference
```

> **Adapters are “stackable”** — same base, different adapters per downstream task.

---

## 9) QLoRA — LoRA on a Quantized Base

**What changes:**

* The base model $W_0$ is stored in **4-bit** quantization (e.g., **NF4** in bitsandbytes).
* **LoRA adapters remain in FP16/BF16/FP32** (exact dtype depends on setup).
* Training computes in higher precision while reading the quantized base.

**Why it works:**

* The heavy memory is in the base; quantize it to save **\~4–8× memory**.
* Keep adapters high-precision so learning remains stable.
* Overall: **nearly full-FT quality** with **tiny memory** footprint.

**Common recipe:**

* Load base with 4-bit quantization (bitsandbytes).
* Wrap with LoRA (PEFT).
* Train only adapters; use a **paged optimizer** to avoid OOM.
* Save adapters as a small file; reload with the quantized base to serve.

---

## 10) Putting It Together — Minimal Code Sketch

> *For illustration; exact names differ by model family.*

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

base_id = "gpt2"  # or a LLaMA-like model
tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(base_id)  # CPU or GPU

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn","c_proj"],  # GPT-2 style
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()  # sanity check

# ... your usual training loop only updates LoRA params ...
# optimizer over model.parameters() will include only A,B (and any allowed biases)

# Save just the adapters (small):
model.save_pretrained("my-lora-adapter")

# Later:
# base = AutoModelForCausalLM.from_pretrained(base_id)
# peft_model = PeftModel.from_pretrained(base, "my-lora-adapter")
# peft_model.generate(...)
```

---

## 11) Practical Tips & Rules of Thumb

* **Start small**: `r=8`, `lora_alpha=16`, `lora_dropout=0.05`.
* **Target attention first**; expand to FFN if underfitting.
* **Tokenizer/domain** mismatch hurts more than small `r`.
* **Regularize** if your dataset is small (dropout, early stopping).
* **Seed for reproducibility**; save adapters + training config.
* **Validate** by freezing base and logging **only** adapter parameter counts.

---

## 12) Sanity Checks You Should Run

1. **Trainable-params printout** should show a tiny fraction vs total.
2. **Forward equivalence at init**: with zero-init on one of $A,B$, model output initially $\approx$ base output.
3. **Loss drops on a tiny batch** (overfit test).
4. **Adapter swap**: same base + different adapters produce different behaviors.

---

## 13) FAQ (Conceptual)

**Q1. Are A and B “new weights” used at inference?**
**Yes.** Inference uses $W = W_0 + \frac{\alpha}{r} BA$. The adapters are not just a training trick — they actively modify the layer.

**Q2. If $r$ increases, why divide by $r$?**
To keep update **magnitude stable** as you add more rank components. $\alpha$ controls strength; $\alpha/r$ normalizes across ranks.

**Q3. Where exactly are adapters applied?**
Wherever you put them via `target_modules`. Most common: **Q, K, V (and O)** projections in attention. Sometimes FFN projections too.

**Q4. What is QLoRA conceptually?**
Same LoRA math. The **only difference** is the **base** $W_0$ is **4-bit quantized** to save memory; adapters are trained in higher precision.

**Q5. What do I ship after fine-tuning?**
You can:

* Ship **base model + small adapter file** (recommended), or
* **Merge** LoRA into base weights for a monolithic model (less flexible).

---

## 14) Extra: Numeric Memory Intuition

Suppose:

* Base model = **124M** params (GPT-2 small).
* FP16 params ≈ **250 MB**; with Adam states (m,v) FP32 → **>1 GB** for full FT.
* **LoRA (r=8)** adapters on attention only ≈ **0.6M** params:
  FP16 weights + grads + Adam states ⇒ **single-digit MBs**.
  **This is the whole point.**

With **QLoRA** (4-bit base):

* Base weights down to **\~1/4** FP16 memory.
* Adapters unchanged (still tiny).
* Fits on a single consumer GPU easily for many tasks.

---

## 15) Derivation Snapshot (why low-rank helps)

A dense update $\Delta W \in \mathbb{R}^{d \times k}$ has $d \cdot k$ degrees of freedom.
Empirically, **task-specific updates lie in a low-dimensional subspace**.
Approximating $\Delta W \approx BA$ with rank $r \ll \min(d,k)$ captures most useful variation with far fewer parameters — the **low-rank hypothesis** behind LoRA.

---

## 16) Reproducibility Checklist

* Fix seeds (`torch`, `numpy`, Python `random`).
* Log and save: `r`, `lora_alpha`, `dropout`, `target_modules`, optimizer, LR schedule, tokenizer SHA, base model commit hash.
* Save: **adapter weights** + **config** + a **sample output** for the same seed/prompt.

---

## 17) Common Pitfalls

* **Wrong `target_modules` names** (nothing trains). Inspect model to confirm names.
* **Too small `r`** (underfitting) or **too big** (overfit/instability).
* **For GPT-2**: qkv might be **fused**; use `c_attn`.
* **For LLaMA**: typical names are `q_proj`, `k_proj`, `v_proj`, `o_proj`; FFN is `gate_proj`, `up_proj`, `down_proj`.
* **Tokenizer mismatch**: hurts way more than LoRA hyperparams.

---

## 18) Tiny Parameter Count Helper (optional)

```python
def lora_params_per_linear(d, k, r):
    return r * (d + k)

# Example: d=k=768, r=8
print(lora_params_per_linear(768, 768, 8))  # -> 12288
```

---

## 19) One-Screen TL;DR

* Freeze base $W_0$; train tiny $A,B$ so $W = W_0 + \frac{\alpha}{r} BA$.
* Put adapters on attention projections first (`q/k/v/o`).
* Start with `r=8–16`, `lora_alpha=2r`, `lora_dropout=0.05`.
* Train only adapters → **huge** memory/compute win.
* QLoRA = same idea + 4-bit base for even bigger savings.
* Ship base + adapter; swap adapters per task.

---

