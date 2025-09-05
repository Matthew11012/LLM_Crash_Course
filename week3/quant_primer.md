
---

# Quantization Primer (LLMs)

**Goal:** explain post-training quantization (PTQ) vs. quantization-aware training (QAT), how 8-bit and 4-bit schemes work, and how to choose for your constraints (speed, memory, quality). Examples reference common LLM tooling (Hugging Face + bitsandbytes / GPTQ / ONNX).

---

## 1) Why quantize?

Modern LLMs ship weights in float16/float32. That’s accurate but heavy:

* **Memory:** fp16 needs 2 bytes/param; int8 needs 1; int4 needs 0.5.
* **Bandwidth:** smaller weights = faster loading and faster matmuls (often).
* **Cost:** fit bigger models on the same GPU, or run on commodity hardware.

Trade-off: lower precision can hurt accuracy, especially for smaller models or tasks with tight numerical margins.

---

## 2) Where quantization applies

* **Weights**: the model’s parameters (W). Most methods quantize these.
* **Activations**: tensors produced during the forward pass (x). Quantizing activations can bring extra speed but is more fragile.
* **KV cache**: attention key/value tensors stored during generation; quantizing these saves a lot of memory for long contexts.
* **Gradients** (during training/QAT): usually left in higher precision.

---

## 3) PTQ vs QAT

### PTQ — Post-Training Quantization

* **What:** take a trained fp16/fp32 model; quantize weights (and maybe activations) **without** retraining.
* **How:** choose scales/zero-points using simple stats or a small **calibration set** (a few hundred–few thousand tokens/sentences).
* **Pros:** fast, simple, no training loop, cheap.
* **Cons:** some accuracy drop vs. QAT, especially at int4; sensitive to calibration.

**Popular PTQ methods for LLMs**

* **LLM.int8() / 8-bit matrix multiply (bitsandbytes)**: outlier-aware per-channel weight quant, keeps some rows in fp16 to preserve outliers.
* **NF4 4-bit (bitsandbytes)**: normal-float 4-bit quantizer on weights with fp16 compute. Very memory-efficient with good quality for many models.
* **GPTQ** (AutoGPTQ, exllama, etc.): layer-by-layer quant that minimizes the output error of each layer using calibration data; strong quality at low bit-width.
* **AWQ**: activation-aware weight quantization; selects “important” channels to keep in higher precision or scale differently.

### QAT — Quantization-Aware Training

* **What:** simulate quantization during training/fine-tuning so the model **learns** to be robust to low precision.
* **Pros:** best accuracy under aggressive quant (esp. int4 or lower).
* **Cons:** needs training data, compute, and time; more complex setup.

**When to prefer QAT:** if you must keep near-baseline quality at **very** low bit-widths (4-bit and below) **and** you can afford a short fine-tune.

---

## 4) 8-bit vs 4-bit (and friends)

| Scheme              | Typical Use      | Memory vs fp16 |               Quality | Notes                                                                |
| ------------------- | ---------------- | -------------: | --------------------: | -------------------------------------------------------------------- |
| **int8**            | Safe default PTQ |          \~50% |           \~near-fp16 | Great speed/memory tradeoff; widely supported.                       |
| **int4 (NF4/GPTQ)** | Max compression  |          \~25% | small → moderate drop | Needs good calibr.; often surprisingly strong on many LLMs.          |
| **int2 / ternary**  | Researchy        |        \~12.5% |            large drop | Requires QAT and careful design.                                     |
| **Mixed-precision** | Practical        |        between |         close to fp16 | Keep sensitive layers (embeds, lm\_head) in fp16; quantize the rest. |

**Precision knobs**

* **Per-tensor vs per-channel:** per-channel (per-row) scaling preserves accuracy better.
* **Symmetric vs asymmetric:** asymmetric adds zero-point; helps when distributions are shifted.
* **Rounding:** stochastic vs nearest; minor but measurable effects.
* **Outlier handling:** don’t squash rare but large channels; keep them in higher precision or scale separately.

---

## 5) Calibration (for PTQ)

* Use a **representative** slice of your expected inputs (e.g., 512–4k prompts).
* Cover your **target domain** (style, length, languages).
* Longer sequences help tools estimate realistic activation ranges.
* For GPTQ/AWQ, follow the tool’s docs for sample size (often \~128–1024 sequences).

---

## 6) What to quantize (practical picks)

* **Weights only** (int8 or int4): simplest and already gives big savings.
* **Weights + KV cache**: huge memory wins for long generation; ensure your runtime supports it.
* **Activations**: extra speed but more brittle; often left in fp16/bf16 in practice.

---

## 7) Quality risks & mitigations

**Risks**

* Loss spikes on certain layers (attention projections, small MLPs).
* Tasks needing numerical sharpness (math, code) degrade earlier.
* Small models are more sensitive than large ones.

**Mitigations**

* Keep **embeddings** and **lm\_head** in fp16.
* Use **per-channel** weight quant; outlier-aware methods (LLM.int8(), AWQ).
* Try **int8 first**, then evaluate int4.
* For int4, prefer **NF4** (bitsandbytes) or **GPTQ** with a good calibration set.
* If needed, do a short **LoRA QAT** on your domain data.

---

## 8) Minimal recipes

> **bitsandbytes (8-bit, outlier-aware PTQ, Hugging Face Transformers)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,       # or load_in_4bit=True
    llm_int8_threshold=6.0,  # outlier threshold
    llm_int8_has_fp16_weight=True,  # keep a fp16 copy (speed/memory tradeoff)
    # For 4-bit:
    # bnb_4bit_quant_type="nf4",  # ["nf4", "fp4"]
    # bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config,   # HF >= 4.33 style
    device_map="auto",
)
```

> **AutoGPTQ (4-bit GPTQ PTQ)**

* Quantize once with calibration → save `.safetensors`.
* Load the quantized checkpoint for inference:

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load already-quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "your-quantized-repo-or-path",
    device="cuda:0",
    use_safetensors=True,
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained("your-quantized-repo-or-path", use_fast=True)
```

> **ONNX Runtime PTQ (CPU/GPU portable)**

1. Export to ONNX, 2) run dynamic/static quantization:

```bash
python -m onnxruntime_tools.transformers.convert_to_onnx -m gpt2 -o gpt2.onnx
# Then quantize (dynamic int8):
python -m onnxruntime.quantization.quantize_dynamic \
  --model_input gpt2.onnx --model_output gpt2-int8.onnx \
  --per_channel --optimization_level 99
```

---

## 9) Measuring the impact

* **Memory:** report peak/steady resident memory (nvidia-smi) and model size on disk.
* **Latency/throughput:** tokens/sec; prompt-processing time; batch effects.
* **Quality:**

  * **Perplexity** on your validation split (lower = better).
  * **Task metrics** (exact match, BLEU/ROUGE for summarization, etc.).
  * **Manual eval**: side-by-side generation for 20–50 prompts.

**Rule of thumb**

* int8 weights-only PTQ often keeps perplexity within \~0–5% of fp16.
* int4 PTQ (good method + calibration) might be \~5–15% worse, but varies by model/task.

---

## 10) Choosing for *your* constraint

**You want:** *“quick win, minimal effort, solid quality”*
→ **PTQ int8 weights** (bitsandbytes LLM.int8()).

* Easiest path, big memory savings, likely negligible quality loss.

**You want:** *“fit a bigger model / very tight VRAM”*
→ Try **PTQ int4** with **NF4** (bitsandbytes) or **GPTQ** (AutoGPTQ).

* Validate perplexity + a few manual prompts.
* Keep embeds/head in fp16 if possible.

**You want:** *“the best possible low-bit quality”* and can fine-tune
→ **QAT (LoRA)** at the target precision (e.g., 4-bit weights) on your domain data.

* Short supervised fine-tune often recovers a lot of quality.

**You care about deployment portability (CPU, diverse GPUs)**
→ **ONNX Runtime PTQ** (int8 dynamic) for broad hardware; great for serving on CPU.

---

## 11) Practical gotchas

* **Tooling/platform support** varies (especially for 4-bit on different OS/GPUs). Check your CUDA + driver + package versions.
* **Generation config** matters: temperature/top-p changes can mask or exaggerate quality differences.
* **Calibration bias:** If your calibration set doesn’t match production, quantization might overfit to the wrong distributions.
* **KV cache & long contexts:** For long prompts, KV cache dominates memory. Consider runtimes that quantize KV cache too.
* **Mixed precision traps:** Accidentally quantizing embeddings or lm\_head can hurt quality; many recipes exclude them on purpose.

---

## 12) TL;DR recommendation

For this project’s constraints (quick, practical fine-tuning and deployment):

1. **Start with PTQ int8 weights** (bitsandbytes `load_in_8bit=True`).

   * Validate perplexity on your val split and a small manual eval set.
2. If you need more headroom, **try int4 NF4** (weights-only).

   * Keep embeddings and lm\_head in fp16 if your framework allows.
3. If int4 quality is not enough, **do a light LoRA QAT** on your task data.

This path usually gets you 2–4× memory savings with modest or minimal quality loss, plus a straightforward serving story.

---

### Appendix: A note on scales & zero-points

A uniform affine quantizer maps real values $x$ to integers $q$ via:

$$
q = \text{round}\left(\frac{x}{s}\right) + z,\quad
x \approx s \cdot (q - z)
$$

* $s$: scale,
* $z$: zero-point (asymmetric); set $z=0$ for symmetric.

Choosing $s$ and $z$ per channel (e.g., per output channel of a linear layer) better matches weight distributions and reduces error.

---

### Appendix: Quick checklists

**Before quantizing**

* [ ] Verify baseline: eval perplexity + manual prompts.
* [ ] Pick a calibration set (PTQ).
* [ ] Decide exclusions (embeddings, lm\_head).

**After quantizing**

* [ ] Re-run perplexity and manual prompts.
* [ ] Measure memory and tokens/sec.
* [ ] Compare a few logits/prob distributions (optional sanity).

---