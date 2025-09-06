import os
import time
import torch
import csv

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BENCHMARK_FILE = "raw_results.csv"

PROMPT = "Explain why the sky is blue."
BATCH_SIZE = [1,2,4,8]
MAX_NEW_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "gpt2"                 
ADAPTER_PATH = "../../week2/outputs/lora_adapter"  

def benchmark_hf(load_in_8bit=True):
    precision = "int8" if load_in_8bit else "fp16"

    print(f"\n[HF] Loading {BASE_MODEL} with adapters ({precision})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        )
    model = PeftModel.from_pretrained(base_model,ADAPTER_PATH,).eval().to(DEVICE)

    for batch_size in BATCH_SIZE:
        batch_prompts = [PROMPT] * batch_size
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        latency = time.time() - start
        gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        tokens_per_sec = gen_tokens / latency
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[{precision}] batch={batch_size} | latency={latency:.3f}s | mem={peak_mem:.1f}MB | speed={tokens_per_sec:.2f} tok/s")

        log_results("HF+bitsandbytes", precision, batch_size, MAX_NEW_TOKENS, latency, peak_mem, tokens_per_sec)


def log_results(framework, precision, batch_size, max_new_tokens, latency, peak_mem, tokens_per_sec):
    row = [framework, precision, batch_size, max_new_tokens, round(latency, 3), round(peak_mem, 1), round(tokens_per_sec, 2)]
    header = ["framework", "precision", "batch_size", "max_new_tokens", "latency_s", "peak_mem_MB", "tokens/sec"]

    write_header = not os.path.exists(BENCHMARK_FILE)
    with open(BENCHMARK_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"[LOGGED] {row}")

if __name__ == "__main__":
    benchmark_hf(load_in_8bit=False)   # fp16 baseline
    benchmark_hf(load_in_8bit=True)    # int8 quantized

