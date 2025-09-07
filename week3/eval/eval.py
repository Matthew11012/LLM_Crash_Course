import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from sentence_transformers import SentenceTransformer, util
import csv
from peft import PeftModel

BASE_MODEL = "gpt2"
ADAPTER_PATH = "../../week2/outputs/lora_adapter"

# Prompts for human check
PROMPTS = [
    "Explain why the sky is blue",
    "List the steps for making a peanut butter and jelly sandwich",
    "Write a short story about a robot who learns to paint.",
    "Compare cats and dogs as pets in a few sentences.",
    "Summarize the causes of the French Revolution in 3 sentences.",
    "Write a haiku about winter mornings.:",
    "Explain how photosynthesis works to a 5th grader.",
    "Translate this sentence into Spanish: Knowledge is power.",
    "What are the advantages and disadvantages of electric cars?",
    "Give step-by-step instructions for tying a shoelace.",
    "Imagine a world where gravity is half as strong. Describe how daily life would change.",
    "Write a dialogue between a doctor and a patient with a cold.",
    "List the first 5 prime numbers and explain why 4 is not prime.",
    "Compose a short motivational message for someone taking an exam."
]

def load_model(quantized=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if quantized:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            quantization_config=quant_config, 
            device_map="auto"
            )
        model_with_lora = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model_with_lora.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.float16, 
            device_map="auto"
            )
        model_with_lora = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model_with_lora.to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def compute_perplexity(model, tokenizer, max_examples=100):
    model.eval()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    losses = []

    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        input = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=256).to(model.device)

        # dataset may contain just "\n" or whitespace
        text = example["text"].strip()
        if not text:  # skip empty lines
            continue

        with torch.no_grad():
            outputs = model(**input, labels=input["input_ids"])
        losses.append(outputs.loss.item())
    
    mean_loss = sum(losses) / len(losses)
    return torch.exp(torch.tensor(mean_loss)).item() # perplexity

def generate_outputs(model, tokenizer, prompts, max_new_tokens=50):
    outputs = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return outputs

if __name__ == "__main__":
    # Load models
    fp16_model, tokenizer = load_model(quantized=False)
    int8_model, _ = load_model(quantized=True)

    # --- Perplexity ---
    ppl_fp16 = compute_perplexity(fp16_model, tokenizer)
    ppl_int8 = compute_perplexity(int8_model, tokenizer)
    print(f"[PPL] FP16={ppl_fp16:.2f}, Int8={ppl_int8:.2f}, Δ={(ppl_int8-ppl_fp16)/ppl_fp16:.2%}")

    # --- Generation + embedding similarity ---
    fp16_outs = generate_outputs(fp16_model, tokenizer, PROMPTS)
    int8_outs = generate_outputs(int8_model, tokenizer, PROMPTS)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    emb_fp16 = embedder.encode(fp16_outs, convert_to_tensor=True)
    emb_int8 = embedder.encode(int8_outs, convert_to_tensor=True)
    sims = util.cos_sim(emb_fp16, emb_int8).diagonal()
    print(f"[SIM] Mean cosine similarity = {sims.mean().item():.3f}")

    # --- Save CSV for human eval ---
    with open("quant_eval_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "FP16 Output", "Int8 Output"])
        for p, o1, o2 in zip(PROMPTS, fp16_outs, int8_outs):
            writer.writerow([p, o1, o2])

    print("[DONE] Saved quant_eval_samples.csv for human evaluation.")

