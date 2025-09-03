import os                                # filesystem utilities
import math                              # for computing perplexity
import argparse                          # parse CLI args
import random
import numpy as np
import torch
from datasets import load_from_disk      # load the tokenized dataset saved earlier
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# -------------------------
# reproducibility utility
# -------------------------
def set_seed(seed: int):
    """
    Set random seeds for python, numpy, torch for reproducible runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------
# collator
# -------------------------
def data_collator(features, tokenizer):
    """
    Collate function for Trainer: pad input_ids and labels to longest in batch.
    - features: list of dicts each with 'input_ids' and 'labels'
    - tokenizer: tokenizer to get pad_token_id
    Returns a dict acceptable by Trainer
    """
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    # pad sequences to batch max length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune script (PEFT + HF Trainer)")
    # dataset & model
    parser.add_argument("--hf_dataset_dir", type=str, default="hf_dataset", help="tokenized HF dataset dir")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="base model id or path")
    parser.add_argument("--output_dir", type=str, default="outputs/lora_adapter", help="where to save LoRA adapter")
    # LoRA hyperparams
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha (scaling)")
    parser.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="*", default=None, help="module names to adapt (optional)")
    # training hyperparams
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="use fp16 (mixed precision) if available")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    args = parser.parse_args()

    # reproducible
    set_seed(args.seed)

    # load tokenized dataset from disk
    ds = load_from_disk(args.hf_dataset_dir)
    print("Dataset splits:", list(ds.keys()))
    print("Example columns:", ds["train"].column_names)

    # load tokenizer (same tokenizer used during dataset creation)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # ensure pad token exists 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # load base model
    # for moderate models (gpt2) this will fit; for large models use k-bit or device_map strategies
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # choose target modules if not provided
    if args.target_modules is None or len(args.target_modules) == 0:
        model_name_l = args.model_name_or_path.lower()
        if "gpt2" in model_name_l or "gpt" in model_name_l:
            target_modules = ["c_attn", "c_proj"]
        else:
            # common default for many HF transformer implementations
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = args.target_modules

    print("Using LoRA on modules:", target_modules)

    # create LoRA config
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=args.dropout,
        bias="none",           # simplest default (don't add extra bias params)
        task_type="CAUSAL_LM"
    )

    # wrap model with LoRA (this returns a PEFT-wrapped model)
    model = get_peft_model(model, lora_config)

    # sanity: print how many params are trainable
    model.print_trainable_parameters()

    # instantiate HF TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        remove_unused_columns=False,
        push_to_hub=False
    )

    # create the trainer (pass a collator closure that knows tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=lambda features: data_collator(features, tokenizer)
    )

    # start training (Trainer handles optimizer, grad accumulation, checkpointing)
    trainer.train()

    # save only the adapter weights (PEFT saves adapter config + weights)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)    # small — contains only LoRA adapters & config
    tokenizer.save_pretrained(args.output_dir)
    print("Saved LoRA adapter to", args.output_dir)

    # final eval
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)
    if eval_loss is not None:
        print("Final eval loss:", eval_loss, "ppl:", math.exp(eval_loss))
    else:
        print("No eval loss available in metrics")

if __name__ == "__main__":
    main()