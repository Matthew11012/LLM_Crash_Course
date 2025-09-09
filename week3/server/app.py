import os
import time
import asyncio
import typing as t
from functools import partial

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cachetools import LRUCache
from cachetools.keys import hashkey
from starlette.responses import JSONResponse

# ---------- Config (edit these) ----------
MODEL_ID = os.environ.get("MODEL_ID", "gpt2")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", None)  # e.g. "../../week2/outputs/lora_adapter"
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 8))
MAX_WAIT_MS = int(os.environ.get("MAX_WAIT_MS", 40))   # how long to wait to form a batch (ms)
LRU_CACHE_SIZE = int(os.environ.get("LRU_CACHE_SIZE", 2048))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 64))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.8))
TOP_K = int(os.environ.get("TOP_K", 50))
TOP_P = float(os.environ.get("TOP_P", 0.95))
QUANTIZE = os.environ.get("QUANTIZE", "false").lower() in ("1", "true", "yes")

# ---------- Request / Response schemas ----------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: t.Optional[int] = None
    temperature: t.Optional[float] = None
    top_k: t.Optional[int] = None
    top_p: t.Optional[float] = None
    seed: t.Optional[int] = None  # for determinism if requested

class GenerateResponse(BaseModel):
    text: str
    time_s: float
    cached: bool = False

# ---------- App + state ----------
app = FastAPI(title="LLM Batching Server")
request_queue: asyncio.Queue = asyncio.Queue()
cache = LRUCache(maxsize=LRU_CACHE_SIZE)

# Simple counters for metrics
METRICS = {"requests": 0, "served": 0, "cache_hits": 0, "batch_calls": 0}

# ---------- Model loading ----------
def load_model():
    print(f"Loading model: {MODEL_ID} quantize={QUANTIZE} device={DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if QUANTIZE:
        # NOTE: adjust BitsAndBytesConfig params for your environment
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32, device_map="auto")

    if ADAPTER_PATH:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    else:
        model = base

    # put in eval mode
    model.eval()
    # move to device only if not already handled by device_map
    try:
        if isinstance(model, torch.nn.Module) and next(model.parameters()).device.type != DEVICE:
            model.to(DEVICE)
    except StopIteration:
        pass

    return model, tokenizer

MODEL, TOKENIZER = load_model()


# ---------- Batching worker ----------
class PendingRequest:
    def __init__(self, prompt: str, params: dict, future: asyncio.Future):
        self.prompt = prompt
        self.params = params
        self.future = future

async def batch_worker():
    """
    Background task that gathers requests from `request_queue` and runs model.generate once.
    Uses MAX_BATCH_SIZE and MAX_WAIT_MS to control batching.
    """
    while True:
        # gather first request (blocking)
        first: PendingRequest = await request_queue.get()
        batch = [first]
        start = time.time()
        # accumulate until MAX_BATCH_SIZE or MAX_WAIT_MS
        while len(batch) < MAX_BATCH_SIZE:
            wait_ms = MAX_WAIT_MS / 1000.0
            try:
                req = await asyncio.wait_for(request_queue.get(), timeout=wait_ms)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        METRICS["batch_calls"] += 1

        # prepare batched inputs
        prompts = [p.prompt for p in batch]
        params_list = [p.params for p in batch]

        # all items use the same tokenizer; pad them
        enc = TOKENIZER(prompts, return_tensors="pt", padding=True, truncation=True).to(next(MODEL.parameters()).device)

        # model generate args: we pick per-request parameters by taking first's or using defaults
        # For simplicity we use the max of requested max_new_tokens to generate once, then slice.
        max_new = max([p.get("max_new_tokens", MAX_NEW_TOKENS) or MAX_NEW_TOKENS for p in params_list])
        temperature = params_list[0].get("temperature", TEMPERATURE)
        top_k = params_list[0].get("top_k", TOP_K)
        top_p = params_list[0].get("top_p", TOP_P)
        seed = params_list[0].get("seed", None)

        # reproducibility (optional): set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # run generation in inference mode
        with torch.inference_mode():
            try:
                # generate with batch; returns tensor [batch, seq_len]
                outputs = MODEL.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask", None),
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            except Exception as e:
                # If generation fails, set exception on futures and continue
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)
                continue

        # convert outputs back to strings and dispatch per request
        # outputs are aligned to prompts order
        for idx, pending in enumerate(batch):
            text = TOKENIZER.decode(outputs[idx].tolist(), skip_special_tokens=True)
            # if client requested shorter generation than produced, we can try to trim heuristically:
            # Note: trimming by tokens is tricky: we could re-tokenize and slice to requested length
            requested_new = pending.params.get("max_new_tokens", MAX_NEW_TOKENS) or MAX_NEW_TOKENS
            if requested_new < max_new:
                # re-tokenize generated text and keep only prompt + requested_new tokens
                full_ids = outputs[idx].tolist()
                input_len = enc["input_ids"].shape[1] if idx == 0 else (enc["input_ids"].shape[1] if False else None)
                # easiest robust approach: decode and then re-encode to limit tokens
                gen_text = TOKENIZER.decode(full_ids, skip_special_tokens=True)
                gen_ids = TOKENIZER.encode(gen_text).ids if hasattr(TOKENIZER.encode, "__call__") else TOKENIZER(gen_text)["input_ids"]
                # last requested_new tokens
                trimmed_ids = gen_ids[: min(len(gen_ids), enc["input_ids"].shape[1] + requested_new)]
                text = TOKENIZER.decode(trimmed_ids, skip_special_tokens=True)

            # set result
            if not pending.future.done():
                pending.future.set_result({"text": text, "time_s": time.time() - start})
        # mark served
        METRICS["served"] += len(batch)


# start background worker on startup
@app.on_event("startup")
async def startup_event():
    app.state.worker = asyncio.create_task(batch_worker())
    print("Batching worker started.")

# ---------- Helpers ----------
def cache_key(prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float):
    # hashable key for cache (include decoding settings)
    return hashkey(prompt, max_new_tokens, temperature, top_k, top_p)

# ---------- Endpoints ----------
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    METRICS["requests"] += 1
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    max_new = req.max_new_tokens or MAX_NEW_TOKENS
    temp = req.temperature or TEMPERATURE
    topk = req.top_k or TOP_K
    topp = req.top_p or TOP_P

    key = cache_key(prompt, max_new, temp, topk, topp)
    if key in cache:
        METRICS["cache_hits"] += 1
        METRICS["served"] += 1
        return GenerateResponse(text=cache[key], time_s=0.0, cached=True)

    # build pending request
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    pending = PendingRequest(prompt, {"max_new_tokens": max_new, "temperature": temp, "top_k": topk, "top_p": topp, "seed": req.seed}, fut)
    await request_queue.put(pending)

    try:
        result = await asyncio.wait_for(fut, timeout=30.0)  # hard timeout for a single request
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timed out")
    text = result["text"]
    cache[key] = text  # cache it
    return GenerateResponse(text=text, time_s=result["time_s"], cached=False)

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "device": DEVICE})

@app.get("/metrics")
async def metrics():
    return JSONResponse(METRICS)
