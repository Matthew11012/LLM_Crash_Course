# run locally (without docker)
export MODEL_ID="gpt2"                # or path to HF model
export ADAPTER_PATH="../../week2/outputs/lora_adapter"  # or blank if none
export DEVICE="cuda"                 # or "cpu"
export QUANTIZE="false"              # "true" to enable load_in_8bit
export MAX_BATCH_SIZE=8
export MAX_WAIT_MS=40

uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# run with docker
docker run --gpus all -p 8000:8000 fastapi-server
