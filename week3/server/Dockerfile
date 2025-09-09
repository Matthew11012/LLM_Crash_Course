# Start with small Python image
FROM python:3.10-slim

# Install only what we need
RUN pip install --no-cache-dir fastapi uvicorn[standard] && pip install torch --index-url https://download.pytorch.org/whl/cu129
RUN pip install transformers peft bitsandbytes cachetools pydantic

# Copy your server code
WORKDIR /app
COPY . /app/

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
