FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version) and LLM inference libraries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers accelerate sentencepiece

# Copy project files
COPY . /app

# Default execution command
CMD ["python3", "test_model_infer.py"]