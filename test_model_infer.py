import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch
from transformers import pipeline

# Ensure UTF-8 output encoding for Khmer characters across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Veaja AI - Model Candidate Smoke Test")
parser.add_argument(
    "--model",
    type=str,
    default="sail/Sailor2-1B-Chat",
    help="Hugging Face model ID (e.g. sail/Sailor2-1B-Chat, SeaLLMs/SeaLLMs-v3-1.5B-Chat, attentionlab/bayon-it)",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="សួស្តី Veaja AI! តើអ្នកអាចជួយអ្វីខ្ញុំបានខ្លះ?",
    help="Test prompt (Khmer or English)",
)
parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
parser.add_argument(
    "--track",
    type=str,
    default="khmer-candidate",
    choices=["khmer-candidate", "baseline", "gpu-prod"],
    help="Run track — determines results sub-folder (khmer-candidate | baseline | gpu-prod)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Results folder convention:  results/<track>/
# Log file name:              <YYYYMMDD_HHMMSS>_<sanitised-model-id>.log
# JSON summary file:          <YYYYMMDD_HHMMSS>_<sanitised-model-id>.json
# ---------------------------------------------------------------------------
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
SAFE_MODEL = args.model.replace("/", "_").replace(":", "-")
RESULTS_DIR = os.path.join("results", args.track)
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH  = os.path.join(RESULTS_DIR, f"{RUN_TS}_{SAFE_MODEL}.log")
JSON_PATH = os.path.join(RESULTS_DIR, f"{RUN_TS}_{SAFE_MODEL}.json")

# Tee stdout to both console and log file
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

log_fh = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, log_fh)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
print("=" * 65)
print("Veaja AI - Model Candidate Inference Test")
print("=" * 65)
print(f"Track            : {args.track}")
print(f"Target Candidate : {args.model}")
print(f"Run Timestamp    : {RUN_TS} UTC")
print(f"Log File         : {LOG_PATH}")
print(f"JSON Summary     : {JSON_PATH}")
print(f"PyTorch Version  : {torch.__version__}")
print(f"CUDA Available   : {torch.cuda.is_available()}")

device = 0 if torch.cuda.is_available() else -1
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------------------------------
# Step 1 — Load model
# ---------------------------------------------------------------------------
print(f"\n[1/3] Loading model candidate '{args.model}'...")
start_load = time.time()
generator = pipeline(
    "text-generation",
    model=args.model,
    device=device,
    dtype=dtype,
)
load_time = time.time() - start_load
print(f"[✓] Model loaded in {load_time:.2f}s")

# ---------------------------------------------------------------------------
# Step 2 — Inference
# ---------------------------------------------------------------------------
messages = [
    {"role": "system", "content": "You are Veaja AI, a helpful and respectful Cambodian AI assistant who communicates fluently in Khmer."},
    {"role": "user",   "content": args.prompt},
]

print("\n[2/3] Executing inference with prompt:")
print(f"Prompt: {args.prompt}")
start_infer = time.time()

try:
    output = generator(
        messages,
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    result = output[0]["generated_text"]
    generated_response = result[-1]["content"] if isinstance(result, list) else result
except Exception:
    # Fallback to direct prompt if chat template fails (e.g., base models)
    output = generator(
        args.prompt,
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    generated_response = output[0]["generated_text"]

infer_time = time.time() - start_infer
print(f"[✓] Inference finished in {infer_time:.2f}s")

# ---------------------------------------------------------------------------
# Step 3 — Print result
# ---------------------------------------------------------------------------
print("\n[3/3] Inference Result:")
print("-" * 65)
print(generated_response)
print("-" * 65)
print("\n[SUCCESS] Model candidate verified successfully.")
print(f"[OUTPUT] Results saved to: {RESULTS_DIR}/")

# ---------------------------------------------------------------------------
# Flush log before closing
# ---------------------------------------------------------------------------
sys.stdout.flush()
log_fh.flush()
log_fh.close()
sys.stdout = sys.__stdout__

# ---------------------------------------------------------------------------
# Write JSON summary
# ---------------------------------------------------------------------------
summary = {
    "run_timestamp_utc": RUN_TS,
    "track": args.track,
    "model": args.model,
    "prompt": args.prompt,
    "max_tokens": args.max_tokens,
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "load_time_s": round(load_time, 2),
    "infer_time_s": round(infer_time, 2),
    "generated_response": generated_response,
    "status": "SUCCESS",
    "log_file": LOG_PATH,
}
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"[JSON] Summary written → {JSON_PATH}")
