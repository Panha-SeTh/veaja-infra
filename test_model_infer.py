import time
import torch
from transformers import pipeline

print("=" * 60)
print("Veaja AI - Sprint 1 Infrastructure Smoke Test (CPU Baseline)")
print("=" * 60)

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()} (Running on Host CPU)")

print("\n[1/3] Loading lightweight test model (Qwen/Qwen2.5-0.5B)...")
start_load = time.time()
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B",
    device=-1,
    torch_dtype=torch.float32,
)
print(f"[✓] Model loaded in {time.time() - start_load:.2f}s")

test_prompt = "Veaja AI infrastructure test: System initialization status is"
print(f"\n[2/3] Executing inference with prompt: '{test_prompt}'")
start_infer = time.time()
output = generator(test_prompt, max_new_tokens=20, do_sample=False)
print(f"[✓] Inference finished in {time.time() - start_infer:.2f}s")

print("\n[3/3] Inference Result:")
print("-" * 60)
print(output[0]["generated_text"])
print("-" * 60)
print("\n[SUCCESS] Baseline container and model pipeline verified.")