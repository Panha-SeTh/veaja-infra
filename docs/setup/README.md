# Veaja AI — Environment & Infrastructure Setup Guide

## System Prerequisites
- **Local Smoke Testing / Development:** Windows 10/11 with WSL2 (Ubuntu 22.04 LTS), Docker Desktop with WSL2 backend enabled.
- **Production Training / High-Throughput Serving:** Dedicated NVIDIA GPU host/VM, NVIDIA Container Toolkit, CUDA compute capability >= 7.0.

---

## Run Log + Results Folder Convention

Every `test_model_infer.py` execution writes two output files into a **track sub-folder** under `results/`:

```
results/
├── baseline/           # Track: Sprint 1 CPU smoke test (Qwen/Qwen2.5-0.5B)
├── khmer-candidate/    # Track: Khmer language model candidates (Sailor2, SeaLLMs, Bayon, …)
└── gpu-prod/           # Track: GPU production / benchmark runs
```

Each run produces:
| File | Description |
|------|-------------|
| `<YYYYMMDD_HHMMSS>_<model-id>.log` | Full console output (tee'd during execution) |
| `<YYYYMMDD_HHMMSS>_<model-id>.json` | Structured JSON summary with timings, model, prompt, and response |

> [!NOTE]
> `results/**/*.log` and `results/**/*.json` are gitignored — only the `.gitkeep` stubs are committed. Copy key evidence logs to `evidence/` manually when archiving sprint deliverables.

### Select a Track via `--track`

| `--track` value | Use case |
|-----------------|----------|
| `khmer-candidate` *(default)* | Testing Khmer-language model candidates |
| `baseline` | CPU baseline / sprint smoke tests |
| `gpu-prod` | GPU production benchmarks |

---

## 1. Local Baseline Container & Test Run (CPU Mode — `baseline` track)

```bash
# Build Docker image
docker build -t veaja-llm-infra:sprint1 -f infra/docker/Dockerfile .

# Execute model test run — results written to results/baseline/
docker run --rm \
  -v $(pwd)/results:/app/results \
  veaja-llm-infra:sprint1 \
  python3 test_model_infer.py \
    --model Qwen/Qwen2.5-0.5B \
    --prompt "Veaja AI infrastructure test: System initialization status is" \
    --track baseline
```

---

## 2. Khmer Model Candidate Evaluation (`khmer-candidate` track)

Run directly on the host (or in Docker) to evaluate a Khmer-language candidate:

```bash
# Example: Sailor2-1B-Chat
python3 test_model_infer.py \
  --model sail/Sailor2-1B-Chat \
  --track khmer-candidate

# Example: SeaLLMs-v3-1.5B-Chat
python3 test_model_infer.py \
  --model SeaLLMs/SeaLLMs-v3-1.5B-Chat \
  --track khmer-candidate

# Example: Bayon (attentionlab)
python3 test_model_infer.py \
  --model attentionlab/bayon-it \
  --track khmer-candidate
```

Results land in `results/khmer-candidate/`.

---

## 3. Dedicated Production GPU Setup (`gpu-prod` track)

### 3a. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

### 3b. Verify GPU Passthrough

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 3c. Run GPU Benchmark

```bash
docker run --rm --gpus all \
  -v $(pwd)/results:/app/results \
  veaja-llm-infra:sprint1-gpu \
  python3 test_model_infer.py \
    --model sail/Sailor2-1B-Chat \
    --track gpu-prod
```

Results land in `results/gpu-prod/`.

---

## Archiving Evidence

After a sprint, copy notable run artefacts to `evidence/` for Git history:

```bash
cp results/khmer-candidate/20261117_144500_sail_Sailor2-1B-Chat.log evidence/khmer_model_test_run.log
```
