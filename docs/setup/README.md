# Veaja AI — Environment & Infrastructure Setup Guide

## System Prerequisites
- **Local Smoke Testing / Development:** Windows 10/11 with WSL2 (Ubuntu 22.04 LTS), Docker Desktop with WSL2 backend enabled.
- **Production Training / High-Throughput Serving:** Dedicated NVIDIA GPU host/VM, NVIDIA Container Toolkit, CUDA compute capability >= 7.0.

---

## 1. Local Baseline Container & Test Run (CPU Mode)
Build and run the verified baseline image:

```bash
# Build Docker image
docker build -t veaja-llm-infra:sprint1 -f infra/docker/Dockerfile .

# Execute model test run and record logs
mkdir -p evidence
docker run --rm -v $(pwd)/evidence:/app/evidence veaja-llm-infra:sprint1 python3 test_model_infer.py | tee evidence/model_test_run.log
```

---

## 2. Dedicated Production GPU Setup (Target Server / Cloud VM)
When running on the team's GPU server or cloud instance:

1. **Install NVIDIA Container Toolkit:**
   ```bash
   curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

2. **Verify GPU Passthrough:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Run GPU Container:**
   ```bash
   docker run --rm --gpus all veaja-llm-infra:sprint1-gpu
   ```