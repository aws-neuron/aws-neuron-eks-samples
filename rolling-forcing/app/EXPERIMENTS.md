# Rolling Forcing Video Streaming Experiments

## Hardware
- **Platform**: AWS Trainium2 (TRN2)
- **Target**: Single chip/LNC (~24GB HBM)
- **Deployment**: Kubernetes with Gradio UI + FastAPI backend

## Model
- **Base Model**: Wan2.1-T2V-1.3B
- **Distillation**: RollingForcing DMD (Distribution Matching Distillation)
- **Checkpoint**: `rolling_forcing_dmd.pt`

---

## Baseline (Working) - April 12, 2026

| Parameter | Value |
|-----------|-------|
| Config | `rolling_forcing_dmd_small.yaml` |
| Frames | 9 |
| Blocks | 1 (num_frame_per_block) |
| Resolution | 480×832 pixels |
| Latent Size | 30×52 |
| Denoising Steps | 5 (1000→800→600→400→200) |
| Memory Used | ~23GB HBM |
| Status | ✅ Working |

---

## Experiment 1: More Frames (21 frames)

**Goal**: Generate longer videos (2.3x more frames) on single chip

| Config | Frames | Blocks | Resolution | Status | Memory | Inference Time | Notes |
|--------|--------|--------|------------|--------|--------|----------------|-------|
| f9_b1 (baseline) | 9 | 1 | 30×52 | ✅ Working | ~23GB | TBD | Baseline |
| f21_b1 | 21 | 1 | 30×52 | 🔄 Testing | TBD | TBD | |

---

## Experiment 2: More Blocks

**Goal**: Process multiple frames in parallel for faster throughput

| Config | Frames | Blocks | Resolution | Status | Memory | Inference Time | Notes |
|--------|--------|--------|------------|--------|--------|----------------|-------|
| f21_b2 | 21 | 2 | 30×52 | 📋 Planned | TBD | TBD | |
| f21_b3 | 21 | 3 | 30×52 | 📋 Planned | TBD | TBD | |

---

## Experiment 3: Higher Resolution

**Goal**: Better video quality with larger spatial dimensions (may need 2 chips)

| Config | Frames | Blocks | Resolution | Status | Memory | Inference Time | Notes |
|--------|--------|--------|------------|--------|--------|----------------|-------|
| f21_b1_hr | 21 | 1 | 60×104 | 📋 Planned | TBD | TBD | May need 2 chips |

---

## Experiment 4: Quality Tuning (More Denoising Steps)

**Goal**: Higher quality output with more denoising iterations

| Config | Frames | Steps | Status | Quality | Inference Time | Notes |
|--------|--------|-------|--------|---------|----------------|-------|
| baseline | 9 | 5 | ✅ Working | Baseline | TBD | |
| f21_b1_7s | 21 | 7 | 📋 Planned | TBD | TBD | |
| f21_b1_10s | 21 | 10 | 📋 Planned | TBD | TBD | |

---

## Key Learnings

1. **OOM Issues**: Initial deployment hit memory limits due to moviepy fork behavior and tensor operations
2. **Neuron Compilation**: NEFF cache significantly speeds up subsequent runs
3. **Logging**: torch_neuronx DEBUG logging can break SSE streaming (set to ERROR)
4. **VAE Decoding**: rearrange operations must happen on CPU before moving to Neuron device

---

## Commands

```bash
# Deploy backend
kubectl apply -f rf-deploy.yaml

# Deploy Gradio UI
kubectl apply -f rf-gradio-cm.yaml rf-gradio-deploy.yaml

# Check logs
kubectl logs -f deployment/rf
kubectl logs -f deployment/rf-gradio

# Restart deployment
kubectl rollout restart deployment rf
```
