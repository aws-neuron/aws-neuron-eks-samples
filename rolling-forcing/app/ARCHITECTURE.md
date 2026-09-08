# Rolling Forcing Video Streaming — Architecture & Data Flow

## Component Diagram (Infrastructure Layer)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                     │
│                                                                       │
│  ┌─────────┐     ┌──────────────┐     ┌──────────────┐              │
│  │   ALB   │────▶│ K8s Ingress  │────▶│  K8s Service │              │
│  │ (HTTPS) │     │ (path: /rf)  │     │  "rf-gradio" │              │
│  └─────────┘     └──────────────┘     └──────┬───────┘              │
│                                               │                       │
│                                               ▼                       │
│                                    ┌──────────────────┐              │
│                                    │  rf-gradio-app   │              │
│                                    │  (Gradio Deploy) │              │
│                                    │   m5 instance    │              │
│                                    └────────┬─────────┘              │
│                                             │ HTTP POST               │
│                                             │ /generate/stream        │
│                                             ▼                         │
│                                    ┌──────────────────┐              │
│                                    │   K8s Service    │              │
│                                    │      "rf"        │              │
│                                    └────────┬─────────┘              │
│                                             │                         │
│                                             ▼                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │            rf-deploy (Rolling Forcing Pipeline)                 │  │
│  │                  Single Trn2 Chip (lnc=2)                      │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │              4 NeuronCores (TP4 via torchrun)            │   │  │
│  │  │                                                          │   │  │
│  │  │  Core 0+1 (ND0)          Core 2+3 (ND1)                │   │  │
│  │  │  ┌──────────────┐        ┌──────────────┐              │   │  │
│  │  │  │    DiT        │        │    DiT        │              │   │  │
│  │  │  │  (3 heads)    │        │  (3 heads)    │              │   │  │
│  │  │  └──────────────┘        └──────────────┘              │   │  │
│  │  │  ┌──────────────┐        ┌──────────────┐              │   │  │
│  │  │  │    DiT        │        │    DiT        │              │   │  │
│  │  │  │  (3 heads)    │        │  (3 heads)    │              │   │  │
│  │  │  └──────────────┘        └──────────────┘              │   │  │
│  │  │                                                          │   │  │
│  │  │  T5 Encoder: Rank 2        VAE Decoder: Rank 0          │   │  │
│  │  │  (text → embeddings)       (latents → pixels)           │   │  │
│  │  └─────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Request Flow (Sequence)

```
User (Browser)
    │
    │ 1. Enter prompt + click "Generate Streaming"
    ▼
ALB (HTTPS, port 443)
    │
    │ 2. Route /rf/* path
    ▼
K8s Ingress → rf-gradio Service (port 8000)
    │
    │ 3. Gradio app receives prompt
    ▼
rf-gradio-app (Python/Gradio on m5)
    │
    │ 4. POST /generate/stream {prompt, num_frames=81}
    │    (Server-Sent Events connection opened)
    ▼
rf Service (port 8000) → rf-deploy Pod (Trn2)
    │
    │ 5. T5 encodes prompt → text embeddings [1, 512, 4096]
    │    (runs once, cached for all frames)
    │
    │ 6. Rolling Forcing autoregressive loop begins:
    │    ┌─────────────────────────────────────────────┐
    │    │  For each block of 3 frames:                 │
    │    │                                              │
    │    │  a) Generate noise for block                 │
    │    │  b) 5-step denoising (DMD distilled):       │
    │    │     DiT forward pass × 5 steps              │
    │    │  c) Finalized latent block → VAE decode     │
    │    │  d) SSE: send decoded frames to client      │
    │    │                                              │
    │    │  Loop continues with next block...           │
    │    └─────────────────────────────────────────────┘
    │
    │ 7. Each decoded frame block streamed back as SSE:
    │    data: {"frame_index": N, "frame": "<base64 PNG>", "total_frames": 81}
    ▼
rf-gradio-app
    │
    │ 8. Renders frames progressively in gallery + builds MP4
    ▼
User (Browser) — sees frames appear progressively
```

## DiT (Wan2.1-T2V-1.3B) Autoregressive Diffusion Process

```
┌───────────────────────────────────────────────────────────────────┐
│                    DiT Transformer Block (×30 layers)               │
│                                                                     │
│  Input: noisy latent tokens [B, F×H×W, 1536]                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1. SELF-ATTENTION (causal, with KV cache)                    │  │
│  │    Purpose: Temporal coherence between frames                 │  │
│  │    - Queries attend to all previous frames via KV cache       │  │
│  │    - Ensures motion consistency & temporal continuity          │  │
│  │    - Uses sliding window + anchor (first block always visible)│  │
│  │                                                               │  │
│  │    RoPE (Rotary Position Embedding):                          │  │
│  │    - Encodes 3D position (frame, height, width) into Q/K      │  │
│  │    - Enables the model to understand spatial and temporal      │  │
│  │      position of each patch token                             │  │
│  │    - Frame dim: relative temporal ordering                     │  │
│  │    - H/W dims: spatial layout within each frame               │  │
│  │                                                               │  │
│  │    ⚡ NKI Kernels: self_attention, rope                       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 2. CROSS-ATTENTION                                           │  │
│  │    Purpose: Text conditioning — injects prompt semantics      │  │
│  │    - Q = visual tokens (current noisy latents)                │  │
│  │    - K, V = T5 text embeddings [1, 512, 1536] (cached)       │  │
│  │    - Each spatial token attends to ALL text tokens            │  │
│  │    - This is HOW the model "follows the prompt"               │  │
│  │                                                               │  │
│  │    QK-Norm (RMSNorm on Q and K before attention):             │  │
│  │    - Stabilizes attention logits at scale                     │  │
│  │    - With TP4: uses TPRMSNorm (all-reduce for global RMS)    │  │
│  │                                                               │  │
│  │    ⚡ NKI Kernel: cross_attention                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 3. FFN (Feed-Forward Network)                                │  │
│  │    Purpose: Non-linear feature transformation                 │  │
│  │    - dim=1536 → ffn_dim=8960 → dim=1536                     │  │
│  │    - GELU activation                                          │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ▼                                      │
│  Output: denoised latent tokens                                    │
└───────────────────────────────────────────────────────────────────┘
```

## NKI Kernels Deployed

| Kernel | Purpose | Location |
|--------|---------|----------|
| `cross_attention` | Fused QKV attention for text conditioning | `kernels/cross_attention.py` |
| `self_attention` | Flash attention with KV cache & mask | `kernels/self_attention.py` |
| `rope` | Rotary position embedding rotation | `kernels/rope.py` |
| KV cache | Uses `tensor.copy_()` DMA (not NKI) | Built-in Neuron DMA |

## Key Concepts Summary

- **Cross-attention**: Conditions video generation on the text prompt. Each visual token queries ALL text tokens to understand "what to generate"
- **Self-attention**: Maintains temporal coherence across frames. Each token can see all previous frames (causal), ensuring smooth motion
- **RoPE**: Encodes 3D spatiotemporal position (frame index, height, width) so the model knows WHERE and WHEN each patch exists in the video
- **Rolling Forcing**: Autoregressive generation where each new block of frames is denoised while conditioning on previously generated (clean) frames via the KV cache
- **DMD Distillation**: Reduces denoising from ~50 steps to 5 steps per block, enabling near-real-time streaming
- **TP4 (Tensor Parallelism)**: DiT's 12 attention heads are split across 4 NeuronCores (3 heads each), with all-reduce after each O-projection and FFN output
