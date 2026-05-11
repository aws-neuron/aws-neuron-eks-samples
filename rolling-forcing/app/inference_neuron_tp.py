"""Neuron TP inference entry point for Wan2.1-T2V-1.3B with embedded FastAPI server.

Runs the rolling-forcing pipeline with tensor parallelism across 4 NeuronCores.
ALL compute on Neuron — compiled via torch.compile(backend='neuron').

Compilation strategy:
  - T5:  torch.compile(backend='neuron') — full model (static shape, no state)
  - VAE: torch.compile(backend='neuron') — full model (static shape, no state)
  - DiT: Sub-module compilation (attention has dynamic KV cache control flow):
      * patch_embedding, text_embedding, time_embedding, time_projection, head — compiled
      * FFN in each block — compiled (Linear→GELU→Linear, pure)
      * Self-attention, cross-attention, RoPE — NKI kernels via wrap_nki HOP
      * Attention orchestration (cache indexing, eviction) — Python (dynamic)
      * dist.all_reduce — simple call, no chunking needed

Usage:
    torchrun --nproc_per_node=4 inference_neuron_tp.py

Architecture:
    4 NeuronCores across 2 NeuronDevices (ND0-ND1), 2 cores per device sharing HBM.
    1.3B model: dim=1536, 12 heads, 30 layers, ffn_dim=8960
    TP=4: 3 heads/rank, ~325M params/rank (~0.65GB bf16)

    Memory layout (HBM banks are shared per NeuronDevice pair):
      HBM Bank 0 (ND0): rank 0 (DiT + VAE) + rank 1 (DiT)   ≈ 5 GB
      HBM Bank 1 (ND1): rank 2 (DiT + T5)  + rank 3 (DiT)   ≈ 12 GB

    T5 on rank 2 (ND1) and VAE on rank 0 (ND0) ensures they never compete
    for the same HBM bank.
"""
import os
import sys
import time
import base64
import asyncio
import logging
from io import BytesIO
from typing import Optional, List
from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s [rank %(process)d]: %(message)s',
    stream=sys.stdout,
    force=True
)
for name in ['torch', 'transformers', 'torch_neuronx', 'torch_neuronx.python_ops',
             'torch_mlir', 'torch_mlir._mlir_libs']:
    logging.getLogger(name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ─── Configuration from environment ──────────────────────────────────────────
REPO_DIR = os.environ.get("REPO_DIR", os.getcwd())
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "gpu/RollingForcing"))

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/default_config.yaml")
MODEL_PATH = os.environ.get("MODEL_PATH", "wan_models/Wan2.1-T2V-1.3B")
VAE_PATH = os.environ.get("VAE_PATH", "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
# RollingForcing DMD distilled checkpoint — required for 5-step denoising.
# Without this, from_pretrained loads base Wan weights which need 50+ steps.
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/rolling_forcing_dmd.pt")
# Fixed frame count for static compilation shapes.
# Rolling forcing with num_frame_per_block=3 produces exactly 2 unique input shapes:
#   Shape 1: initial window (15 frames → 23400 tokens)
#   Shape 2: subsequent windows (3 frames → 4680 tokens)
# Changing this value changes the number of windows but NOT the per-window shapes,
# so compiled NEFFs remain valid for any multiple of num_frame_per_block + 2.
# 161 frames = 10.0 seconds at 16fps (42 latent frames after VAE temporal compress).
DEFAULT_NUM_FRAMES = int(os.environ.get("DEFAULT_NUM_FRAMES", "161"))
DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "16"))
TP_DEGREE = int(os.environ.get("TP_DEGREE", "4"))

# T5 encoder rank — placed on rank 2 (ND1) to separate from VAE (rank 0, ND0).
# This avoids T5 + VAE competing for the same HBM bank.
T5_RANK = int(os.environ.get("T5_RANK", "2"))
# VAE decoder rank
VAE_RANK = 0

# ─── Distributed setup ────────────────────────────────────────────────────────

def setup_distributed():
    """Initialize distributed process group for Trainium TP.

    Uses the 'neuron' backend which handles per-rank core assignment.
    torch.neuron.set_device(local_rank) pins each rank to its logical device.
    After set_device, torch.device("neuron") refers to the current rank's core.
    """
    assert "LOCAL_RANK" in os.environ, (
        "inference_neuron_tp.py must be launched via torchrun (LOCAL_RANK not set)"
    )

    dist.init_process_group(backend="neuron")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


# ─── Pipeline state ──────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    text_encoder: object = None
    tokenizer: object = None
    dit_pipeline: object = None
    vae_model: object = None
    vae_scale: object = None
    config: object = None
    latent_h: int = 60
    latent_w: int = 104
    frame_seq_length: int = 1560
    rank: int = 0
    world_size: int = 1


# After torch.neuron.set_device(local_rank), "neuron" refers to current core
NEURON_DEVICE = torch.device("neuron")


def load_pipeline(rank: int, world_size: int) -> PipelineState:
    """Load all models with TP sharding for DiT.

    Memory distribution:
      - T5 (~9.6 GB): loaded on T5_RANK (rank 2, ND1)
      - VAE (~0.66 GB): loaded on VAE_RANK (rank 0, ND0)
      - DiT/4 (~0.65 GB/rank): loaded on all ranks
    """
    state = PipelineState(rank=rank, world_size=world_size)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    # Load config
    state.config = OmegaConf.load(CONFIG_PATH)
    default_path = "configs/default_config.yaml"
    if os.path.exists(default_path):
        state.config = OmegaConf.merge(OmegaConf.load(default_path), state.config)

    # Get spatial dimensions
    if hasattr(state.config, 'image_or_video_shape'):
        state.latent_h = state.config.image_or_video_shape[3]
        state.latent_w = state.config.image_or_video_shape[4]
    else:
        state.latent_h = getattr(state.config, "spatial_h", 60)
        state.latent_w = getattr(state.config, "spatial_w", 104)

    state.frame_seq_length = (state.latent_h * state.latent_w) // 4
    if rank == 0:
        logger.info(f"Spatial: {state.latent_h}x{state.latent_w}, frame_seq_length={state.frame_seq_length}")

    # ── Load T5 on T5_RANK (rank 2, ND1 — separate HBM bank from VAE) ───────
    from wan.modules.tokenizers import HuggingfaceTokenizer

    if rank == T5_RANK:
        logger.info(f"Loading T5 encoder (rank {T5_RANK}, on Neuron with torch.compile)...")
        from wan.modules.t5 import umt5_xxl

        state.text_encoder = umt5_xxl(
            encoder_only=True, return_tokenizer=False,
            dtype=torch.bfloat16, device=torch.device('cpu')
        ).eval().requires_grad_(False)

        weights_path = os.path.join(MODEL_PATH, "models_t5_umt5-xxl-enc-bf16.pth")
        state.text_encoder.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=False)
        )
        # Move T5 to Neuron and compile
        state.text_encoder = state.text_encoder.to(NEURON_DEVICE)
        state.text_encoder = torch.compile(state.text_encoder, backend='neuron', dynamic=False)
        logger.info(f"T5 loaded on Neuron with torch.compile (rank {T5_RANK})")

    # All ranks need the tokenizer (lightweight, CPU-only)
    tokenizer_path = os.path.join(MODEL_PATH, "google/umt5-xxl/")
    state.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean='whitespace')

    # ── Load DiT with TP sharding (all ranks) ─────────────────────────
    if rank == 0:
        logger.info(f"Loading DiT 1.3B with TP={TP_DEGREE} (rank {rank})...")

    from models.tp_utils import init_tp_group
    from models.causal_inference_pipeline_tp import CausalInferencePipelineTP

    init_tp_group(tp_degree=TP_DEGREE)

    state.dit_pipeline = CausalInferencePipelineTP(
        denoising_step_list=list(getattr(state.config, "denoising_step_list", [1000, 800, 600, 400, 200])),
        num_frame_per_block=getattr(state.config, "num_frame_per_block", 3),
        context_noise=getattr(state.config, "context_noise", 0.0),
        warp_denoising_step=getattr(state.config, "warp_denoising_step", True),
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=getattr(state.config, "timestep_shift", 5.0),
        frame_seq_length=state.frame_seq_length,
        tp_degree=TP_DEGREE,
    )

    # Load RollingForcing DMD distilled weights (required for 5-step denoising).
    # from_pretrained loaded base Wan weights; this overlays the DMD-trained weights.
    # load_distilled_weights handles TP-aware sharding: it takes full checkpoint
    # weights and extracts each rank's shard (column-parallel for Q/K/V/fc1,
    # row-parallel for O/fc2, replicated for norms/embeddings).
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Loading DMD checkpoint: {CHECKPOINT_PATH} (rank {rank})")
        state.dit_pipeline.generator.load_distilled_weights(CHECKPOINT_PATH, use_ema=True)
    else:
        logger.warning(f"DMD checkpoint not found: {CHECKPOINT_PATH} — using base weights (will produce noise with 5-step schedule!)")

    # Move TP-sharded DiT to this rank's Neuron core
    state.dit_pipeline.generator.model = state.dit_pipeline.generator.model.to(NEURON_DEVICE)

    # Compile pure (stateless, static-shape) sub-modules within DiT
    dit_model = state.dit_pipeline.generator.model
    dit_model.patch_embedding = torch.compile(dit_model.patch_embedding, backend='neuron', dynamic=False)
    dit_model.text_embedding = torch.compile(dit_model.text_embedding, backend='neuron', dynamic=False)
    dit_model.time_embedding = torch.compile(dit_model.time_embedding, backend='neuron', dynamic=False)
    dit_model.time_projection = torch.compile(dit_model.time_projection, backend='neuron', dynamic=False)
    dit_model.head = torch.compile(dit_model.head, backend='neuron', dynamic=False)
    # Compile FFN in each block (pure: Linear→GELU→Linear, no state)
    for block in dit_model.blocks:
        block.ffn = torch.compile(block.ffn, backend='neuron', dynamic=False)

    if rank == 0:
        logger.info(f"DiT 1.3B TP-sharded on neuron (rank {rank}, {TP_DEGREE} ranks total)")
        logger.info(f"  Compiled sub-modules: patch_embed, text_embed, time_embed, time_proj, head, FFN×{len(dit_model.blocks)}")
        logger.info(f"  NKI kernels: self_attn, cross_attn, rope (via wrap_nki HOP)")

    # ── Load VAE on VAE_RANK (rank 0, ND0 — separate HBM bank from T5) ──────
    if rank == VAE_RANK:
        logger.info(f"Loading VAE (rank {VAE_RANK}, on Neuron with torch.compile)...")
        from wan.modules.vae import _video_vae

        state.vae_model = _video_vae(pretrained_path=VAE_PATH, z_dim=16).eval().requires_grad_(False)
        state.vae_model = state.vae_model.to(dtype=torch.bfloat16, device=NEURON_DEVICE)
        state.vae_model = torch.compile(state.vae_model, backend='neuron', dynamic=False)
        logger.info(f"VAE loaded on Neuron with torch.compile (rank {VAE_RANK})")

    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=torch.bfloat16)

    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=torch.bfloat16)

    # VAE scale must be on same device as VAE model (Neuron for VAE_RANK)
    if rank == VAE_RANK:
        state.vae_scale = [mean.to(NEURON_DEVICE), (1.0 / std).to(NEURON_DEVICE)]
    else:
        state.vae_scale = [mean, 1.0 / std]

    # Sync all ranks before starting
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        logger.info(f"All models loaded! Pipeline ready.")
        logger.info(f"  T5 on rank {T5_RANK} (ND{T5_RANK // 2})")
        logger.info(f"  VAE on rank {VAE_RANK} (ND{VAE_RANK // 2})")
        logger.info(f"  DiT TP={TP_DEGREE} on all ranks")

    return state


# ─── Inference helpers ────────────────────────────────────────────────────────

# Command codes for rank coordination
CMD_GENERATE = torch.tensor([1], dtype=torch.long)
CMD_STREAM = torch.tensor([2], dtype=torch.long)
CMD_SHUTDOWN = torch.tensor([99], dtype=torch.long)
CMD_IDLE = torch.tensor([0], dtype=torch.long)


def encode_prompt_distributed(state: PipelineState, prompt: str) -> torch.Tensor:
    """Encode text prompt using T5 on rank T5_RANK, broadcast result to all.

    Flow:
      1. Rank 0 tokenizes (CPU, fast) and broadcasts token IDs + mask
      2. Rank T5_RANK runs T5 on Neuron and broadcasts embeddings
      3. All ranks receive embeddings for DiT

    Returns prompt_embeds on NEURON_DEVICE for the calling rank.
    """
    rank = state.rank

    # Step 1: Rank 0 tokenizes and broadcasts IDs + mask to all ranks
    if rank == 0:
        ids, mask = state.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(torch.long)
        mask = mask.to(torch.long)
    else:
        # Allocate buffers for receiving (tokenizer always produces [1, 512])
        ids = torch.zeros(1, 512, dtype=torch.long)
        mask = torch.zeros(1, 512, dtype=torch.long)

    # Broadcast token IDs and mask from rank 0 to all (on Neuron device)
    ids_device = ids.to(NEURON_DEVICE)
    mask_device = mask.to(NEURON_DEVICE)
    dist.broadcast(ids_device, src=0)
    dist.broadcast(mask_device, src=0)

    # Step 2: Rank T5_RANK encodes with T5 on Neuron
    if rank == T5_RANK:
        seq_len = mask_device.gt(0).sum(dim=1).long()
        with torch.no_grad():
            prompt_embeds = state.text_encoder(ids_device, mask_device)
        # Zero-out padding
        prompt_embeds[0, seq_len[0]:] = 0.0
        prompt_embeds = prompt_embeds.to(torch.bfloat16).contiguous()
    else:
        # Allocate buffer to receive embeddings: [1, 512, 4096] for umt5-xxl
        prompt_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

    # Step 3: Broadcast embeddings from T5_RANK to all ranks
    dist.broadcast(prompt_embeds, src=T5_RANK)

    return prompt_embeds


def decode_latents(state: PipelineState, latents: torch.Tensor) -> List[np.ndarray]:
    """Decode latents through VAE on Neuron (VAE_RANK only).

    All ops on Neuron — clamp included.
    """
    # Rearrange on CPU before moving to device (avoids non-contiguous on Neuron)
    latents_bcthw = rearrange(latents, 'b t c h w -> b c t h w')
    latents_bcthw = latents_bcthw.to(torch.bfloat16).to(NEURON_DEVICE)

    with torch.no_grad():
        video = state.vae_model.decode(latents_bcthw, state.vae_scale)

    # All post-processing on Neuron, then move to CPU at end
    video = rearrange(video, 'b c t h w -> b t h w c')
    video = (video * 0.5 + 0.5).clamp(0, 1).cpu()
    video_np = (255.0 * video[0]).to(torch.uint8).numpy()

    return [video_np[i] for i in range(video_np.shape[0])]


def run_dit_inference(state: PipelineState, noise: torch.Tensor,
                      conditional_dict: dict) -> torch.Tensor:
    """Run DiT inference across all TP ranks.

    All ranks must call this simultaneously (coordinated by broadcast).
    KV cache + shared buffers stay resident — 1.3B has plenty of HBM headroom.
    """
    latents = state.dit_pipeline.inference_rolling_forcing(noise, conditional_dict)
    return latents.cpu()


# ─── Worker loop (ranks 1-7) ─────────────────────────────────────────────────

def worker_loop(state: PipelineState):
    """Non-rank-0 workers: wait for commands and participate in TP computation.

    Rank T5_RANK additionally handles T5 encoding when triggered.
    """
    rank = state.rank
    logger.info(f"[Rank {rank}] Entering worker loop (device=neuron)...")

    while True:
        # Wait for command from rank 0
        cmd = torch.zeros(1, dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(cmd, src=0)

        if cmd.item() == CMD_SHUTDOWN.item():
            logger.info(f"[Rank {rank}] Received shutdown command.")
            break
        elif cmd.item() == CMD_GENERATE.item() or cmd.item() == CMD_STREAM.item():
            # Receive metadata
            meta = torch.zeros(3, dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(meta, src=0)
            num_frames = meta[0].item()
            seed = meta[1].item()

            torch.manual_seed(seed)

            # All ranks participate in distributed T5 encoding:
            # - Receive token IDs broadcast from rank 0
            # - Rank T5_RANK runs T5 encoder
            # - Rank T5_RANK broadcasts embeddings to all
            ids_device = torch.zeros(1, 512, dtype=torch.long, device=NEURON_DEVICE)
            mask_device = torch.zeros(1, 512, dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(ids_device, src=0)
            dist.broadcast(mask_device, src=0)

            if rank == T5_RANK:
                seq_len = mask_device.gt(0).sum(dim=1).long()
                with torch.no_grad():
                    prompt_embeds = state.text_encoder(ids_device, mask_device)
                prompt_embeds[0, seq_len[0]:] = 0.0
                prompt_embeds = prompt_embeds.to(torch.bfloat16).contiguous()
            else:
                prompt_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

            dist.broadcast(prompt_embeds, src=T5_RANK)

            # Generate noise (deterministic from seed)
            noise = torch.randn(
                1, num_frames, 16, state.latent_h, state.latent_w,
                dtype=torch.bfloat16
            ).to(NEURON_DEVICE)

            conditional_dict = {"prompt_embeds": prompt_embeds}

            # Participate in TP forward pass (must match rank 0's code path)
            if cmd.item() == CMD_STREAM.item():
                # Streaming: rank 0 calls inference_rolling_forcing_streaming,
                # workers must call the same to stay in all-reduce lockstep
                for _ in state.dit_pipeline.inference_rolling_forcing_streaming(
                    noise, conditional_dict
                ):
                    pass  # Workers don't decode, just participate in DiT TP
            else:
                _ = run_dit_inference(state, noise, conditional_dict)

        elif cmd.item() == CMD_IDLE.item():
            continue

    logger.info(f"[Rank {rank}] Worker loop exited.")


# ─── Rank 0: FastAPI server ──────────────────────────────────────────────────

def run_server(state: PipelineState):
    """Rank 0: run FastAPI server that coordinates TP inference.

    Encoding flow:
      1. Rank 0 tokenizes prompt (CPU) and broadcasts token IDs to all ranks
      2. Rank T5_RANK encodes with T5 (Neuron) and broadcasts embeddings
      3. All ranks run DiT (TP, Neuron)
      4. Rank 0 decodes latents with VAE (Neuron)
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn

    app = FastAPI(title=f"Rolling Forcing Video Generation API (1.3B, TP={TP_DEGREE})")

    class GenerateRequest(BaseModel):
        prompt: str
        num_frames: Optional[int] = Field(default=None, ge=9, le=481)
        seed: Optional[int] = Field(default=None)
        fps: Optional[int] = Field(default=None, ge=1, le=60)

    class GenerateResponse(BaseModel):
        video: str
        frames: List[str]
        execution_time: float
        num_frames: int

    def broadcast_command_and_meta(num_frames: int, seed: int, stream: bool = False):
        """Broadcast command and metadata to all TP ranks."""
        cmd = (CMD_STREAM if stream else CMD_GENERATE).to(NEURON_DEVICE)
        dist.broadcast(cmd, src=0)

        meta = torch.tensor([num_frames, seed, 0], dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(meta, src=0)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_video(request: GenerateRequest):
        num_frames = request.num_frames or DEFAULT_NUM_FRAMES
        fps = request.fps or DEFAULT_FPS
        seed = request.seed or 0

        torch.manual_seed(seed)
        start_time = time.time()

        try:
            # Step 1: Broadcast command + metadata to workers
            broadcast_command_and_meta(num_frames, seed, stream=False)

            # Step 2: Tokenize and broadcast IDs (rank 0 → all)
            ids, mask = state.tokenizer([request.prompt], return_mask=True, add_special_tokens=True)
            ids_device = ids.to(torch.long).to(NEURON_DEVICE)
            mask_device = mask.to(torch.long).to(NEURON_DEVICE)
            dist.broadcast(ids_device, src=0)
            dist.broadcast(mask_device, src=0)

            # Step 3: Receive embeddings from T5_RANK
            prompt_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
            dist.broadcast(prompt_embeds, src=T5_RANK)

            # Step 4: DiT inference (all ranks in sync)
            noise = torch.randn(
                1, num_frames, 16, state.latent_h, state.latent_w,
                dtype=torch.bfloat16
            ).to(NEURON_DEVICE)

            conditional_dict = {"prompt_embeds": prompt_embeds}
            latents = run_dit_inference(state, noise, conditional_dict)

            # Step 5: VAE decode (rank 0 only)
            frames_np = decode_latents(state, latents)

            # Encode frames as base64
            frames_b64 = []
            for frame_np in frames_np:
                img = Image.fromarray(frame_np)
                buf = BytesIO()
                img.save(buf, format='PNG')
                frames_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

            # Encode video
            from torchvision.io import write_video
            import tempfile

            video_tensor = torch.from_numpy(np.stack(frames_np))
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                tmp_path = f.name
            write_video(tmp_path, video_tensor, fps=fps)
            with open(tmp_path, 'rb') as f:
                video_b64 = base64.b64encode(f.read()).decode('utf-8')
            os.remove(tmp_path)

            return GenerateResponse(
                video=video_b64,
                frames=frames_b64,
                execution_time=time.time() - start_time,
                num_frames=len(frames_np),
            )
        except Exception as e:
            logger.error(f"Generate error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate/stream")
    async def generate_video_streaming(request: GenerateRequest):
        """TRUE streaming: interleaves DiT block generation with VAE decode.

        Uses inference_rolling_forcing_streaming() which yields finalized
        latent blocks as they complete. Each block is decoded through VAE
        immediately and sent to the client as SSE frames.
        """
        num_frames = request.num_frames or DEFAULT_NUM_FRAMES
        seed = request.seed or 0

        torch.manual_seed(seed)

        async def generate_frames():
            import json
            try:
                # Step 1: Broadcast command + metadata
                broadcast_command_and_meta(num_frames, seed, stream=True)

                # Step 2: Tokenize and broadcast IDs
                ids, mask = state.tokenizer([request.prompt], return_mask=True, add_special_tokens=True)
                ids_device = ids.to(torch.long).to(NEURON_DEVICE)
                mask_device = mask.to(torch.long).to(NEURON_DEVICE)
                dist.broadcast(ids_device, src=0)
                dist.broadcast(mask_device, src=0)

                # Step 3: Receive embeddings from T5_RANK
                prompt_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
                dist.broadcast(prompt_embeds, src=T5_RANK)

                # Step 4: Prepare noise
                noise = torch.randn(
                    1, num_frames, 16, state.latent_h, state.latent_w,
                    dtype=torch.bfloat16
                ).to(NEURON_DEVICE)

                conditional_dict = {"prompt_embeds": prompt_embeds}

                # Step 5: TRUE streaming — yields finalized blocks during DiT inference
                frame_count = 0
                for start_frame, latent_block in state.dit_pipeline.inference_rolling_forcing_streaming(
                    noise, conditional_dict
                ):
                    # latent_block: [B, nfpb, C, H, W] — decode immediately
                    frames_np = decode_latents(state, latent_block.cpu())

                    for i, frame_np in enumerate(frames_np):
                        img = Image.fromarray(frame_np)
                        buf = BytesIO()
                        img.save(buf, format='PNG')
                        frame_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                        data = {
                            "frame_index": start_frame + i,
                            "frame": frame_b64,
                            "total_frames": num_frames
                        }
                        logger.info(f"[Stream] Sending frame {start_frame + i}/{num_frames}")
                        yield f"data: {json.dumps(data)}\n\n"
                        frame_count += 1
                        await asyncio.sleep(0)

                logger.info(f"[Stream] Done — sent {frame_count} frames")
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                logger.error(f"[Stream] Error: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        import json
        return StreamingResponse(generate_frames(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/readiness")
    async def readiness():
        return {"status": "ready", "model_loaded": True, "tp_degree": TP_DEGREE}

    @app.get("/")
    async def root():
        return {
            "service": f"Rolling Forcing Video Generation API (1.3B, TP={TP_DEGREE})",
            "model": "Wan2.1-T2V-1.3B",
            "tp_degree": TP_DEGREE,
            "t5_rank": T5_RANK,
            "vae_rank": VAE_RANK,
            "endpoints": ["/generate", "/generate/stream", "/health", "/readiness"],
            "default_num_frames": DEFAULT_NUM_FRAMES,
            "default_fps": DEFAULT_FPS,
        }

    logger.info("Starting uvicorn server on rank 0 (port 8000)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    rank, world_size = setup_distributed()

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"Wan2.1-T2V-1.3B with Tensor Parallelism (TP={TP_DEGREE})")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  TP degree: {TP_DEGREE}")
        logger.info(f"  T5 rank: {T5_RANK} (ND{T5_RANK // 2})")
        logger.info(f"  VAE rank: {VAE_RANK} (ND{VAE_RANK // 2})")
        logger.info(f"  Config: {CONFIG_PATH}")
        logger.info(f"  Model: {MODEL_PATH}")
        logger.info("=" * 60)

    # All ranks load DiT (TP-sharded)
    # Rank T5_RANK additionally loads T5
    # Rank VAE_RANK additionally loads VAE
    state = load_pipeline(rank, world_size)

    if rank == 0:
        # Rank 0 runs the FastAPI server
        run_server(state)
    else:
        # Ranks 1-3 enter the worker loop
        worker_loop(state)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
