"""Neuron TP inference entry point for Wan2.1-T2V-1.3B with embedded FastAPI server.

Runs the rolling-forcing pipeline with tensor parallelism across 4 NeuronCores.
ALL compute on Neuron — compiled via torch.compile(backend='neuron') HYBRID mode.

Compilation strategy:
  - T5:  torch.compile(backend='neuron') — full model (static shape, no state)
  - VAE: torch.compile(backend='neuron') — full model (static shape, no state)
  - DiT: Whole-block compilation (SDK fix preserves .contiguous() for NKI HOP):
      * patch_embedding, text_embedding, time_embedding, time_projection, head — compiled
      * Each transformer block — compiled with graph breaks at NKI boundaries
      * Linear projections, norms, FFN — fused into compiled NEFFs
      * Self-attention, cross-attention, RoPE — NKI kernels in EAGER mode (@torch.compiler.disable)
      * dist.all_reduce — handled by Neuron backend inside compiled graph

Usage:
    torchrun --nproc_per_node=4 inference_neuron_tp.py

Architecture:
    TP_DEGREE NeuronCores (1 core per rank, device mapping handled by runtime).
    1.3B model: dim=1536, 12 heads, 30 layers, ffn_dim=8960
    TP=4: 3 heads/rank, ~325M params/rank (~0.65GB bf16)

    Model placement:
      - DiT: TP-sharded across all ranks
      - T5:  loaded on T5_RANK (separate rank from VAE to distribute memory)
      - VAE: loaded on VAE_RANK
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
import torch._dynamo
torch._dynamo.config.cache_size_limit = 128
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

# T5 encoder rank — placed on a different rank than VAE to distribute memory load.
T5_RANK = int(os.environ.get("T5_RANK", "2"))
# VAE TP: how many ranks to shard the VAE decoder across (1=single rank, 2=2-way TP)
VAE_TP_DEGREE = int(os.environ.get("VAE_TP_DEGREE", "1"))
# VAE decoder ranks — first VAE_TP_DEGREE ranks (e.g. [0] or [0,1])
VAE_RANKS = list(range(VAE_TP_DEGREE))
VAE_RANK = 0  # primary VAE rank (for backward compat)

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

    # Initialize SP/TP parallel groups
    sp_degree = world_size // TP_DEGREE
    if sp_degree > 1:
        from models.parallel_state import init_parallel_groups
        init_parallel_groups(sp_degree, TP_DEGREE)

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
      - T5 (~9.6 GB): loaded on T5_RANK
      - VAE (~0.66 GB): loaded on VAE_RANK
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

    # ── Load T5 on T5_RANK (single rank for m-trn2 which has enough memory) ────
    from wan.modules.tokenizers import HuggingfaceTokenizer

    if rank == T5_RANK:
        logger.info(f"Loading T5 encoder (rank {T5_RANK}, eager)...")
        from wan.modules.t5 import umt5_xxl

        state.text_encoder = umt5_xxl(
            encoder_only=True, return_tokenizer=False,
            dtype=torch.bfloat16, device=torch.device('cpu')
        ).eval().requires_grad_(False)

        weights_path = os.path.join(MODEL_PATH, "models_t5_umt5-xxl-enc-bf16.pth")
        state.text_encoder.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=False)
        )
        state.text_encoder = state.text_encoder.to(NEURON_DEVICE)
        logger.info(f"T5 loaded on Neuron (rank {T5_RANK}, eager)")

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

    # Sub-module compilation with fullgraph=True:
    # Compile individual ops (no dynamo guards on Python ints).
    _compile = lambda m: torch.compile(m, backend='neuron', dynamic=False, fullgraph=True)

    dit_model = state.dit_pipeline.generator.model
    dit_model.patch_embedding = _compile(dit_model.patch_embedding)
    dit_model.text_embedding = _compile(dit_model.text_embedding)
    dit_model.time_embedding = _compile(dit_model.time_embedding)
    dit_model.time_projection = _compile(dit_model.time_projection)
    dit_model.head = _compile(dit_model.head)
    dit_model._sinusoidal_embedding_1d = _compile(dit_model._sinusoidal_embedding_1d)
    dit_model._unpatchify = _compile(dit_model._unpatchify)
    if hasattr(dit_model, '_expand_e_shard_neuron'):
        dit_model._expand_e_shard_neuron = _compile(dit_model._expand_e_shard_neuron)
    state.dit_pipeline.generator._convert_flow_pred_to_x0 = _compile(
        state.dit_pipeline.generator._convert_flow_pred_to_x0)
    state.dit_pipeline._add_noise = _compile(state.dit_pipeline._add_noise)

    for i, block in enumerate(dit_model.blocks):
        block.self_attn.q = _compile(block.self_attn.q)
        block.self_attn.k = _compile(block.self_attn.k)
        block.self_attn.v = _compile(block.self_attn.v)
        block.self_attn.o.local_linear = _compile(block.self_attn.o.local_linear)
        block.cross_attn.q = _compile(block.cross_attn.q)
        block.cross_attn.k = _compile(block.cross_attn.k)
        block.cross_attn.v = _compile(block.cross_attn.v)
        block.cross_attn.o.local_linear = _compile(block.cross_attn.o.local_linear)
        block.norm1 = _compile(block.norm1)
        block.norm2 = _compile(block.norm2)
        block.norm3 = _compile(block.norm3)
        block.ffn[0] = _compile(block.ffn[0])
        block.ffn[1] = _compile(block.ffn[1])
        block._modulated_norm_scale = _compile(block._modulated_norm_scale)
        block._modulated_norm_shift = _compile(block._modulated_norm_shift)
        block._modulated_residual = _compile(block._modulated_residual)

    if rank == 0:
        logger.info(f"DiT 1.3B TP-sharded on neuron (rank {rank}, {TP_DEGREE} ranks total)")
        logger.info(f"  Sub-module compilation (fullgraph=True): Q/K/V/O + FFN per block")
        logger.info(f"  NKI kernels + norms: eager")

    # ── Load VAE (width-sharded across ALL ranks) ──────────────────────────────
    from models.vae_wshard import build_vae, init_vae_parallel_group
    init_vae_parallel_group()
    logger.info(f"Loading width-sharded VAE (rank {rank})...")
    state.vae_model = build_vae(dtype=torch.bfloat16)
    logger.info(f"VAE width-sharded on Neuron (rank {rank})")

    # Sync all ranks before starting
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        logger.info(f"All models loaded! Pipeline ready.")
        logger.info(f"  T5 on rank {T5_RANK}")
        logger.info(f"  VAE on rank {VAE_RANK}")
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
        prompt_embeds[0, seq_len[0]:] = 0.0
        prompt_embeds = prompt_embeds.to(torch.bfloat16).contiguous()
    else:
        prompt_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

    # Step 3: Broadcast embeddings from T5_RANK to all ranks
    dist.broadcast(prompt_embeds, src=T5_RANK)

    return prompt_embeds


def w_shard(tensor, rank, world):
    W = tensor.shape[-1]
    assert W % world == 0, f"W={W} not divisible by world={world}"
    s = W // world
    return tensor[..., rank * s:(rank + 1) * s].contiguous()


def decode_latents_wshard(state: PipelineState, latent_block: torch.Tensor,
                          chunk_idx: int) -> List[np.ndarray]:
    """Decode latents through width-sharded VAE on all ranks with streaming cache."""
    rank = state.rank
    world = state.world_size

    latent_device = latent_block.to(torch.bfloat16).to(NEURON_DEVICE)
    chunk_latent = w_shard(latent_device, rank, world)

    with torch.no_grad():
        chunk_device = state.vae_model.decode_to_pixel_device(
            chunk_latent, use_cache=True, chunk_idx=chunk_idx)
        chunk_video = state.vae_model.postprocess_pixels(chunk_device)

    # chunk_video is [1, T, C, H, W_local] on CPU
    # Gather all width shards on rank 0 via file exchange
    import tempfile
    scratch_dir = os.path.join(tempfile.gettempdir(), "vae_shards")
    if rank == 0:
        os.makedirs(scratch_dir, exist_ok=True)
    dist.barrier()
    torch.save(chunk_video, os.path.join(scratch_dir, f"shard_rank{rank}.pt"))
    dist.barrier()

    frames = []
    if rank == 0:
        shards = [
            torch.load(os.path.join(scratch_dir, f"shard_rank{r}.pt"), map_location="cpu")
            for r in range(world)
        ]
        video = torch.cat(shards, dim=-1)  # [1, T, C, H, W_full]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = rearrange(video[0], 't c h w -> t h w c')
        video_np = (255.0 * video).to(torch.uint8).numpy()
        frames = [video_np[i] for i in range(video_np.shape[0])]

    dist.barrier()
    if rank == 0:
        for r in range(world):
            os.remove(os.path.join(scratch_dir, f"shard_rank{r}.pt"))
    return frames


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
                # Streaming: all ranks participate in DiT + width-sharded VAE
                state.vae_model.model.clear_cache()
                block_idx = 0
                for start_frame, latent_block in state.dit_pipeline.inference_rolling_forcing_streaming(
                    noise, conditional_dict
                ):
                    _ = decode_latents_wshard(state, latent_block, block_idx)
                    block_idx += 1
            else:
                _ = run_dit_inference(state, noise, conditional_dict)
                # VAE TP workers participate in non-streaming decode too
                if VAE_TP_DEGREE > 1 and rank in VAE_RANKS and state.vae_model is not None:
                    # Rank 0 will broadcast latents to VAE TP workers — but in the
                    # current flow rank 0 calls decode_latents directly. The all-reduce
                    # inside RowParallelCausalConv3d requires all VAE ranks to call decode.
                    # For non-streaming, rank 0 calls decode after run_dit_inference,
                    # so workers need a matching decode call. We receive latents via broadcast.
                    pass  # TODO: need latent broadcast for non-streaming VAE TP

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


# ─── Benchmark mode (no server) ──────────────────────────────────────────────

def _run_single_generation(state: PipelineState, prompt: str, num_frames: int, seed: int):
    """Run a single streaming generation on ALL ranks symmetrically.

    All ranks execute the same code path. Only rank 0 collects frames/timing.
    """
    rank = state.rank
    torch.manual_seed(seed)
    state.vae_model.model.clear_cache()

    # T5 encoding: rank 0 tokenizes and broadcasts, T5_RANK encodes and broadcasts
    if rank == 0:
        ids, mask = state.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids_device = ids.to(torch.long).to(NEURON_DEVICE)
        mask_device = mask.to(torch.long).to(NEURON_DEVICE)
    else:
        ids_device = torch.zeros(1, 512, dtype=torch.long, device=NEURON_DEVICE)
        mask_device = torch.zeros(1, 512, dtype=torch.long, device=NEURON_DEVICE)
    dist.broadcast(ids_device, src=0)
    dist.broadcast(mask_device, src=0)

    t5_start = time.time()
    if rank == T5_RANK:
        seq_len = mask_device.gt(0).sum(dim=1).long()
        with torch.no_grad():
            prompt_embeds = state.text_encoder(ids_device, mask_device)
        prompt_embeds[0, seq_len[0]:] = 0.0
        prompt_embeds = prompt_embeds.to(torch.bfloat16).contiguous()
    else:
        prompt_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
    dist.broadcast(prompt_embeds, src=T5_RANK)
    t5_time = time.time() - t5_start
    if rank == 0:
        logger.info(f"  T5:          {t5_time*1000:7.1f} ms")

    # Prepare noise (same seed on all ranks = same noise)
    noise = torch.randn(
        1, num_frames, 16, state.latent_h, state.latent_w,
        dtype=torch.bfloat16
    ).to(NEURON_DEVICE)
    conditional_dict = {"prompt_embeds": prompt_embeds}

    # Streaming inference with per-block timing
    per_block = []
    all_frame_arrays = []
    total_pixel_frames = 0
    block_idx = 0
    gen_start = time.time()
    last_yield_time = time.time()

    for start_frame, latent_block in state.dit_pipeline.inference_rolling_forcing_streaming(
        noise, conditional_dict
    ):
        dit_time = time.time() - last_yield_time

        # VAE decode (width-sharded across all ranks)
        vae_start = time.time()
        frames_np = decode_latents_wshard(state, latent_block, block_idx)
        vae_time = time.time() - vae_start

        block_e2e = dit_time + vae_time
        n_frames = len(frames_np)
        total_pixel_frames += n_frames
        all_frame_arrays.extend(frames_np)

        per_block.append({
            "block": block_idx,
            "dit_ms": dit_time * 1000,
            "vae_ms": vae_time * 1000,
            "block_total_ms": block_e2e * 1000,
            "n_frames": n_frames,
            "wall_s": time.time() - gen_start,
        })
        if rank == 0:
            block_fps = n_frames / block_e2e if block_e2e > 0 else 0
            logger.info(f"  block {block_idx:2d}: DiT {dit_time*1000:7.1f} ms  VAE {vae_time*1000:6.1f} ms  {n_frames:2d} frames  {block_fps:5.2f} fps")
        block_idx += 1
        last_yield_time = time.time()

    gen_time = time.time() - gen_start
    return per_block, all_frame_arrays, t5_time, gen_time, total_pixel_frames


def run_benchmark(state: PipelineState):
    """ALL ranks: warmup (compile), then run benchmark, report FPS (rank 0 only prints)."""
    import json
    from datetime import datetime

    rank = state.rank
    num_frames = DEFAULT_NUM_FRAMES
    fps = DEFAULT_FPS
    num_benchmark_runs = int(os.environ.get("BENCHMARK_RUNS", "3"))

    warmup_prompt = "A cat walking on the beach at sunset, cinematic lighting, 4k"
    benchmark_prompt = (
        "A dynamic action shot in the style of a professional skateboard magazine, "
        "featuring a young male longboarder accelerating downhill. He is fully focused, "
        "his expression intense and determined, carving through tight turns with precision. "
        "His longboard glides smoothly over the pavement, creating a blur of motion. "
        "He wears a black longboard shirt, blue jeans, and white sneakers, with a backpack "
        "slung over one shoulder. His hair flows behind him as he moves, and he grips the "
        "board tightly with both hands. The background shows a scenic urban street with "
        "blurred buildings and trees, hinting at a lively cityscape. The photo captures "
        "the moment just after he exits a turn, with a slight bounce in the board and a "
        "sense of speed and agility. A medium shot with a slightly elevated camera angle."
    )

    # Create timestamped run directory (rank 0 only)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.environ.get("OUTPUT_DIR", f"/tmp/rf_run_{timestamp}")
    frames_dir = os.path.join(run_dir, "frames")
    if rank == 0:
        os.makedirs(frames_dir, exist_ok=True)
        logger.info(f"Run output directory: {run_dir}")
        logger.info("=" * 60)
        logger.info("  ROLLING FORCING BENCHMARK")
        logger.info(f"  Model: Wan2.1-T2V-1.3B | TP={TP_DEGREE} | Device: Trainium2")
        logger.info(f"  Frames: {num_frames} | Benchmark runs: {num_benchmark_runs}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("=" * 60)
        logger.info("  PHASE 1: WARMUP (triggers compilation)")
        logger.info(f"  Prompt: {warmup_prompt[:60]}...")
        logger.info("=" * 60)

    warmup_start = time.time()
    warmup_blocks, _, _, warmup_gen_time, warmup_frames = _run_single_generation(
        state, warmup_prompt, num_frames, seed=42
    )
    compilation_time = time.time() - warmup_start
    if rank == 0:
        logger.info(f"  Warmup complete: {compilation_time:.1f}s ({warmup_frames} frames)")
        logger.info(f"  Block 0 (compilation): {warmup_blocks[0]['block_total_ms']/1000:.1f}s")
        if len(warmup_blocks) > 1:
            warmup_steady = sum(b['block_total_ms'] for b in warmup_blocks[1:]) / (len(warmup_blocks)-1) / 1000
            logger.info(f"  Blocks 1-{len(warmup_blocks)-1} avg: {warmup_steady:.3f}s/block")

    # ── Phase 2: BENCHMARK (post-compilation measurement) ─────────────────────
    if rank == 0:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  PHASE 2: BENCHMARK (post-compilation, no compile overhead)")
        logger.info(f"  Prompt: {benchmark_prompt[:60]}...")
        logger.info(f"  Runs: {num_benchmark_runs}")
        logger.info("=" * 60)

    all_runs = []
    all_frame_arrays = []  # frames from last run for video output

    for run_idx in range(num_benchmark_runs):
        run_seed = 100 + run_idx
        if rank == 0:
            logger.info(f"  Run {run_idx+1}/{num_benchmark_runs} (seed={run_seed})...")

        per_block, frame_arrays, t5_time, gen_time, pixel_frames = _run_single_generation(
            state, benchmark_prompt, num_frames, seed=run_seed
        )

        if rank == 0:
            run_fps = pixel_frames / gen_time if gen_time > 0 else 0
            avg_block_ms = sum(b['block_total_ms'] for b in per_block) / len(per_block)
            avg_dit_ms = sum(b['dit_ms'] for b in per_block) / len(per_block)
            avg_vae_ms = sum(b['vae_ms'] for b in per_block) / len(per_block)

            logger.info(f"    → {pixel_frames} frames in {gen_time:.2f}s = {run_fps:.2f} FPS")
            logger.info(f"    → Avg block: {avg_block_ms:.0f}ms (DiT:{avg_dit_ms:.0f}ms + VAE:{avg_vae_ms:.0f}ms)")

            all_runs.append({
                "run": run_idx,
                "seed": run_seed,
                "num_frames": pixel_frames,
                "gen_time_s": gen_time,
                "fps": run_fps,
                "t5_time_s": t5_time,
                "per_block": per_block,
            })

            if run_idx == num_benchmark_runs - 1:
                all_frame_arrays = frame_arrays

    # Workers are done — only rank 0 aggregates and prints results
    if rank != 0:
        return

    # ── Aggregate results across all benchmark runs ───────────────────────────
    total_benchmark_frames = sum(r["num_frames"] for r in all_runs)
    total_benchmark_time = sum(r["gen_time_s"] for r in all_runs)
    overall_fps = total_benchmark_frames / total_benchmark_time if total_benchmark_time > 0 else 0

    all_blocks = [b for r in all_runs for b in r["per_block"]]
    num_frame_per_block = getattr(state.config, "num_frame_per_block", 3)
    avg_dit_per_block = sum(b["dit_ms"] for b in all_blocks) / len(all_blocks) / 1000
    avg_vae_per_block = sum(b["vae_ms"] for b in all_blocks) / len(all_blocks) / 1000
    avg_e2e_per_block = sum(b["block_total_ms"] for b in all_blocks) / len(all_blocks) / 1000
    stream_fps = num_frame_per_block / avg_e2e_per_block if avg_e2e_per_block > 0 else 0
    vae_fps = num_frame_per_block / avg_vae_per_block if avg_vae_per_block > 0 else 0
    realtime_ratio = stream_fps / fps if fps > 0 else 0

    per_run_fps = [r["fps"] for r in all_runs]
    avg_run_fps = sum(per_run_fps) / len(per_run_fps)
    min_run_fps = min(per_run_fps)
    max_run_fps = max(per_run_fps)

    # Print results
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK RESULTS (post-compilation streaming)              │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  Compilation warmup:    {compilation_time:>8.1f}s (cat prompt, excluded) │")
    print(f"│  Benchmark runs:        {num_benchmark_runs:>8}                        │")
    print(f"│  Total frames measured: {total_benchmark_frames:>8}                        │")
    print(f"│  Total benchmark time:  {total_benchmark_time:>8.2f}s                      │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  DiT/block avg:         {avg_dit_per_block:>8.3f}s (5 steps)         │")
    print(f"│  VAE/block avg:         {avg_vae_per_block:>8.3f}s ({num_frame_per_block} frames)    │")
    print(f"│  E2E/block avg:         {avg_e2e_per_block:>8.3f}s (DiT+VAE)         │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  STREAMING FPS:         {stream_fps:>8.2f} frames/sec           │")
    print(f"│  VAE decode FPS:        {vae_fps:>8.2f} frames/sec           │")
    print(f"│  Real-time ratio:       {realtime_ratio:>8.3f}x (vs {fps}fps)        │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  Per-run FPS:  avg={avg_run_fps:.2f}  min={min_run_fps:.2f}  max={max_run_fps:.2f}  │")
    for i, r in enumerate(all_runs):
        print(f"│    Run {i+1}: {r['fps']:.2f} fps ({r['num_frames']} frames / {r['gen_time_s']:.1f}s)      │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()

    if stream_fps >= fps:
        print(f"  ✅ Streaming FPS ({stream_fps:.1f}) >= playback FPS ({fps}) — REAL-TIME CAPABLE!")
    else:
        speedup_needed = fps / stream_fps if stream_fps > 0 else float('inf')
        print(f"  ⚠️  Need {speedup_needed:.1f}x speedup to reach real-time ({fps}fps playback)")

    # Build results JSON
    results = {
        "benchmark_type": "post_compilation_streaming",
        "compilation_time_s": compilation_time,
        "num_benchmark_runs": num_benchmark_runs,
        "num_pixel_frames_total": total_benchmark_frames,
        "total_benchmark_time_s": total_benchmark_time,
        "stream_fps": stream_fps,
        "vae_fps": vae_fps,
        "avg_dit_per_block_s": avg_dit_per_block,
        "avg_vae_per_block_s": avg_vae_per_block,
        "avg_e2e_per_block_s": avg_e2e_per_block,
        "realtime_ratio": realtime_ratio,
        "per_run_fps": per_run_fps,
        "avg_run_fps": avg_run_fps,
        "playback_fps": fps,
        "config": {
            "tp_degree": TP_DEGREE,
            "num_frame_per_block": num_frame_per_block,
            "denoising_steps": getattr(state.dit_pipeline, 'denoising_steps', 5),
            "latent_spatial": f"{state.latent_h}x{state.latent_w}",
            "warmup_prompt": warmup_prompt,
            "benchmark_prompt": benchmark_prompt[:80] + "...",
        },
        "runs": all_runs,
    }

    # Save frames from last benchmark run
    try:
        logger.info(f"Saving {len(all_frame_arrays)} frames as PNGs to {frames_dir}/ ...")
        for i, frame in enumerate(all_frame_arrays):
            img = Image.fromarray(frame)
            img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
        logger.info(f"Frames saved: {frames_dir}/frame_0000.png ... frame_{len(all_frame_arrays)-1:04d}.png")
    except Exception as e:
        logger.warning(f"Failed to save frames: {e}")

    # Save benchmark results JSON to run directory
    results["run_dir"] = run_dir
    results_path = os.path.join(run_dir, "benchmark.json")
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved: {results_path}")
    except Exception as e:
        logger.warning(f"Failed to save benchmark.json: {e}")

    print()
    print(json.dumps(results, indent=2))
    print()
    logger.info(f"Benchmark complete! All outputs in: {run_dir}")

    # Signal workers to exit
    cmd = CMD_SHUTDOWN.to(NEURON_DEVICE)
    dist.broadcast(cmd, src=0)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode: generate video directly, measure FPS, exit")
    args, _ = parser.parse_known_args()

    rank, world_size = setup_distributed()

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"Wan2.1-T2V-1.3B with Tensor Parallelism (TP={TP_DEGREE})")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  TP degree: {TP_DEGREE}")
        logger.info(f"  T5 rank: {T5_RANK}")
        logger.info(f"  VAE rank: {VAE_RANK}")
        logger.info(f"  Config: {CONFIG_PATH}")
        logger.info(f"  Model: {MODEL_PATH}")
        logger.info(f"  Mode: {'BENCHMARK' if args.benchmark else 'SERVER'}")
        logger.info("=" * 60)

    # All ranks load DiT (TP-sharded)
    # Rank T5_RANK additionally loads T5
    # Rank VAE_RANK additionally loads VAE
    state = load_pipeline(rank, world_size)

    # ── Warmup: run a short generation to trigger compilation ──
    WARMUP_FRAMES = int(os.environ.get("WARMUP_FRAMES", "0"))
    if WARMUP_FRAMES > 0:
        if rank == 0:
            logger.info("=" * 60)
            logger.info(f"  WARMUP: Generating {WARMUP_FRAMES}-frame video (triggers compilation)")
            logger.info("=" * 60)

        warmup_prompt = "A cat walking on a sunny beach"
        warmup_seed = 42
        warmup_num_frames = WARMUP_FRAMES

        # All ranks: broadcast command (same protocol as server/worker)
        cmd = CMD_GENERATE.to(NEURON_DEVICE)
        dist.broadcast(cmd, src=0)
        meta = torch.tensor([warmup_num_frames, warmup_seed, 0], dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(meta, src=0)

        # Tokenize and broadcast IDs
        ids, mask_tok = state.tokenizer([warmup_prompt], return_mask=True, add_special_tokens=True)
        ids_device = ids.to(torch.long).to(NEURON_DEVICE)
        mask_tok_device = mask_tok.to(torch.long).to(NEURON_DEVICE)
        dist.broadcast(ids_device, src=0)
        dist.broadcast(mask_tok_device, src=0)

        # T5 encode (T5_RANK encodes, broadcasts to all)
        prompt_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        if rank == T5_RANK:
            seq_len = mask_tok_device.gt(0).sum(dim=1).long()
            with torch.no_grad():
                prompt_embeds = state.text_encoder(ids_device, mask_tok_device)
            prompt_embeds[0, seq_len[0]:] = 0.0
            prompt_embeds = prompt_embeds.to(torch.bfloat16).contiguous()
        dist.broadcast(prompt_embeds, src=T5_RANK)

        # DiT inference (all ranks participate via TP)
        noise = torch.randn(
            1, warmup_num_frames, 16, state.latent_h, state.latent_w,
            dtype=torch.bfloat16
        ).to(NEURON_DEVICE)
        conditional_dict = {"prompt_embeds": prompt_embeds}
        latents = run_dit_inference(state, noise, conditional_dict)

        # VAE decode (all ranks, width-sharded)
        state.vae_model.model.clear_cache()
        decode_latents_wshard(state, latents, 0)

        dist.barrier()
        if rank == 0:
            logger.info("=" * 60)
            logger.info("  WARMUP COMPLETE — all kernels compiled")
            logger.info("=" * 60)

    if args.benchmark:
        run_benchmark(state)
    elif rank == 0:
        run_server(state)
    else:
        worker_loop(state)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
