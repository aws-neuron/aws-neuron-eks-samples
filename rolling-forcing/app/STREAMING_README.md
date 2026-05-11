# 🎬 Streaming Video Generation

This module adds streaming capabilities to the Rolling Forcing video generation pipeline, allowing users to see generated frames progressively instead of waiting for the full video to complete.

## Overview

The streaming implementation leverages the autoregressive nature of the Rolling Forcing algorithm. Since frames are generated in blocks and pass through a denoising window, we can output finalized frames as soon as they complete their denoising steps.

### Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌─────────────┐
│  T5 Encoder │ ──▶ │ DiT (Rolling     │ ──▶ │ Streaming   │ ──▶ │   Gradio    │
│  (prompt)   │     │ Forcing)         │     │ VAE Decode  │     │   WebUI     │
└─────────────┘     └──────────────────┘     └─────────────┘     └─────────────┘
                           │                       │
                           ▼                       ▼
                    Yields latent            Yields decoded
                    blocks as they           frames/images
                    are finalized
```

## Components

### 1. `streaming_pipeline.py`

The core streaming inference pipeline that wraps `CausalInferencePipeline`.

```python
from streaming_pipeline import StreamingInferencePipeline, StreamingConfig

config = StreamingConfig(
    config_path="configs/rolling_forcing_dmd_small.yaml",
    checkpoint_path="checkpoints/rolling_forcing_dmd.pt",
    use_ema=True,
    device="neuron",
)

pipe = StreamingInferencePipeline(config)

# Frame-by-frame streaming
for frame_idx, frame in pipe.generate_streaming(prompt="A cat walking"):
    display(frame)  # PIL Image appears progressively

# Chunk-based streaming (better quality)
for chunk_path in pipe.generate_chunked(prompt="A cat walking", chunk_size=6):
    play_video(chunk_path)  # Video segments
```

### 2. `streaming_vae.py`

Optimized VAE decoder for incremental decoding.

```python
from streaming_vae import create_decoder

decoder = create_decoder(
    vae_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    device="neuron",
)

# Decode blocks as they arrive
for latent_block in dit_generator:
    frames = decoder.decode_block(latent_block)
    yield frames
```

### 3. `streaming_app.py`

Gradio web interface with three modes:

- **🚀 Streaming Mode**: See frames as they're generated (lower latency)
- **🎥 Quality Mode**: Get properly encoded video (better quality)
- **⚖️ Comparison Mode**: Side-by-side comparison of both modes

## Quick Start

### 1. Test with Mock Pipeline (No GPU Required)

```bash
cd video-streaming-develop
python streaming_app.py --mock --port 7860
```

Open http://localhost:7860 in your browser.

### 2. Run with Real Model on Neuron

```bash
python streaming_app.py \
    --config configs/rolling_forcing_dmd_small.yaml \
    --checkpoint checkpoints/rolling_forcing_dmd.pt \
    --use_ema \
    --device neuron \
    --port 7860
```

### 3. Create Public Share Link

```bash
python streaming_app.py \
    --config configs/rolling_forcing_dmd_small.yaml \
    --checkpoint checkpoints/rolling_forcing_dmd.pt \
    --use_ema \
    --share
```

## Streaming Modes Explained

### Streaming Mode (Frame-by-Frame)

**How it works:**
1. DiT generates frames in blocks (e.g., 3 frames per block)
2. After each block passes through all denoising steps, it's "finalized"
3. Finalized latents are immediately decoded by VAE
4. Decoded frames are sent to the browser via Gradio

**Pros:**
- See first frame in ~1-2 minutes (vs ~10 min for full video)
- Interactive feedback during generation
- Can cancel early if output looks wrong

**Cons:**
- Each frame decoded separately (less efficient)
- No inter-frame video compression

### Quality Mode (Chunk-Based)

**How it works:**
1. Frames are collected into chunks (e.g., 6 frames)
2. Each chunk is encoded as a video segment (H.264)
3. Segments are streamed to the player

**Pros:**
- Proper video compression
- Smoother playback
- Smaller file size

**Cons:**
- Higher latency to first visual
- Requires more frames before output

## API Usage

### Programmatic Streaming

```python
from streaming_pipeline import StreamingInferencePipeline, StreamingConfig

config = StreamingConfig(
    config_path="configs/rolling_forcing_dmd_small.yaml",
    checkpoint_path="checkpoints/rolling_forcing_dmd.pt",
    device="neuron",
)

pipe = StreamingInferencePipeline(config)

# Simple streaming
frames = []
for idx, frame in pipe.generate_streaming("A rocket launching"):
    frames.append(frame)
    print(f"Got frame {idx}")

# Save as video
import imageio
imageio.mimwrite("output.mp4", [np.array(f) for f in frames], fps=16)
```

### True Streaming (Yield During DiT)

For maximum streaming benefit, use `generate_streaming_true()` which yields frames during the DiT inference loop itself:

```python
for frame_idx, frame, latent in pipe.generate_streaming_true(prompt):
    # frame is available as soon as the DiT finalizes it
    # No need to wait for full inference to complete
    yield frame
```

## Configuration Options

### StreamingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config_path` | Required | Path to model config YAML |
| `checkpoint_path` | None | Path to checkpoint file |
| `model_path` | `wan_models/Wan2.1-T2V-1.3B` | Base model directory |
| `vae_path` | `wan_models/.../Wan2.1_VAE.pth` | VAE weights path |
| `num_frames` | 21 | Default frames to generate |
| `use_ema` | True | Use EMA weights from checkpoint |
| `seed` | 0 | Random seed |
| `fps` | 16 | Frames per second |
| `device` | `neuron` | Device: `neuron`, `cuda`, or `cpu` |

### VAEConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decode_batch_size` | 4 | Frames per decode call |
| `use_tiled_decode` | False | Memory-efficient tiled decoding |
| `tile_size` | 256 | Tile size for tiled decode |
| `tile_overlap` | 32 | Overlap between tiles |

## Performance Considerations

### Latency Breakdown (21 frames, ~10 min total)

| Stage | Time | Streaming Benefit |
|-------|------|-------------------|
| T5 Encoding | ~30s | One-time cost |
| DiT Inference | ~8 min | Frames finalized progressively |
| VAE Decode | ~1.5 min | Decode as blocks arrive |
| **First Frame** | **~1-2 min** | vs 10 min without streaming |

### Memory Usage

- Streaming doesn't increase peak memory
- Tiled VAE decode available for large frames
- Frame buffers cleaned after streaming

## Troubleshooting

### "Pipeline not initialized"

Make sure to pass `--config` or use `--mock`:

```bash
python streaming_app.py --config your_config.yaml
```

### Frames not appearing in browser

Enable Gradio queue:

```python
demo.queue()  # Already enabled in streaming_app.py
```

### Out of memory during VAE decode

Use tiled decoding:

```python
from streaming_vae import create_decoder

decoder = create_decoder(
    vae_path="...",
    use_tiled=True,
    tile_size=128,
)
```

## Future Improvements

1. **Async VAE Decode**: Overlap VAE decode with next DiT block
2. **WebSocket Streaming**: Direct frame streaming without polling
3. **HLS/DASH Output**: Standard adaptive streaming protocols
4. **Progressive JPEG**: Send low-quality preview, refine to full quality

## Files

```
video-streaming-develop/
├── streaming_pipeline.py    # Main streaming inference pipeline
├── streaming_vae.py         # Incremental VAE decoder
├── streaming_app.py         # Gradio web interface
├── STREAMING_README.md      # This file
└── run_inference_combined.py # Original non-streaming inference
```
