# Video Streaming Develop

## Setup

| Property | Value |
|----------|-------|
| **Repository** | https://github.com/yahavb/video-streaming-develop.git |
| **Branch** | `main` |
| **Instance Type** | `trn2.48xlarge` |
| **Neuron Devices** | 16 devices × 4 cores = 64 Neuron Cores (96 GB per device) |

## Execution

### Single device (sequential)

```bash
./run_pipeline.sh --prompt "A cat walking on the beach" \
  --config configs/rolling_forcing_dmd_small.yaml \
  --output output.mp4 \
  --checkpoint checkpoints/rolling_forcing_dmd.pt \
  --use_ema
```

### Multi-device (separate Neuron devices)

```bash
./run_pipeline.sh --prompt "A cat walking on the beach" \
  --config configs/rolling_forcing_dmd_small.yaml \
  --output output.mp4 \
  --checkpoint checkpoints/rolling_forcing_dmd.pt \
  --use_ema \
  --t5_device neuron:0 \
  --dit_device neuron:1 \
  --vae_device neuron:2
```

### Core pinning (specific NeuronCore IDs)

```bash
./run_pipeline.sh --prompt "A cat walking on the beach" \
  --config configs/rolling_forcing_dmd_small.yaml \
  --output output.mp4 \
  --checkpoint checkpoints/rolling_forcing_dmd.pt \
  --use_ema \
  --t5_device neuron \
  --dit_device neuron \
  --vae_device neuron \
  --t5_cores 0 \
  --dit_cores 4-7 \
  --vae_cores 8
```

## Run unit tests

```bash
python -m pytest tests -n auto -v
```
