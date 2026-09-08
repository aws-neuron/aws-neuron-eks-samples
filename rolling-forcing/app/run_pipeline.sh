#!/bin/bash
# Sequential video generation pipeline on Neuron.
#
# Each step runs on a different NeuronCore (device):
#   Step 1: T5 encoding     → neuron:0
#   Step 2: DiT denoising   → neuron:1
#   Step 3: VAE decode      → neuron:2
#
# Usage:
#   ./run_pipeline.sh --prompt "A cat walking on the beach" --output output.mp4
#   ./run_pipeline.sh --embedding prompt_embeds.pt --output output.mp4  # Skip T5

set -e

# Default values
# Use small config (30x52, 21 frames) to fit in ~11GB HBM per NC
# Full config (60x104, 126 frames) requires ~23GB and OOMs
CONFIG_PATH="configs/rolling_forcing_dmd_small.yaml"
PROMPT=""
EMBEDDING_PATH=""
OUTPUT_PATH="output.mp4"
CHECKPOINT_PATH=""
VAE_PATH="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
MODEL_PATH="wan_models/Wan2.1-T2V-1.3B"
NUM_FRAMES=21
SEED=0
FPS=16
USE_EMA=""
WORK_DIR="./pipeline_tmp"
NO_COMPILE=""

# Device assignments
T5_DEVICE="neuron:0"
DIT_DEVICE="neuron:1"
VAE_DEVICE="neuron:2"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --embedding)
            EMBEDDING_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --vae_path)
            VAE_PATH="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --use_ema)
            USE_EMA="--use_ema"
            shift
            ;;
        --work_dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --no_compile)
            NO_COMPILE="--no_compile"
            shift
            ;;
        --t5_device)
            T5_DEVICE="$2"
            shift 2
            ;;
        --dit_device)
            DIT_DEVICE="$2"
            shift 2
            ;;
        --vae_device)
            VAE_DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --prompt TEXT       Text prompt (required if --embedding not provided)"
            echo "  --embedding PATH    Pre-computed embedding .pt file (skip T5 step)"
            echo "  --config PATH       Config file path (default: configs/rolling_forcing_dmd.yaml)"
            echo "  --output PATH       Output video path (default: output.mp4)"
            echo "  --checkpoint PATH   Checkpoint file path (optional)"
            echo "  --vae_path PATH     VAE checkpoint path"
            echo "  --model_path PATH   Model directory (default: wan_models/Wan2.1-T2V-1.3B)"
            echo "  --num_frames N      Number of output frames (default: 126)"
            echo "  --seed N            Random seed (default: 0)"
            echo "  --fps N             Video FPS (default: 16)"
            echo "  --use_ema           Use EMA parameters from checkpoint"
            echo "  --work_dir DIR      Directory for intermediate files"
            echo "  --no_compile        Skip torch.compile for T5 (run eager)"
            echo "  --t5_device DEV     Device for T5 (default: neuron:0)"
            echo "  --dit_device DEV    Device for DiT (default: neuron:1)"
            echo "  --vae_device DEV    Device for VAE (default: neuron:2)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PROMPT" && -z "$EMBEDDING_PATH" ]]; then
    echo "Error: Either --prompt or --embedding is required"
    exit 1
fi

# Create work directory
mkdir -p "$WORK_DIR"

# Set embedding path if not provided
if [[ -z "$EMBEDDING_PATH" ]]; then
    EMBEDDING_PATH="${WORK_DIR}/prompt_embeds.pt"
fi

# Set latent path
LATENT_PATH="${WORK_DIR}/latents.pt"

echo "=============================================="
echo "Rolling Forcing Video Generation Pipeline"
echo "=============================================="
echo "Prompt:     ${PROMPT:-<using pre-computed embedding>}"
echo "Config:     $CONFIG_PATH"
echo "Output:     $OUTPUT_PATH"
echo "Checkpoint: ${CHECKPOINT_PATH:-<none>}"
echo "VAE:        $VAE_PATH"
echo "Model:      $MODEL_PATH"
echo "Frames:     $NUM_FRAMES"
echo "Devices:    T5=$T5_DEVICE, DiT=$DIT_DEVICE, VAE=$VAE_DEVICE"
echo "Work dir:   $WORK_DIR"
echo "=============================================="
echo ""

# Step 1: T5 Text Encoding (skip if embedding provided)
if [[ -n "$PROMPT" ]]; then
    echo "=============================================="
    echo "Step 1: T5 Text Encoding on $T5_DEVICE"
    echo "=============================================="

    T5_CMD="python encode_prompt_neuron.py \
        --prompt \"$PROMPT\" \
        --output $EMBEDDING_PATH \
        --model_path $MODEL_PATH \
        --device $T5_DEVICE"
    
    if [[ -n "$NO_COMPILE" ]]; then
        T5_CMD="$T5_CMD $NO_COMPILE"
    fi

    echo "Running: $T5_CMD"
    eval $T5_CMD

    if [[ ! -f "$EMBEDDING_PATH" ]]; then
        echo "Error: T5 encoding failed - no embedding file produced"
        exit 1
    fi

    echo ""
    echo "T5 encoding complete. Embeddings saved to: $EMBEDDING_PATH"
    echo ""
else
    echo "=============================================="
    echo "Step 1: T5 Text Encoding (SKIPPED - using pre-computed)"
    echo "=============================================="
    echo "Using: $EMBEDDING_PATH"
    echo ""
fi

# Step 2: DiT Inference
echo "=============================================="
echo "Step 2: DiT Inference on $DIT_DEVICE"
echo "=============================================="

DIT_CMD="python run_dit_inference.py \
    --config_path $CONFIG_PATH \
    --embedding_path $EMBEDDING_PATH \
    --output_path $LATENT_PATH \
    --num_output_frames $NUM_FRAMES \
    --seed $SEED \
    --device $DIT_DEVICE"

if [[ -n "$CHECKPOINT_PATH" ]]; then
    DIT_CMD="$DIT_CMD --checkpoint_path $CHECKPOINT_PATH"
fi

if [[ -n "$USE_EMA" ]]; then
    DIT_CMD="$DIT_CMD $USE_EMA"
fi

echo "Running: $DIT_CMD"
eval $DIT_CMD

if [[ ! -f "$LATENT_PATH" ]]; then
    echo "Error: DiT inference failed - no latent file produced"
    exit 1
fi

echo ""
echo "DiT inference complete. Latents saved to: $LATENT_PATH"
echo ""

# Step 3: VAE Decode
echo "=============================================="
echo "Step 3: VAE Decode on $VAE_DEVICE"
echo "=============================================="

VAE_CMD="python run_vae_decode.py \
    --latent_path $LATENT_PATH \
    --output_path $OUTPUT_PATH \
    --vae_path $VAE_PATH \
    --device $VAE_DEVICE \
    --fps $FPS"

echo "Running: $VAE_CMD"
eval $VAE_CMD

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Output: $OUTPUT_PATH"
echo ""
