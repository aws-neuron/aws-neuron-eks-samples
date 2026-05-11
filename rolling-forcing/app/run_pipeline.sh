#!/bin/bash
# Full video generation pipeline on Neuron
#
# This script runs the complete pipeline in three separate steps:
# 1. T5 text encoding - encodes prompt to embeddings
# 2. DiT inference (denoising) - generates latents from embeddings
# 3. VAE decode - converts latents to video
#
# Each step runs as a separate process, releasing HBM after completion.
# This allows the full pipeline to run on trn2.48xl by not loading all
# models simultaneously.
#
# Usage:
#   ./run_pipeline.sh --prompt "A cat walking on the beach" --output output.mp4
#   ./run_pipeline.sh --embedding prompt_embeds.pt --output output.mp4  # Skip T5

set -e

# Default values
CONFIG_PATH="configs/rolling_forcing_dmd.yaml"
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
DEVICE="neuron"
WORK_DIR="./pipeline_tmp"
NO_COMPILE=""
# Per-step device indices (empty = use DEVICE)
T5_DEVICE=""
DIT_DEVICE=""
VAE_DEVICE=""
# Per-step Neuron core pinning (e.g., "0" or "0,1" or "2-3")
T5_CORES=""
DIT_CORES=""
VAE_CORES=""

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
        --device)
            DEVICE="$2"
            shift 2
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
        --t5_cores)
            T5_CORES="$2"
            shift 2
            ;;
        --dit_cores)
            DIT_CORES="$2"
            shift 2
            ;;
        --vae_cores)
            VAE_CORES="$2"
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
            echo "  --num_frames N      Number of output frames (default: 21)"
            echo "  --seed N            Random seed (default: 0)"
            echo "  --fps N             Video FPS (default: 16)"
            echo "  --use_ema           Use EMA parameters from checkpoint"
            echo "  --device DEVICE     Device (neuron, cuda, cpu; default: neuron)"
            echo "  --t5_device DEV     Device for T5 encoder (e.g., neuron:0; default: --device)"
            echo "  --dit_device DEV    Device for DiT model (e.g., neuron:1; default: --device)"
            echo "  --vae_device DEV    Device for VAE decoder (e.g., neuron:0; default: --device)"
            echo "  --t5_cores CORES    Pin T5 to specific Neuron cores (e.g., 0 or 0,1)"
            echo "  --dit_cores CORES   Pin DiT to specific Neuron cores (e.g., 2-3)"
            echo "  --vae_cores CORES   Pin VAE to specific Neuron cores (e.g., 0)"
            echo "  --work_dir DIR      Directory for intermediate files"
            echo "  --no_compile        Skip torch.compile for T5 (run eager)"
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

# Set per-step devices to default if not specified
[[ -z "$T5_DEVICE" ]] && T5_DEVICE="$DEVICE"
[[ -z "$DIT_DEVICE" ]] && DIT_DEVICE="$DEVICE"
[[ -z "$VAE_DEVICE" ]] && VAE_DEVICE="$DEVICE"

# Create work directory
mkdir -p "$WORK_DIR"

# Set embedding path if not provided
if [[ -z "$EMBEDDING_PATH" ]]; then
    EMBEDDING_PATH="${WORK_DIR}/prompt_embeds.pt"
fi

# Set latent path
LATENT_PATH="${WORK_DIR}/latents.pt"

echo "=============================================="
echo "Video Generation Pipeline on Neuron"
echo "=============================================="
echo "Prompt:     ${PROMPT:-<using pre-computed embedding>}"
echo "Config:     $CONFIG_PATH"
echo "Output:     $OUTPUT_PATH"
echo "Checkpoint: ${CHECKPOINT_PATH:-<none>}"
echo "VAE:        $VAE_PATH"
echo "Model:      $MODEL_PATH"
echo "Frames:     $NUM_FRAMES"
echo "Device:     $DEVICE (T5=$T5_DEVICE, DiT=$DIT_DEVICE, VAE=$VAE_DEVICE)"
echo "Cores:      T5=${T5_CORES:-auto}, DiT=${DIT_CORES:-auto}, VAE=${VAE_CORES:-auto}"
echo "Work dir:   $WORK_DIR"
echo "=============================================="
echo ""

# Step 1: T5 Text Encoding (skip if embedding provided)
if [[ -n "$PROMPT" ]]; then
    echo "=============================================="
    echo "Step 1: T5 Text Encoding"
    echo "=============================================="

    T5_CMD="python encode_prompt_neuron.py \
        --prompt \"$PROMPT\" \
        --output $EMBEDDING_PATH \
        --model_path $MODEL_PATH \
        --device $T5_DEVICE"
    
    if [[ -n "$NO_COMPILE" ]]; then
        T5_CMD="$T5_CMD $NO_COMPILE"
    fi

    # Run with core pinning if specified
    if [[ -n "$T5_CORES" ]]; then
        echo "Pinning to Neuron cores: $T5_CORES"
        echo "Running: NEURON_RT_VISIBLE_CORES=$T5_CORES $T5_CMD"
        NEURON_RT_VISIBLE_CORES=$T5_CORES eval $T5_CMD
    else
        echo "Running: $T5_CMD"
        eval $T5_CMD
    fi

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
echo "Step 2: DiT Inference (Denoising)"
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

# Run with core pinning if specified
if [[ -n "$DIT_CORES" ]]; then
    echo "Pinning to Neuron cores: $DIT_CORES"
    echo "Running: NEURON_RT_VISIBLE_CORES=$DIT_CORES $DIT_CMD"
    NEURON_RT_VISIBLE_CORES=$DIT_CORES eval $DIT_CMD
else
    echo "Running: $DIT_CMD"
    eval $DIT_CMD
fi

if [[ ! -f "$LATENT_PATH" ]]; then
    echo "Error: DiT inference failed - no latent file produced"
    exit 1
fi

echo ""
echo "DiT inference complete. Latents saved to: $LATENT_PATH"
echo ""

# Step 3: VAE Decode
echo "=============================================="
echo "Step 3: VAE Decode"
echo "=============================================="

# Note: --no_compile is required for VAE on Neuron because torch.compile
# doesn't support float tensors as indices (used in grid sampling)
VAE_CMD="python run_vae_decode.py \
    --latent_path $LATENT_PATH \
    --output_path $OUTPUT_PATH \
    --vae_path $VAE_PATH \
    --device $VAE_DEVICE \
    --fps $FPS \
    --no_compile"

# Run with core pinning if specified
if [[ -n "$VAE_CORES" ]]; then
    echo "Pinning to Neuron cores: $VAE_CORES"
    echo "Running: NEURON_RT_VISIBLE_CORES=$VAE_CORES $VAE_CMD"
    NEURON_RT_VISIBLE_CORES=$VAE_CORES eval $VAE_CMD
else
    echo "Running: $VAE_CMD"
    eval $VAE_CMD
fi

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Output: $OUTPUT_PATH"
echo ""

# Optionally clean up intermediate files
# rm -rf "$WORK_DIR"
