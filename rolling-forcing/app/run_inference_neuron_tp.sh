#!/bin/bash
# Wan2.1-T2V-1.3B inference with Tensor Parallelism (TP=4) on Trainium
#
# Architecture:
#   - 4 NeuronCores across 2 NDs (single chip), all dedicated to DiT via TP
#   - 1.3B: dim=1536, 12 heads, 30 layers → 3 heads/rank
#   - T5 on rank 2 (ND1), VAE on rank 0 (ND0) — separate HBM banks
#   - All-reduce communication for O-proj and FFN-down after each block
#
# Memory layout:
#   Bank 0 (ND0): rank 0 (DiT/4 + VAE) + rank 1 (DiT/4)  ≈ 5 GB
#   Bank 1 (ND1): rank 2 (DiT/4 + T5)  + rank 3 (DiT/4)  ≈ 12 GB
#
# Prerequisites:
#   - wan_models/Wan2.1-T2V-1.3B/ directory with HF diffusers weights + T5 + VAE

set -e

# Number of NeuronCores (TP degree)
NPROC=${NPROC:-4}

echo "============================================================"
echo "Wan2.1-T2V-1.3B Inference Server (TP=${NPROC})"
echo "  Config:     ${CONFIG_PATH:-configs/rolling_forcing_dmd_1.3b_tp4.yaml}"
echo "  Model:      ${MODEL_PATH:-wan_models/Wan2.1-T2V-1.3B}"
echo "  T5 rank:    ${T5_RANK:-2} (ND1)"
echo "  VAE rank:   0 (ND0)"
echo "============================================================"

torchrun --nproc_per_node=$NPROC inference_neuron_tp.py
