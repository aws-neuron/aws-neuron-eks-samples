#!/bin/bash
PROJECT_DIR=/shared/trn1_llama_kuberay

mkdir -p $PROJECT_DIR/hf_llama3.1-8b \
&& python3 download_llama_and_dolly.py --tmp_model_dir $PROJECT_DIR/hf_llama3.1-8b --output_dir $PROJECT_DIR \
&& mkdir -p $PROJECT_DIR/Meta-Llama-3.1-8B/pretrained_weight \
&& python3 convert_checkpoints.py --tp_size 8 --convert_from_full_state --config ./config.json \
  --input_dir $PROJECT_DIR/llama3.1-8b-hf-pretrained.pt \
  --output_dir $PROJECT_DIR/Meta-Llama-3.1-8B/pretrained_weight/
