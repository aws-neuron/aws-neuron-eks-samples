"""Encode text prompts to T5 embeddings on Neuron.

Runs the UMT5-XXL text encoder on Neuron using torch.compile and saves prompt_embeds.pt,
so the main pipeline can load pre-computed embeddings instead of running T5 at inference time.

Usage:
    python encode_prompt_neuron.py --prompt "A cat walking on the beach"
    python encode_prompt_neuron.py --prompt_file prompts/my_prompts.txt --output prompt_embeds.pt
"""
import argparse
import os
import sys
import time

import torch

# Add gpu/RollingForcing to path for wan modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gpu", "RollingForcing"))

from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl


def main():
    parser = argparse.ArgumentParser(description="Encode text prompts with T5 on Neuron")
    parser.add_argument("--prompt", type=str, default=None, help="Single text prompt")
    parser.add_argument("--prompt_file", type=str, default=None, help="Text file with one prompt per line")
    parser.add_argument("--output", type=str, default="prompt_embeds.pt",
                        help="Output path. With --output_dir, saves per-prompt files instead.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save each prompt as a separate file: <output_dir>/prompt_000.pt, ...")
    parser.add_argument("--model_path", type=str, default="wan_models/Wan2.1-T2V-1.3B",
                        help="Path to Wan model directory")
    parser.add_argument("--device", type=str, default="neuron",
                        help="Device (neuron, neuron:0, neuron:1, cuda, cpu)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Skip torch.compile (run eager mode)")
    args = parser.parse_args()

    assert args.prompt or args.prompt_file, "Provide --prompt or --prompt_file"

    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load T5 encoder
    print("Loading UMT5-XXL encoder...")
    text_encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.bfloat16,
        device=torch.device('cpu')
    ).eval().requires_grad_(False)
    
    weights_path = os.path.join(args.model_path, "models_t5_umt5-xxl-enc-bf16.pth")
    print(f"Loading weights from {weights_path}...")
    text_encoder.load_state_dict(
        torch.load(weights_path, map_location='cpu', weights_only=False)
    )
    
    # Move to device
    print(f"Moving model to {device}...")
    text_encoder = text_encoder.to(device=device)

    # Load tokenizer
    tokenizer_path = os.path.join(args.model_path, "google/umt5-xxl/")
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_path, seq_len=512, clean='whitespace')

    # Warmup / shape establishment with dummy input
    print("Warmup pass for shape establishment...")
    dummy_ids, dummy_mask = tokenizer(["warmup text"], return_mask=True, add_special_tokens=True)
    dummy_ids = dummy_ids.to(device)
    dummy_mask = dummy_mask.to(device)
    with torch.no_grad():
        _ = text_encoder(dummy_ids, dummy_mask)
    torch.neuron.synchronize()

    # Compile with torch.compile (optional but recommended)
    if not args.no_compile:
        print("Compiling with torch.compile(backend='neuron')...")
        text_encoder.forward = torch.compile(
            text_encoder.forward, 
            backend="neuron", 
            fullgraph=True, 
            dynamic=False
        )
        
        # Warmup compiled model
        print("Warmup compiled model (NEFF compilation)...")
        start = time.time()
        with torch.no_grad():
            _ = text_encoder(dummy_ids, dummy_mask)
        torch.neuron.synchronize()
        print(f"Compilation done in {time.time() - start:.2f}s")

    # Encode prompts
    print(f"Encoding {len(prompts)} prompt(s)...")
    results = []
    for i, prompt in enumerate(prompts):
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_len = mask.gt(0).sum(dim=1).long()

        start = time.time()
        with torch.no_grad():
            context = text_encoder(ids, mask)
        torch.neuron.synchronize()
        elapsed = time.time() - start

        # Zero out padding positions
        context_cpu = context.cpu()
        context_cpu[0, seq_len[0].cpu():] = 0.0
        results.append(context_cpu)
        print(f"  [{i}] {elapsed:.3f}s: {prompt[:80]}...")

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for i, emb in enumerate(results):
            path = os.path.join(args.output_dir, f"prompt_{i:03d}.pt")
            torch.save(emb, path)
        print(f"Saved {len(results)} files to {args.output_dir}/")
    else:
        all_embeds = torch.cat(results, dim=0)
        torch.save(all_embeds, args.output)
        print(f"Saved {args.output}, shape={all_embeds.shape}, dtype={all_embeds.dtype}")


if __name__ == "__main__":
    main()
