"""T5 text encoding on Neuron (single device/core).

Step 1 of the pipeline: Encode text prompt to embeddings on neuron:0.

Usage:
    python encode_prompt_neuron.py \
        --prompt "A cat walking on the beach" \
        --output prompt_embeds.pt \
        --device neuron:0
"""
import argparse
import os
import time

import torch

from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl


def main():
    parser = argparse.ArgumentParser(description="T5 encoding on Neuron")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to encode")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for embeddings .pt file")
    parser.add_argument("--model_path", type=str, default="wan_models/Wan2.1-T2V-1.3B",
                        help="Path to model directory")
    parser.add_argument("--device", type=str, default="neuron:0",
                        help="Device (neuron:0, neuron:1, etc.)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Skip torch.compile (run eager)")
    args = parser.parse_args()

    print(f"[encode_prompt] Starting T5 encoding on {args.device}")
    print(f"  Prompt: {args.prompt[:100]}...")
    print(f"  Output: {args.output}")

    device = torch.device(args.device)

    # Load T5 model
    print("[encode_prompt] Loading UMT5-XXL encoder...")
    text_encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.bfloat16,
        device=torch.device('cpu')
    ).eval().requires_grad_(False)

    text_encoder.load_state_dict(
        torch.load(f"{args.model_path}/models_t5_umt5-xxl-enc-bf16.pth",
                   map_location='cpu', weights_only=False, mmap=True)
    )

    # Move to specific Neuron device (core)
    print(f"[encode_prompt] Moving T5 to {args.device}...")
    text_encoder = text_encoder.to(device=device)

    # Compile with torch.compile
    if not args.no_compile:
        print("[encode_prompt] Compiling T5 with torch.compile(backend='neuron')...")
        compiled_encoder = torch.compile(
            text_encoder,
            backend='neuron',
            fullgraph=True,
            dynamic=False
        )

        # Warmup pass
        print("[encode_prompt] Running warmup pass...")
        warmup_start = time.time()
        dummy_ids = torch.zeros(1, 512, dtype=torch.long, device=device)
        dummy_mask = torch.ones(1, 512, dtype=torch.long, device=device)
        _ = compiled_encoder(dummy_ids, dummy_mask)
        print(f"[encode_prompt] Warmup complete in {time.time() - warmup_start:.2f}s")
    else:
        compiled_encoder = text_encoder
        print("[encode_prompt] Running in eager mode (no compile)")

    # Load tokenizer
    tokenizer = HuggingfaceTokenizer(
        name=f"{args.model_path}/google/umt5-xxl/", seq_len=512, clean='whitespace')

    # Encode prompt
    print("[encode_prompt] Encoding prompt...")
    encode_start = time.time()
    
    ids, mask = tokenizer([args.prompt], return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    
    with torch.no_grad():
        context = compiled_encoder(ids, mask)
    
    for u, v in zip(context, seq_lens):
        u[v:] = 0.0

    print(f"[encode_prompt] Encoding complete in {time.time() - encode_start:.2f}s")
    print(f"[encode_prompt] Embeddings shape: {context.shape}")

    # Save embeddings
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(context.cpu(), args.output)
    print(f"[encode_prompt] Saved to {args.output}")
    print("[encode_prompt] Done!")


if __name__ == "__main__":
    main()
