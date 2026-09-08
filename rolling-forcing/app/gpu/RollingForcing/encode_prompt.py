"""Encode text prompts to T5 embeddings.

Runs the UMT5-XXL text encoder standalone and saves prompt_embeds.pt,
so the main pipeline can load pre-computed embeddings instead of
running T5 at inference time.

Usage:
    python encode_prompt.py --prompt "A cat walking on the beach"
    python encode_prompt.py --prompt_file prompts/my_prompts.txt --output prompt_embeds.pt
"""
import argparse

import torch
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl


def main():
    parser = argparse.ArgumentParser(description="Encode text prompts with T5")
    parser.add_argument("--prompt", type=str, default=None, help="Single text prompt")
    parser.add_argument("--prompt_file", type=str, default=None, help="Text file with one prompt per line")
    parser.add_argument("--output", type=str, default="prompt_embeds.pt",
                        help="Output path. With --output_dir, saves per-prompt files instead.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save each prompt as a separate file: <output_dir>/prompt_000.pt, ...")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    assert args.prompt or args.prompt_file, "Provide --prompt or --prompt_file"

    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    device = torch.device(args.device)

    # Load T5 encoder
    print("Loading UMT5-XXL encoder...")
    text_encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.bfloat16,
        device=torch.device('cpu')
    ).eval().requires_grad_(False)
    text_encoder.load_state_dict(
        torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                    map_location='cpu', weights_only=False)
    )
    text_encoder = text_encoder.to(device=device)

    tokenizer = HuggingfaceTokenizer(
        name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    # Encode one prompt at a time (T5 seq_len=512 is large, batch>1 may OOM)
    print(f"Encoding {len(prompts)} prompt(s)...")
    results = []
    for i, prompt in enumerate(prompts):
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_len = mask.gt(0).sum(dim=1).long()

        with torch.no_grad():
            context = text_encoder(ids, mask)

        context[0, seq_len[0]:] = 0.0
        results.append(context.cpu())
        print(f"  [{i}] done: {prompt[:80]}...")

    # Save
    if args.output_dir:
        import os
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
