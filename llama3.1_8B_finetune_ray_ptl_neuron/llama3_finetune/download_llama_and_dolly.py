import torch
import os
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from huggingface_hub import snapshot_download
import shutil
from datasets import load_dataset
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tmp_model_dir", required=True, type=str, help="temporary directory for downloading llama model checkpoint")
    parser.add_argument("--output_dir", required=True, type=str, help="output directory for storing modified llama checkpoint and dolly dataset files")
    args = parser.parse_args()

    # Download llama checkpoint
    shutil.copy('config.json', os.path.join(args.tmp_model_dir, "config.json"))
    snapshot_download(repo_id="NousResearch/Meta-Llama-3.1-8B", allow_patterns=["*.safetensors", "*.json"], ignore_patterns="config.json", local_dir=args.tmp_model_dir)
    model = LlamaForCausalLM.from_pretrained(args.tmp_model_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "llama3.1-8b-hf-pretrained.pt"))

    # Download dolly dataset
    dataset = "databricks/databricks-dolly-15k"
    target_dataset_path = os.path.join(args.output_dir, dataset)
    load_dataset(dataset, cache_dir=target_dataset_path)
    dolly_dataset = load_dataset(dataset, cache_dir=target_dataset_path)
    print(f"Dataset downloaded and saved to: {target_dataset_path}")
