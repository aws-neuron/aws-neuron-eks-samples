import torch
import os
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import datasets
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil
from datasets import load_dataset

save_dir="/shared"
save_model_dir="/shared/llama-3.1"
shutil.copy('config.json', os.path.join(save_model_dir, "config.json"))
snapshot_download(repo_id="NousResearch/Meta-Llama-3.1-8B", allow_patterns=["*.safetensors", "*.json"], ignore_patterns="config.json", local_dir=save_model_dir)
model = LlamaForCausalLM.from_pretrained(save_model_dir)
torch.save(model.state_dict(), os.path.join(save_dir, "llama3.1-8b-hf-pretrained.pt"))

#downloading dolly dataset
dataset="databricks/databricks-dolly-15k"
target_dataset_path = save_dir + "/" + dataset
load_dataset(dataset, cache_dir=target_dataset_path)
dolly_dataset = load_dataset(dataset, cache_dir=target_dataset_path)
print(f"Dataset downloaded and saved to: {target_dataset_path}")