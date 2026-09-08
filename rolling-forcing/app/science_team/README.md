# Rolling Forcing on Trn2

The Neuron Science Team has implemented distributed **Rolling Forcing**
Text-to-Video inference on AWS Trn2.
The pipeline runs end-to-end on a single Trn2 chip (8 NeuronCores at LNC=1),
sharded as:

| Stage | Script | Parallelism (8 ranks) |
| --- | --- | --- |
| T5 prompt encoder | `encode_prompt.py` | TP=8 |
| DiT denoising (rolling forcing) | `generate_latents.py` | TP=4 × SP=2 |
| VAE decoder | `decode_latents.py` | W-shard × 8 |
| All three fused | `e2e_pipeline.py` | TP=8 -> TP=4 × SP=2 -> W=8 |

We measured the following video generation performance (warm cache):

| Stage | Latency |
| --- | --- |
| T5 prompt encoding | 30.5 ms |
| DiT denoising (12 frames) | 1315.4 ms |
| VAE decoding (12 frames) | 485.6 ms |
| **Generation frame rate** | **6.66 fps** |


## 1. Install dependencies

> Assumes a Trn2 host with the Neuron driver, runtime, and tools already
> installed (`aws-neuronx-dkms`, `aws-neuronx-runtime-lib`,
> `aws-neuronx-collectives`, `aws-neuronx-tools`). If not, follow the
> [Neuron SDK setup guide](https://awsdocs-neuron.readthedocs-hosted.com/)
> first.

We use [`uv`](https://github.com/astral-sh/uv) for venv + install (much
faster than `pip` for the multi-GiB Neuron wheels).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if not already installed

cd rolling_forcing_neuron_science_team
uv venv --python 3.12
source .venv/bin/activate

# 1. AWS Neuron toolchain (public stable repo) — torch-neuronx, NKI, neuronx-cc.
uv pip install \
    --prerelease=allow --index-strategy unsafe-best-match \
    --extra-index-url https://pip.repos.neuron.amazonaws.com \
    torch-neuronx neuronx-cc nki

# 2. Modeling dependencies.
uv pip install \
    "diffusers==0.37.1" \
    "transformers==5.8.1" \
    "huggingface-hub==1.16.4" \
    "click" \
    "einops==0.8.2" \
    "omegaconf==2.3.0" \
    "ftfy==6.3.1" \
    "regex==2026.5.9" \
    "av==17.0.1"
```

Verify the install:

```bash
python -c "import torch, torch_neuronx, nki; print(torch.__version__, nki.__version__)"
```

> **Note:** The versions tested with this drop are the alpha-channel
> wheels:
> ```
> torch-neuronx        2.11.3.0.17324+4b683e9.dev
> neuronx-cc           2.0.256271.0a0+34d9d159
> nki                  0.4.0b4+25816723762.geeb7644d
> neuron-torch-mlir    20260507.107
> ```
> Reach out to the authors if you'd like access to these exact alpha-channel wheels.


## 2. Get model weights

Two checkpoint sets are required: the public **Wan2.1-T2V-1.3B** weights
(used by T5 and VAE) and the **rolling-forcing DMD** distilled DiT weights.

### Wan2.1-T2V-1.3B (T5 encoder + VAE decoder)

Pull from Hugging Face into `wan_models/`:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir wan_models/Wan2.1-T2V-1.3B
```

Layout after download:

```
wan_models/Wan2.1-T2V-1.3B/
├── Wan2.1_VAE.pth                       # VAE
├── models_t5_umt5-xxl-enc-bf16.pth      # T5 encoder
├── config.json
├── google/                              # T5 tokenizer/spm
└── ...
```

### Rolling-forcing DMD checkpoint (DiT)

Pull from Hugging Face into `checkpoints/`:

```bash
hf download TencentARC/RollingForcing \
    checkpoints/rolling_forcing_dmd.pt \
    --local-dir .
```

Final layout:

```
checkpoints/
└── rolling_forcing_dmd.pt
```


## 3. Get the deterministic CPU RNG states

The diffusion noise tensors are reproduced from saved per-prompt CPU RNG
states (`cpu_rng_states/prompt_NNN.pt`). They are used by
`generate_latents.py` and `e2e_pipeline.py` via `--rng_state_path` to
reproduce reference videos exactly.
To request them, please contact the authors.

Place the files at:

```
cpu_rng_states/
├── prompt_000.pt
├── prompt_001.pt
└── ...
```


## 4. Run

All three stages share the same 8-rank `torchrun` launch pattern. Run from
inside `rolling_forcing_neuron_science_team/` with the venv active.

### Option A — all stages in one launch (`e2e_pipeline.py`)

```bash
bash scripts/run_e2e_pipeline_distributed.sh
```

This loops over the prompts in `prompts/example_prompts.txt` and writes
`videos_pipeline/prompt_NNN.mp4`.

### Option B — per-stage runs (useful for debugging / profiling)

```bash
# 1. T5 → text_embeds/prompt_NNN.pt
bash scripts/run_encode_prompt_distributed.sh

# 2. DiT → output_latent.pt
bash scripts/run_generate_latents_distributed.sh

# 3. VAE → output.mp4
bash scripts/run_decode_latents_distributed.sh
```

## Layout

```
rolling_forcing_neuron_science_team/
├── encode_prompt.py            # T5 stage entry
├── generate_latents.py         # DiT stage entry
├── decode_latents.py           # VAE stage entry
├── e2e_pipeline.py             # Fused T5 + DiT + VAE entry
├── kernels/                    # NKI flash-attn / cache / RoPE / halo kernels
├── models/                     # T5, DiT, VAE — sharded for TP/SP/W
├── utils/                      # logging, parallel state, scheduler, video I/O
├── scripts/                    # 4× torchrun launchers
├── configs/                    # rolling-forcing config YAMLs
└── prompts/                    # sample prompt file
```


## License

Apache 2.0. See per-file headers.
