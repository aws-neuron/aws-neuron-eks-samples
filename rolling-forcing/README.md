# Rolling Forcing Video Generation on EKS with AWS Trainium2

This sample demonstrates how to deploy [Rolling Forcing](https://github.com/TencentARC/RollingForcing) — a real-time video generation technique — on Amazon EKS using AWS Trainium2 (trn2) instances with Dynamic Resource Allocation (DRA) and the Neuron DRA driver.

## Overview

Rolling Forcing enables near-real-time text-to-video generation by applying Distribution Matching Distillation (DMD) to the Wan2.1 diffusion model, reducing denoising from 50 steps to just 5. This sample shows how to:

1. **Create an EKS cluster** with Trainium2 node groups using Capacity Reservations
2. **Configure DRA ResourceClaimTemplates** to allocate Neuron devices with logical NeuronCore slicing
3. **Deploy the inference backend** with tensor parallelism (TP=4) across NeuronCores
4. **Deploy a Gradio frontend** for interactive text-to-video generation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  EKS Cluster (rolling-forcing-sample)                   │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  Gradio Frontend │───▶│  Rolling Forcing Backend    │ │
│  │  (rf-gradio)     │    │  (rf - torchrun TP=4)      │ │
│  │  m5 node         │    │  trn2.48xlarge node         │ │
│  └─────────────────┘    │                             │ │
│                          │  ┌───────────────────────┐  │ │
│                          │  │ Neuron DRA Driver     │  │ │
│                          │  │ ResourceClaimTemplate │  │ │
│                          │  │ (s-lnc2-trn2)        │  │ │
│                          │  └───────────────────────┘  │ │
│                          └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- **AWS CLI** configured with permissions for EKS, EC2, and ECR
- **eksctl** >= 0.200.0
- **kubectl** >= 1.35
- **Capacity Reservation** for `trn2.48xlarge` in your target AZ
- **HuggingFace token** (for downloading Wan2.1 model weights)
- **Container image** with Neuron SDK pre-installed (PyTorch-NeuronX, NeuronX-CC)

## Directory Structure

```
rolling-forcing/
├── README.md                          # This file
├── cluster/
│   ├── eks-cluster.yaml               # EKS cluster definition
│   └── trn2-48xl-capacity-reservation-nodegroup.yaml  # Trainium2 nodegroup with ODCR
├── dra/
│   ├── s-trn2-rct.yaml               # Small: 1 device, 1 logical NeuronCore
│   ├── s-lnc2-trn2-rct.yaml          # Small: 1 device, 2 logical NeuronCores
│   └── m-trn2-rct.yaml               # Medium: 2 devices, 2 logical NeuronCores each (TP=4)
├── app/
│   ├── rf-deploy.yaml                # Backend inference deployment
│   ├── rf-gradio-cm.yaml             # Gradio frontend ConfigMap
│   └── rf-gradio-deploy.yaml         # Gradio frontend deployment
│   └── ...                            # Application source code
└── deploy/                            # Additional deployment manifests
```

## Setup Instructions

### Step 1: Create the EKS Cluster

```bash
eksctl create cluster -f cluster/eks-cluster.yaml
```

This creates a Kubernetes 1.35 cluster with a system node group (m5.xlarge) and standard EKS add-ons including the Mountpoint for S3 CSI driver (used for model weight caching).

### Step 2: Add Trainium2 Node Group with Capacity Reservation

Edit `cluster/trn2-48xl-capacity-reservation-nodegroup.yaml` to specify your:
- **Subnet ID** — must be in the same AZ as your Capacity Reservation
- **Capacity Reservation ID** — your ODCR for trn2.48xlarge

```bash
eksctl create nodegroup -f cluster/trn2-48xl-capacity-reservation-nodegroup.yaml
```

### Step 3: Install the Neuron DRA Driver

Install the Neuron device plugin and DRA driver. This enables Kubernetes to discover and allocate Neuron devices via ResourceClaims:

```bash
# Install the Neuron DRA driver (check latest version at https://github.com/aws-neuron/neuron-helm-charts)
helm install neuron-device-plugin oci://public.ecr.aws/neuron/neuron-helm-chart \
  --set "devicePlugin.enabled=true" \
  --set "draPlugin.enabled=true"
```

### Step 4: Apply ResourceClaimTemplates

ResourceClaimTemplates define how Neuron devices are sliced into logical NeuronCores. The Rolling Forcing backend uses `s-lnc2-trn2` (1 Neuron device with 2 logical NeuronCores) to run TP=4 inference:

```bash
kubectl apply -f dra/s-trn2-rct.yaml
kubectl apply -f dra/s-lnc2-trn2-rct.yaml
kubectl apply -f dra/m-trn2-rct.yaml
```

### Step 5: Create Secrets

```bash
# HuggingFace token for model downloads
kubectl create secret generic hf-token --from-literal=HF_TOKEN=<your-hf-token>

# GitHub token (if using private repo for app code)
kubectl create secret generic github-token --from-literal=GITHUB_TOKEN=<your-github-token>
```

### Step 6: Deploy the Backend

```bash
kubectl apply -f app/rf-deploy.yaml
```

The backend:
- Downloads Wan2.1-T2V-1.3B model weights (cached to S3 via Mountpoint)
- Downloads Rolling Forcing DMD checkpoint
- Launches inference with `torchrun --nproc_per_node=4` for TP=4
- Exposes a FastAPI endpoint on port 8000

### Step 7: Deploy the Gradio Frontend

```bash
kubectl apply -f app/rf-gradio-cm.yaml
kubectl apply -f app/rf-gradio-deploy.yaml
```

The Gradio UI connects to the backend service and provides an interactive text-to-video interface.

### Step 8: Access the Application

```bash
# Port-forward to access the Gradio UI locally
kubectl port-forward svc/rf-gradio 7860:7860
```

Then open http://localhost:7860 in your browser.

## Understanding DRA ResourceClaimTemplates

The Neuron DRA driver uses ResourceClaimTemplates to configure how Neuron devices are allocated to pods:

| Template | Devices | NeuronCores/Device | Total NeuronCores | Use Case |
|----------|---------|-------------------|-------------------|----------|
| `s-lnc1-trn2` | 1 | 1 | 1 | Small single-core workloads |
| `s-lnc2-trn2` | 1 | 2 | 2 | Standard inference (used by RF backend) |
| `m-trn2` | 2 | 2 | 4 | Tensor-parallel inference (TP=4) |

Key concepts:
- **`devicegroup1_id`** / **`devicegroup4_id`**: Device grouping constraints ensure allocated devices can communicate efficiently (same NUMA domain for TP)
- **`logicalNeuronCore`**: Slices a physical Neuron device into logical NeuronCores for fine-grained allocation
- **`deviceClassName: neuron.aws.com`**: The Neuron DRA driver device class

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `NEURON_LOGICAL_NC_CONFIG` | Number of logical NeuronCores per device (must match RCT) |
| `NEURON_CC_FLAGS` | Compiler flags (e.g., `--model-type=transformer`) |
| `TP_DEGREE` | Tensor parallelism degree (4 for this sample) |
| `USE_NKI_KERNELS` | Enable NKI custom kernels for attention |
| `RF_DEVICE_BACKEND` | Set to `neuron` for Trainium execution |

## Troubleshooting

- **Pod stuck in Pending**: Check that the Neuron DRA driver is installed and ResourceClaimTemplates are applied
- **Compilation timeout**: First run compiles Neuron graphs (NEFFs) which can take 30-60 minutes. Enable `USE_NEFF_CACHE=true` to cache compiled graphs to S3
- **OOM errors**: Ensure resource requests match available capacity (trn2.48xlarge has 192 NeuronCores, 1.5 TB memory)

## References

- [Rolling Forcing Paper](https://arxiv.org/abs/2503.07197)
- [Wan2.1 Video Model](https://github.com/Wan-Video/Wan2.1)
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
- [EKS DRA Documentation](https://docs.aws.amazon.com/eks/latest/userguide/manage-dra.html)
- [Neuron DRA Driver](https://github.com/aws-neuron/neuron-helm-charts)
