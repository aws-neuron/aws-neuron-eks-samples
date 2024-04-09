# AWS Neuron EKS Samples

This repository contains samples for [Amazon Elastic Kubernetes Service (EKS)](https://aws.amazon.com/eks/) and [AWS Neuron](https://aws.amazon.com/machine-learning/neuron/), the software development kit (SDK) that enables machine learning (ML) inference and training workloads on the AWS ML accelerator chips [Inferentia](https://aws.amazon.com/machine-learning/inferentia/) and [Trainium](https://aws.amazon.com/machine-learning/trainium/).

The samples in this repository demonstrate the types of patterns that can be used to deliver inference and distributed training on EKS using Inferentia and Trainium. The samples can be used as-is, or easily modified to support additional models and use cases.

Samples are organized by use case below:

## Training

| Link | Description | Instance Type |
| --- | --- | --- |
| [BERT pretraining](dp_bert_hf_pretrain) | End-end workflow for creating an EKS cluster with 2 trn1.32xl nodes and running BERT phase1 pretraining (64-worker DataParallel)| Trn1 |
| [MLP training](mlp_train) | Introductory workflow for creating an EKS cluster with 1 node and running a simple MLP training job| Trn1 |

## Inference

| Link | Description | Instance Type |
| --- | --- | --- |
| [SD inference](sd_hf_serve) | SD Inference workflow for creating an inference endpoint forwarded by ALB LoadBalancer powered by Karpenter's NodePool | Inf2 |

## Getting Help

If you encounter issues with any of the samples in this repository, please open an issue via the GitHub Issues feature.

## Contributing

Please refer to the [CONTRIBUTING](CONTRIBUTING.md) document for details on contributing additional samples to this repository.


## Release Notes

Please refer to the [Change Log](releasenotes.md).

