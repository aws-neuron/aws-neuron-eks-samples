
# Distributed Training Concepts Guide

The guide describes the components used for the setup and run of the Trainium distributed training of a HuggingFace BERT model on EKS

* [Distributed Training](#DistributedTraining)

* [Distributed Training Components](#DistributedTrainingComponents)

* [Distributed Training Flow](#DistributedTrainingFlow)


## Distributed Training <a name="DistributedTraining"></a>


What is [Kubernetes?](https://kubernetes.io/docs/concepts/overview/)

Kubernetes, also known as K8s, is an open-source system for automating deployment, scaling, and management of containerized applications.

What is [Pytorch?](https://pytorch.org/)

PyTorch is a machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing,originally developed by Meta AI and now part of the Linux Foundation umbrella.

What is [TorchX?](https://pytorch.org/torchx/latest/)

TorchX is a universal job launcher for PyTorch applications. TorchX is designed to have fast iteration time for training/research and support for E2E production ML pipelines

What is [Volcano?](https://pytorch.org/torchx/0.1.0/schedulers/kubernetes.html)

Volcano is a Cloud native batch scheduling system for compute-intensive workloads 
TorchX kubernetes scheduler depends on volcano and requires etcd intalled for distributed job execution.

What is [etcd server?](https://etcd.io/)

Supports TorchX with parallel data distribution.
etcd is a strongly consistent, distributed key-value store that provides a reliable way to store data that needs to be accessed by a distributed system or cluster of machines. It gracefully handles leader elections during network partitions and can tolerate machine failure, even in the leader node.

[Kubernetes Architecture](https://kb.novaordis.com/index.php/Kubernetes_Control_Plane_and_Data_Plane_Concepts)

What is **distributed training** on Kubernetes?

Distributed training involves multiple nodes running a training on a cluster. The nodes are assigned ranks from 0 to N, with 0 being the master:
* Each node has a copy of the model
* Each node samples a partition of the dataset to train its model copy
* The gradiants are accumulated and synchronised to update the model

What is [EFA?](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)

An Elastic Fabric Adapter (EFA) is a network device that you can attach to your Amazon EC2 instance to accelerate High Performance Computing (HPC) and machine learning applications. EFA enables you to achieve the application performance of an on-premises HPC cluster, with the scalability, flexibility, and elasticity provided by the AWS Cloud

What is Amazon [FSx?](https://aws.amazon.com/fsx/lustre/)

FSx for Lustre is a fully managed shared storage built on the world's most popular high-performance file system



## Distributed Training Components <a name="DistributedTrainingComponents"></a>


## Distributed Training Flow <a name="DistributedTrainingFlow"></a>


