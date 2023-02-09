# Training machine learning models using AWS Trainium

This topic describes how to create an Amazon EKS cluster with nodes running [Amazon EC2 Trn1](http://aws.amazon.com/ec2/instance-types/trn1/) instances, and optionally deploy a sample application. Amazon EC2 Trn1 instances are powered by [AWS Trainium](http://aws.amazon.com/machine-learning/trainium/) chips, which are custom-built by AWS to provide high performance and lowest cost training in the cloud. Machine learning models are deployed to containers using [AWS Neuron](http://aws.amazon.com/machine-learning/neuron/), a specialized software development kit (SDK) consisting of a compiler, runtime, and profiling tools that optimize the machine learning training performance of Trainium chips. AWS Neuron supports popular machine learning frameworks such as TensorFlow, PyTorch, and MXNet.

### Prerequisites

* [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) 
* [Install or upgrade eksctl to v0.123.0+](https://docs.aws.amazon.com/eks/latest/userguide/eksctl.html)
* [Install kubectl](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html)
* [Install Docker](https://docs.docker.com/get-docker/)
* (Optional) [Install Python 3+](https://www.python.org/downloads/)
* (Optional) [Install Finch (a Docker alternative) on macOS](https://github.com/runfinch/finch)

### Create a cluster

*To create a cluster with Trn1 Amazon EC2 instance nodes*

1. First, you will collect availability zones that support Trainium instance types in the region where you will provision the cluster (`us-west-2` in this case). Trainium instance types are generally available in select us-east-1 and us-west-2 regions, and we will continue to add support for more regions over time. Run the following command to determine which availability zones support Trn1 instances.  You may replace `trn1.32xlarge` with another [Trn1 instance type](https://aws.amazon.com/ec2/instance-types/trn1/) if you choose. This document covers a single-node example with trn1. For a multi-node example, see [Tutorial: Launch a Multi-Node PyTorch Neuron Training Job on Trainium Using TorchX and EKS](https://github.com/aws-neuron/aws-neuron-eks-samples/tree/master/dp_bert_hf_pretrain). 

```
aws ec2 describe-instance-type-offerings \
    --region us-west-2 \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=trn1.32xlarge" \
    --output text \
    --query 'InstanceTypeOfferings[].Location'
    
#Sample Output

us-west-2d
```

1. Create the cluster. The `eksctl` utility detects that you are launching a node group with a `Trn1` instance type and will start your nodes using one of the Amazon EKS-optimized accelerated Amazon Linux AMIs.


Eksctl requires 2 availability zones to be specified in the `--zones` flag.  One of the availability zones must be the zone where Trn1 instance types are available. The other can be any other available zone in the region. 

```
# Here are two snippets to improve the "copy-pastability"
CLUSTER_NAME=trainium-test
REGION=us-west-2
CLUSTER_AZS="us-west-2d,us-west-2a,us-west-2b"
SSH_PUBLIC_KEYNAME=your-public-keyname

# Create cluster command
eksctl create cluster \
    --name $CLUSTER_NAME \
    --region $REGION \
    --zones $CLUSTER_AZS \
    --nodegroup-name ng-trn1 \
    --node-type trn1.32xlarge \
    --nodes 1 \
    --node-zones us-west-2d \
    --nodes-min 1 \ 
    --nodes-max 4 \ 
    --ssh-access \ 
    --ssh-public-key $SSH_PUBLIC_KEYNAME \ 
    --with-oidc
```


Save the value of the following line returned in your command prompt. It's used later on in this tutorial (optional).

```
[9]  adding identity "arn:aws:iam::111122223333:role/eksctl-trainium-nodegroup-ng-in-NodeInstanceRole-FI7HIYS3BS09" to auth ConfigMap
```

[eksctl](https://eksctl.io/) will automatically install the AWS Neuron Kubernetes device plugin. This plugin advertises Neuron devices as a system resource to the [Kubernetes Scheduler](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/), which can be requested by a container.

1. Run the following command to make sure that all pods are up and running.

```
kubectl get pods -n kube-system
```

Abbreviated output:

```
NAME                                   READY   STATUS    RESTARTS   AGE
...
neuron-device-plugin-daemonset-6djhp   1/1     Running   0          5m
neuron-device-plugin-daemonset-hwjsj   1/1     Running   0          5m
```

# (Optional) Train a model

In this section, you’ll build and deploy a sample MLP training model that runs on the Trainium instance in your cluster. This example runs DataParallel training of the MLP model using 32 workers on a single trn1.32xlarge instance. For an example of distributed training using EKS with multiple trn1 instances, see Tutorial:[Launch a Multi-Node PyTorch Neuron Training Job on Trainium Using TorchX and EKS](https://github.com/aws-neuron/aws-neuron-eks-samples/tree/master/dp_bert_hf_pretrain).

## Create a Dockerfile

1. First, get the relevant MLP files from the AWS Neuron Samples repository. These files will be used in the steps to build the Docker image.

```
# Choose a work directory for mlp sample build
cd $HOME
WORKDIR=mlp-neuron-sample
mkdir $WORKDIR
cd $WORKDIR

git clone https://github.com/aws-neuron/aws-neuron-samples.git
cd aws-neuron-samples/torch-neuronx/training/mnist_mlp
mkdir -p $HOME/$WORKDIR/dockerbuild
cp train_torchrun.py $HOME/$WORKDIR/dockerbuild
cp model.py $HOME/$WORKDIR/dockerbuild
```

1. Next, create a Dockerfile in your work directory. 

```
cd $HOME/$WORKDIR

cat > dockerbuild/Dockerfile << EOF
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04
COPY model.py model.py
COPY train_torchrun.py train_torchrun.py
EOF
```

## Push the image to Amazon ECR

In this section, you will build the Docker image and push it to a repository in [Amazon ECR](https://aws.amazon.com/ecr/).

1. Login to ECR and authenticate with the repository that contains the base image used to build the image. 

```
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
```

1. Next, you will build the docker image. This process may take a while because of the size of the image being built.

```
# With docker
docker build $HOME/$WORKDIR/dockerbuild -t k8simage

# With finch
finch build $HOME/$WORKDIR/dockerbuild -t k8simage
```

1. Create a repository in the account where your cluster is running and push the built docker image to this ECR repo. Uploading the image may take some time depending on the size of the image and your network connection. 

```
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-west-2
REPO=eks_trn1_example

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker tag k8simage $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:mlp
aws ecr create-repository --repository-name $REPO
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:mlp

```

1. Create the YAML file for the Pod that will run the training job. Here are a couple items of note:

* `--nproc_per_node` should be set to the number of cores. For trn1.2xlarge: 2 and for trn1.32xlarge:32

* `aws.amazon.com/neuron: <NUM_DEVICES>`. For trn1.2xlarge:1. trn1.32xlarge:16

```
cd $HOME/$WORKDIR

cat > trn1_mlp.yaml << EOF
apiVersion: v1
kind: Pod
metadata:
  name: trn1-mlp
spec:
  restartPolicy: Never
  schedulerName: default-scheduler
  nodeSelector:
    beta.kubernetes.io/instance-type: trn1.32xlarge
  containers:
    - name: trn1-mlp
      command: ['torchrun']
      args:
        - '--nnodes=1'
        - '--nproc_per_node=32'
        - 'train_torchrun.py'
      image: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:mlp
      imagePullPolicy: IfNotPresent
      resources:
        limits: 
          aws.amazon.com/neuron: 16
EOF
```

1. Update `kubeconfig` with recently created cluster.

```
aws eks --region $REGION update-kubeconfig --name $CLUSTER_NAME
```

1. Apply the yaml file.

```
kubectl apply -f $HOME/trn1_mlp.yaml
```

1. Check that pod status is set to “Running”.

```
`kubectl get pods`
```

1. Check the logs.

```
kubectl logs trn1-mlp | grep loss
```

If everything worked, you should see the following in the logs:

```
Final loss is 0.1973
----------End Training
```

