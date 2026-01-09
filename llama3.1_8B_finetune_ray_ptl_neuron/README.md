# Instructions for fine-tuning LLama3.1 on AWS Trainium using Ray + Pytorch Lightning + AWS Neuron

## Overview <a name="overview2"></a>

This tutorial shows how to launch a Ray + PyTorch Lightning + AWS Neuron training job on multiple Trn1 nodes within an Amazon Elastic Kubernetes Service (EKS) cluster. In this example, the [Llama3.1 8B](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B) model will undergo fine-tuning using the opensource dataset: [Hugging face databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k). Ray will be used to launch the job on 2 trn1.32xlarge (or trn1n.32xlarge) instances, with 32 NeuronCores per instance.

### What are Ray, PyTorch Lightning, and AWS Neuron?

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) developed by Lightning AI organization, is a library that provides a high-level interface for PyTorch, and helps you organize your code and reduce boilerplate. By abstracting away engineering code, it makes deep learning experiments easier to reproduce and improves developer productivity.

[Ray](https://docs.ray.io/en/latest/ray-core/examples/overview.html) enhances ML workflows by seamlessly scaling fine-tuning and inference across distributed clusters, transforming single-node code into high-performance, multi-node operations with minimal effort.

[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) is an SDK with a compiler, runtime, and profiling tools that unlocks high-performance and cost-effective deep learning (DL) acceleration. It supports high-performance training on AWS Trainium instances. For model deployment, it supports high-performance and low-latency inference on AWS Inferentia.

### Combining Ray + PyTorch Lightning + AWS Neuron:
The integration of Ray, PyTorch Lightning (PTL), and AWS Neuron combines PTL's intuitive model development API, Ray Train's robust distributed computing capabilities for seamless scaling across multiple nodes, and AWS Neuron's hardware optimization for Trainium, significantly simplifying the setup and management of distributed training environments for large-scale AI projects, particularly those involving computationally intensive tasks like large language models.

The tutorial covers all steps required to prepare the EKS environment and launch the training job:

 1. [Sandbox Environment](#prepjumphost)
 2. [Setup EKS cluster and tools](#setupeksclusterandtools)
 3. [Create ECR repo and upload docker image](#createdockerimage)
 4. [Creating Ray Cluster](#creatingraycluster)
 5. [Preparing Data](#preparingdata)
 6. [Monitoring Jobs](#viewingraydashboard)
 7. [Fine-tuning Model](#finetuningmodel)
 8. [Deleting the environment](#cleanup)

# Multi-Node Ray + PyTorch Lightning + Neuron Flow

![Architecture Diagram](images/rayptlneuron-architecture.png)

## 1. Sandbox Environment <a name="prepjumphost"></a>

### 1.1 Launch a Linux jump host

<b>Supported Regions:</b>
Begin by choosing an AWS region that supports both EKS and Trainium (ex: us-west-2 / us-east-1 / us-east-2). 

In your chosen region (for ex: us-east-2), use the AWS Console or AWS CLI to launch an instance with the following configuration:

* **Instance Type:** m5.large
* **AMI:** Amazon Linux 2023 AMI (HVM)
* **Key pair name:** (choose a key pair that you have access to) 
* **Auto-assign public IP:** Enabled
* **Storage:** 100 GiB root volume

### 1.2 Configure AWS credentials on the jump host

#### Create a new IAM user in the AWS Console:

Refer to the [AWS IAM documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console) in order to create a new IAM user with the following parameters:

* **User name:** `eks_tutorial`
* **Select AWS credential type:** enable `Access key - Programmatic access`
* **Permissions:** choose _Attach existing policies directly_ and then select `AdministratorAccess`

Be sure to record the ACCESS_KEY_ID and SECRET_ACCESS_KEY that were created for the new IAM user.

#### Log into your jump host instance using one of the following techniques:

* Connect to your instance via the AWS Console using [EC2 Instance Connect](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Connect-using-EC2-Instance-Connect.html)
* SSH to your instance's public IP using the key pair you specified above.
  * Ex: `ssh -i KEYPAIR.pem ec2-user@INSTANCE_PUBLIC_IP_ADDRESS`

#### Configure the AWS CLI with your IAM user's credentials:

Run `aws configure`, entering the ACCESS_KEY_ID and SECRET_ACCESS_KEY you recorded above. For _Default region name_ be sure to specify the same region used to launch your jump host, ex: `us-east-2`.

<pre style="background: black; color: #ddd">
bash> <b>aws configure</b>
AWS Access Key ID [None]:  ACCESS_KEY_ID
AWS Secret Access Key [None]: SECRET_ACCESS_KEY
Default region name [None]: us-east-2
Default output format [None]: 
</pre>

## 2. Setup EKS cluster and tools <a name="setupeksclusterandtools"></a>

### 2.1 Setup kubectl, docker, terraform on your jump host

Before we begin, ensure you have all the prerequisites in place to make the deployment process smooth and hassle-free. Ensure that you have installed the following tools on your jump host.

* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
* [kubectl](https://kubernetes.io/docs/tasks/tools/)
* [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
* [docker](https://www.docker.com/)

<b>Automation for Pre-requisities:</b><br/>
To install all the pre-reqs above on the jump host, you can run this [script](https://github.com/awslabs/data-on-eks/blob/main/ai-ml/trainium-inferentia/examples/llama2/install-pre-requsites-for-ec2.sh) which is compatible with Amazon Linux 2023.

### 2.2 Clone the AI on EKS repository
```
cd ~
git clone https://github.com/awslabs/ai-on-eks.git
```

### 2.3 Setup EKS Cluster

Navigate to the trainium-inferentia directory:

```
cd ai-on-eks/infra/trainium-inferentia
```

Let's run the below export command to set environment variables.

```
# Set the region according to your requirements. Check Trn1 instance availability in the specified region.
export TF_VAR_region=us-east-2
```

Run the installation script to provision an EKS cluster with all the add-ons needed for the solution.

```
./install.sh
```

### 2.4 Verify the resources
Verify the Amazon EKS Cluster:
```
aws eks --region us-east-2 describe-cluster --name trainium-inferentia
```

```
# Creates k8s config file to authenticate with EKS
aws eks --region us-east-2 update-kubeconfig --name trainium-inferentia

kubectl get nodes # Output shows the EKS Managed Node group nodes
```

### 2.5 Verify if the Neuron Device Plugin is running

Use the following kubectl command:

<pre style="background: black; color: #ddd">
kubectl get ds neuron-device-plugin --namespace kube-system
NAME                           DESIRED CURRENT READY UP-TO-DATE AVAILABLE NODE SELECTOR AGE
neuron-device-plugin-daemonset 2         2      2        2          2      <none> 17d
</pre>

### 2.6 Verify that the EKS cluster has allocatable Neuron cores and devices

Use the following kubectl command:

<pre style="background: black; color: #ddd">
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,NeuronCore:.status.allocatable.aws\.amazon\.com/neuroncore"
NAME NeuronCore
ip-192-168-65-41.us-west-2.compute.internal 32
ip-192-168-87-81.us-west-2.compute.internal 32
</pre>

## 3. Create ECR repo and upload docker image to ECR <a name="createdockerimage"></a>

### 3.1 Clone this repo

On your jump host:

<pre style="background: black; color: #ddd">
sudo yum install -y git
git clone https://github.com/aws-neuron/aws-neuron-eks-samples.git
cd aws-neuron-eks-samples/llama3.1_8B_finetune_ray_ptl_neuron
</pre>

### 3.2 Execute the script

The script `0-kuberay-trn1-llama3-finetune-build-image.sh` checks if the ECR repo `kuberay_trn1_llama3.1_pytorch2` exists in the AWS Account and creates it if it does not exist. <br/><br/>
This script also builds the docker image and uploads the image to this repo. 

<pre style="background: black; color: #ddd">
bash> chmod +x 0-kuberay-trn1-llama3-finetune-build-image.sh
bash> ./0-kuberay-trn1-llama3-finetune-build-image.sh
bash> Enter the appropriate AWS region: For example: us-east-2
</pre>

If you have required credentials, the docker image should be successfully created and uploaded to Amazon ECR in the repository in the specific AWS region.

Verify if the repository `kuberay_trn1_llama3.1_pytorch2` is created successfully by heading to Amazon ECR service in AWS Console.

## 4. Creating Ray cluster <a name="creatingraycluster"></a>

The script `1-llama3-finetune-trn1-create-raycluster.yaml` creates Ray cluster with a head pod and worker pods.

Update the `<AWS_ACCOUNT_ID>` and `<REGION>` fields in the `1-llama3-finetune-trn1-create-raycluster.yaml` file using commands below (to reflect the correct ECR image ARN created above):

<pre style="background: black; color: #ddd">
bash> export AWS_ACCOUNT_ID=&lt;enter_your_aws_account_id&gt; # for ex: 111222333444
bash> export REGION=&lt;enter_your_aws_region&gt; # for ex: us-east-2
bash> sed -i "s/&lt;AWS_ACCOUNT_ID&gt;/$AWS_ACCOUNT_ID/g" *.yaml
bash> sed -i "s/&lt;REGION&gt;/$REGION/g" *.yaml
</pre>

Use the command below to create Ray cluster:
<pre style="background: black; color: #ddd">
kubectl apply -f 1-llama3-finetune-trn1-create-raycluster.yaml
kubectl get pods # Ensure all head and worker pods are in Running state
</pre>

The Ray cluster contains 1 head pod and 2 worker pods. Worker pods are deployed on the 2 Trainium instances (trn1.32xlarge). 

## 5. Preparing data <a name="preparingdata"></a>

Use the command below to submit a Ray job for downloading the [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset and the [Llama3.1 8B](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B) model:

<pre style="background: black; color: #ddd">
kubectl apply -f 2-llama3-finetune-trn1-rayjob-create-data.yaml
</pre>

You can check the output of `kubectl get pods` to find out if the job has completed:

<pre style="background: black; color: #ddd">
kubectl get pods
NAME                                              READY   STATUS      RESTARTS   AGE
2-llama3-finetune-trn1-rayjob-create-data-8qjfk   0/1     Completed   0          7m
cmd-shell                                         1/1     Running     0          10d
kuberay-trn1-head-zplg7                           1/1     Running     0          14m
kuberay-trn1-worker-workergroup-lwc2f             1/1     Running     0          14m
kuberay-trn1-worker-workergroup-zsm2z             1/1     Running     0          14m
</pre>

## 6. Monitoring jobs via Ray Dashboard <a name="viewingraydashboard"></a>

To view the Ray dashboard from the browser in your local machine:

<pre style="background: black; color: #ddd">
kubectl port-forward service/kuberay-trn1-head-svc 8265:8265 &
Head to: http://localhost:8265/ on your local browser.
</pre>

You can monitor the progress of the job in Ray Dashboard. 

## 7. Fine-tuning Llama3.1 8B model <a name="finetuningmodel"></a>

Use the command below to submit a Ray job for fine-tuning the model:

<pre style="background: black; color: #ddd">
kubectl apply -f 3-llama3-finetune-trn1-rayjob-submit-finetuning-job.yaml
</pre>

Model artifacts will be created under `/shared/trn1_llama_kuberay/neuron_cache`. Check the Ray logs for “Training Completed” message.

## 8. Clean-up <a name="cleanup"></a>

 When you are finished with the tutorial, run the following commands on the jump host to remove the EKS cluster and associated resources:

```
# Delete Ray Jobs
kubectl delete -f 3-llama3-finetune-trn1-rayjob-submit-finetuning-job.yaml
kubectl delete -f 2-llama3-finetune-trn1-rayjob-create-data.yaml

# Delete Ray Cluster
kubectl delete -f 1-llama3-finetune-trn1-create-raycluster.yaml

# Delete ECR Repo
Head to the AWS console and delete the ECR repo: kuberay_trn1_llama3.1_pytorch2

# Clean Up the EKS Cluster and Associated Resources:
cd data-on-eks/ai-ml/trainium-inferentia
./cleanup.sh

Terminate your EC2 jump host instance 

Delete the eks_tutorial IAM user via the AWS Console.
```

## Contributors<a name="contributors"></a>
Pradeep Kadubandi - AWS ML Engineer<br/>
Chakra Nagarajan - AWS Principal Specialist SA - Accelerated Computing<br/>
Sindhura Palakodety - AWS Senior ISV Generative AI Solutions Architect<br/>
Scott Perry - AWS Principal SA - Annapurna Labs<br/>
