#!/bin/bash

# This script automates the process of building a multi-architecture Docker image
# and pushing it to an Amazon ECR (Elastic Container Registry) repository.
# The script performs the following steps:
# 1. Ensures the script is run on an x86_64-based instance.
# 2. Checks if Docker is installed and running.
# 3. Verifies that AWS CLI is installed and configured.
# 4. Prompts the user for the desired AWS region.
# 5. Checks if the specified ECR repository exists, and creates it if it does not.
# 6. Logs into Amazon ECR.
# 7. Creates a Docker Buildx builder instance to support multi-architecture builds.
# 8. Builds and pushes a multi-architecture Docker image to the specified ECR repository.

# Note: It is preferable to use AWS Cloud9 IDE with at least 100GB of storage and atleast 8GB memory for creating this image
# to avoid storage issues during the build process. You can use your local Mac or Windows machine,
# but ensure that you have enough memory and storage allocated for Docker to build this image.

# Replace with your desired repository name
ECR_REPO_NAME="kuberay_trn1_llama3.1_pytorch2"

# # Check that we are running on an x86_64 instance to avoid issues with docker build
# arch=$(uname -m)
# if [[ ! "$arch" = "x86_64" ]]; then
#   echo "Error: please run this script on an x86_64-based instance"
#   exit 1
# fi

# Check if docker is installed
junk=$(which docker 2>&1 > /dev/null)
if [[ "$?" -ne 0 ]]; then
  echo "Error: please install docker and try again. ex: for AL2023 you can run:"
  echo "  sudo yum install docker -y"
  echo "  sudo systemctl start docker"
  echo "  sudo usermod -aG docker ec2-user"
  echo "  newgrp docker"
  exit 1
fi

# Check that AWS CLI is installed and configured
junk=$(aws sts get-caller-identity)
if [[ "$?" -ne 0 ]]; then
  echo "Error: please make sure that the AWS CLI is installed and configured using 'aws configure'."
  exit 1
fi

# Prompt user for desired region
read -p "Enter the ECR region (ex: us-east-2): " region
echo $region > .eks_region

# Check if the ECR repository exists
if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$region" >/dev/null 2>&1; then
  echo "ECR repository '$ECR_REPO_NAME' already exists."

  # Get the ECR_REPO_URI for the existing repository
  ECR_REPO_URI=$(aws ecr describe-repositories --repository-name "$ECR_REPO_NAME" --query 'repositories[0].repositoryUri' --region "$region" --output text)
  echo "Repository URL: $ECR_REPO_URI"
else
  # Create the ECR repository
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$region"

  # Get the ECR_REPO_URI for the newly created repository
  ECR_REPO_URI=$(aws ecr describe-repositories --repository-name "$ECR_REPO_NAME" --query 'repositories[0].repositoryUri' --region "$region" --output text)
  echo "ECR repository '$ECR_REPO_NAME' created successfully."
  echo "Repository URL: $ECR_REPO_URI"
fi

# Store ECR REPO URI for later use
echo $ECR_REPO_URI > .ecr_repo_uri

# Login to ECR
echo -e "\nLogging in to ECR"
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin $ECR_REPO_URI
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${region}.amazonaws.com/pytorch-training-neuronx

# Create and use a new builder instance for multi-arch builds
docker buildx create --use --name mybuilder --driver docker-container
docker buildx inspect mybuilder --bootstrap

echo -e "\nBuilding kuberay_trn1 docker image" \
  && docker buildx build --platform linux/amd64 -t $ECR_REPO_URI:latest --build-arg REGION=$region . --push \
  && echo -e "\nImage successfully pushed to ECR"
