#!/bin/bash -x
docker logout
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $NEURON_DLC_IMAGE 
docker pull $NEURON_DLC_IMAGE
dlc_image_id=$(docker images | grep $NEURON_DLC_IMAGE | awk '{print $3}')
docker images
docker logout

export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

ASSETS="-assets"
export BASE_IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$BASE_IMAGE_TAG
export ASSETS_IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$IMAGE_TAG$ASSETS
export IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$IMAGE_TAG

docker tag $dlc_image_id $IMAGE
docker images

cat Dockerfile.template | envsubst > Dockerfile
cat Dockerfile
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $BASE_IMAGE
docker build -t $IMAGE .
docker push $IMAGE
