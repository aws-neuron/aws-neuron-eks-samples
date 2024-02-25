#!/bin/bash -x

ASSETS="-assets"
BUCKET="sdinfer"
export IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$IMAGE_TAG$ASSETS
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $IMAGE
docker build -t $IMAGE --build-arg bucket=$BUCKET  -f Dockerfile-assets .
docker push $IMAGE
