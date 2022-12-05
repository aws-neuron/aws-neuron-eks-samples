#!/bin/bash
CLUSTER_NAME=my-trn1-cluster
REGION_CODE=$(aws configure get region)

eksctl create iamserviceaccount \
    --name fsx-csi-controller-sa \
    --namespace kube-system \
    --cluster $CLUSTER_NAME \
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonFSxFullAccess \
    --approve \
    --role-name AmazonEKSFSxLustreCSIDriverFullAccess \
    --region $REGION_CODE \
    --override-existing-serviceaccounts

