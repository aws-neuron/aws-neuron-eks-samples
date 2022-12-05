#!/bin/bash

CLUSTER_NAME=my-trn1-cluster
REGION_CODE=$(aws configure get region)

VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME \
    --query cluster.resourcesVpcConfig.vpcId --output text)
EKS_SG=$(aws eks describe-cluster --name $CLUSTER_NAME \
    --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)

LUSTRE_SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=eks-fsx-lustre-sg \
    --query SecurityGroups[].GroupId --output text)
[ -z $LUSTRE_SG ] && LUSTRE_SG=$(aws ec2 create-security-group --group-name eks-fsx-lustre-sg \
    --vpc-id $VPC_ID --description "Lustre Security Group for EKS" --query "GroupId" --output text)

aws ec2 authorize-security-group-ingress --group-id $LUSTRE_SG --protocol tcp --port 988 \
    --source-group $LUSTRE_SG --region $REGION_CODE
aws ec2 authorize-security-group-ingress --group-id $LUSTRE_SG --protocol tcp --port 988 \
    --source-group $EKS_SG --region $REGION_CODE

echo -e "\nVPC_ID:$VPC_ID EKS_SG:$EKS_SG LUSTRE_SG:$LUSTRE_SG"
