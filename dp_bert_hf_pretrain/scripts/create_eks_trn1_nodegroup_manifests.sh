#!/bin/bash

source /tmp/ekstut/eks_azs.sh
REGION_CODE=$(aws configure get region)
LT_ID_TRN1=$(aws cloudformation describe-stacks --stack-name eks-trn1-ng-stack \
	--query "Stacks[0].Outputs[?OutputKey=='LaunchTemplateIdTrn1'].OutputValue" \
	--output text)
LT_ID_TRN1N=$(aws cloudformation describe-stacks --stack-name eks-trn1-ng-stack \
	--query "Stacks[0].Outputs[?OutputKey=='LaunchTemplateIdTrn1n'].OutputValue" \
	--output text)

cat <<EOF > trn1_nodegroup.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: my-trn1-cluster
  region: $REGION_CODE
  version: "1.25"

iam:
  withOIDC: true

availabilityZones: ["$EKSAZ1","$EKSAZ2"]

managedNodeGroups:
  - name: trn1-32xl-ng1
    launchTemplate:
      id: $LT_ID_TRN1
    minSize: 2
    desiredCapacity: 2
    maxSize: 2
    availabilityZones: ["$EKSAZ1"]
    privateNetworking: true
    efaEnabled: true
EOF


cat <<EOF > trn1n_nodegroup.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: my-trn1-cluster
  region: $REGION_CODE
  version: "1.25"

iam:
  withOIDC: true

availabilityZones: ["$EKSAZ1","$EKSAZ2"]

managedNodeGroups:
  - name: trn1n-32xl-ng1
    launchTemplate:
      id: $LT_ID_TRN1N
    minSize: 2
    desiredCapacity: 2
    maxSize: 2
    availabilityZones: ["$EKSAZ1"]
    privateNetworking: true
    efaEnabled: true
EOF
