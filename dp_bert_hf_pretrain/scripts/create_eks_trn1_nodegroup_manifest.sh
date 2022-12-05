#!/bin/bash

source /tmp/ekstut/eks_azs.sh
REGION_CODE=$(aws configure get region)
LT_ID=$(aws cloudformation describe-stacks --stack-name eks-ng-stack \
	--query "Stacks[0].Outputs[?OutputKey=='LaunchTemplateID'].OutputValue" \
	--output text)

cat <<EOF > trn1_nodegroup.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: my-trn1-cluster
  region: $REGION_CODE
  version: "1.23"

iam:
  withOIDC: true

availabilityZones: ["$EKSAZ1","$EKSAZ2"]

managedNodeGroups:
  - name: trn1-ng1
    launchTemplate:
      id: $LT_ID
    minSize: 2
    desiredCapacity: 2
    maxSize: 2
    availabilityZones: ["$EKSAZ1"]
    privateNetworking: true
    efaEnabled: true
EOF
