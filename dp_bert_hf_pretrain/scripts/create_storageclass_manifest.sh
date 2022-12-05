#!/bin/bash

CLUSTER_NAME=my-trn1-cluster
SUBNET_ID=$(aws ec2 describe-instances --filters Name=tag-value,Values=my-trn1-cluster \
    --query "Reservations[0].Instances[0].SubnetId" --output text)
#LUSTRE_SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=eks-fsx-lustre-sg \
#    --query SecurityGroups[].GroupId --output text)
LUSTRE_SG=$(aws cloudformation describe-stacks --stack-name eks-ng-stack \
    --query "Stacks[0].Outputs[?OutputKey=='LustreSecurityGroup'].OutputValue" \
    --output text)

cat <<EOF > storageclass.yaml
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: fsx-sc
provisioner: fsx.csi.aws.com
parameters:
  subnetId: $SUBNET_ID
  securityGroupIds: $LUSTRE_SG
  deploymentType: PERSISTENT_1
  automaticBackupRetentionDays: "1"
  dailyAutomaticBackupStartTime: "00:00"
  copyTagsToBackups: "true"
  perUnitStorageThroughput: "200"
  dataCompressionType: "NONE"
  weeklyMaintenanceStartTime: "7:09:00"
  fileSystemTypeVersion: "2.12"
mountOptions:
  - flock
EOF
