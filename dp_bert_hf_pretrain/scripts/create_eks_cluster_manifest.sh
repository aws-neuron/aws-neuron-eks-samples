#!/bin/bash
source /tmp/ekstut/eks_azs.sh
REGION_CODE=$(aws configure get region)

cat <<EOF > eks_cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: my-trn1-cluster
  region: $REGION_CODE
  version: "1.23"

iam:
  withOIDC: true

availabilityZones: ["$EKSAZ1","$EKSAZ2"]

EOF

if [ $? -ne 0 ]
then
    echo -e "\nSomething went wrong.\n"
else
    echo -e "\nSuccessfully wrote eks_cluster.yaml\n"
fi
