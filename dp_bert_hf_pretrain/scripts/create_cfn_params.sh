#!/bin/bash

CLUSTER_SG=$(eksctl get cluster my-trn1-cluster -o json|jq -r ".[0].ResourcesVpcConfig.ClusterSecurityGroupId")
VPC_ID=$(eksctl get cluster my-trn1-cluster -o json|jq -r ".[0].ResourcesVpcConfig.VpcId")

cat <<EOF > cfn_params.json
[
    {
        "ParameterKey": "ClusterName",
        "ParameterValue": "my-trn1-cluster"
    },

    {
        "ParameterKey": "ClusterControlPlaneSecurityGroup",
        "ParameterValue": "$CLUSTER_SG"
    },

    {
        "ParameterKey": "VpcId",
        "ParameterValue": "$VPC_ID"
    }
]
EOF
