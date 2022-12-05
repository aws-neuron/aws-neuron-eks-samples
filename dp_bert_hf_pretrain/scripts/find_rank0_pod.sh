#!/bin/bash
# Determine which of the running pods is the rank0 worker for the BERT
# pretraining job. This is useful because only the rank0 worker outputs
# training metrics to standard output, and rank is randomly assigned
# during the TorchElastic rendezvous during job initialization.

running_pods=$(kubectl get pods -o json| jq -r '.items[] 
    | select(.status.phase == "Running") 
    | select(.spec.schedulerName == "volcano") 
    | .metadata.name')

FOUND=0
for pod in $running_pods; do
    kubectl logs $pod|grep -q "rank 0"; 
    if [ $? -eq 0 ]
    then
        echo "$pod is your rank0 worker pod."
        FOUND=1
    fi
done

if [ $FOUND -ne 1 ]
then
    echo "No rank0 worker pod found!"
    echo "Please make sure that you have a training job running, and try again."
fi
