#!/bin/bash

cat <<EOF > test_queue.yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: test
spec:
  weight: 1
  reclaimable: false
  capability:
    cpu: 2
EOF

kubectl apply -f test_queue.yaml
