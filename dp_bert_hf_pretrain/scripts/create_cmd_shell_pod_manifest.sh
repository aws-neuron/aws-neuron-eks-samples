#!/bin/bash
ECR_REPO=$(aws ecr describe-repositories --repository-name eks_torchx_tutorial \
    --query repositories[0].repositoryUri --output text)

cat <<EOF > cmd_shell_pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: cmd-shell
spec:
  containers:
  - name: app
    image: $ECR_REPO:cmd_shell
    command: ["/bin/sh", "-c"]
    args: ["while true; do sleep 30; done"]
    volumeMounts:
    - name: persistent-storage
      mountPath: /data
  volumes:
  - name: persistent-storage
    persistentVolumeClaim:
      claimName: fsx-claim
  restartPolicy: Never
EOF
