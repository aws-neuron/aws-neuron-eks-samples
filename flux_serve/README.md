# Compile and serve ultra large vision transformers like Flux on Neuron devices at scale

challenges in deploying large models for serving: 
1/large model graph traces load - caching S3 buckets on EKS nodes with CSI drivers
https://docs.aws.amazon.com/eks/latest/userguide/s3-csi.html

Using DLCs to simplify OCI image build - easy to streamline with automation

https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/monitoring-tools.html#monitoring-tools are availible out of the box with container insights

cost effectiveness requires to allocate the minimal required cores. The device plugin and myscheduler through Helm simplify deployment
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/kubernetes-getting-started.html
Neuron device plugin, Neuron scheduler extension, Neuron scheduler

## Walkthrough
* [Create cluster with Karpenter node pools that provisions `trn1` instances](https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/)
* Create an S3 bucket to store the model compiled graph and export the S3 bucket name as follow:
```bash
export MODEL_GRAPH_BUCKET="black-forest-labs-flux1-dev-neuron"
```
* [Deploy your HggingFace user secret as a k8s secret](https://kubernetes.io/docs/concepts/configuration/secret/)
```bash
echo -n 'hf_myhftoken' | base64
```
replace the value with the `HUGGINGFACE_TOKEN` and apply the secret into the cluster
```yaml
apiVersion: v1
kind: Secret
type: Opaque
metadata:
  name: hf-secrets
  namespace: default
data:
  HUGGINGFACE_TOKEN: encodedhfmyhftoken
```

* [Enable Amazon S3 objects with Mountpoint for Amazon S3 CSI driver](https://docs.aws.amazon.com/eks/latest/userguide/s3-csi.html)
* [Deploy the OCI image pipeline](./oci-image-build)
* [Deploy the Neuron device plugin and scheduler extention](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/kubernetes-getting-started.html#deploy-neuron-device-plugin)
```bash
helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
    --set "npd.enabled=false"
helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
    --set "scheduler.enabled=true" \
    --set "npd.enabled=false"
``` 
* Deploy the Karpenter `NodeClass` and `NodePool` that provisions `trn1` instances upon requests (`nodeSelector:karpenter.sh/nodepool: amd-neuron-trn1`)
```bash
kubectl -f specs/amd-neuron-trn1-nodepool.yaml 
```
* Deploy the S3 CSI driver storage PersistentVolume and PersistentVolumeClaim that stores the compiled Flux graphs.
Edit `bucketName` for the right bucket name created; `accessModes` set to `ReadWriteMany` because we demonstrate graph compilation (upload) and serving (download).
Note the `PersistentVolumeClaim` name; we will need it for the app deployment.
```bash
kubectl apply -f specs/flux-model-s3-storage.yaml 
```
* Compile the model for the requires shapes. We will demonstrate three shapes: 1024x576, 256x144, and 512x512 with `bfloat16`
```bash
kubectl -f specs/compile-flux-1024x576.yaml
```

