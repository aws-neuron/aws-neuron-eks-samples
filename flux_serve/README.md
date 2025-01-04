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
* Create an S3 bucket to store the model compiled graph and [enable Amazon S3 objects with Mountpoint for Amazon S3 CSI driver](https://docs.aws.amazon.com/eks/latest/userguide/s3-csi.html)
* [Deploy the OCI image pipeline](./oci-image-build)
* [Deploy AWS Load Balancer controller](https://docs.aws.amazon.com/eks/latest/userguide/aws-load-balancer-controller.html) to enable public ingress access to the inference pods; export the k8s deployment as yaml and enforce nodeSelector to the non-neuron instances to avoid IMDS v1 limitation. 
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
kubectl apply -f specs/compile-flux-1024x576.yaml
kubectl apply -f specs/compile-flux-256x144.yaml
kubectl apply -f specs/compile-flux-512x512.yaml
```
Note the three pending Jobs that Karpenter seeks to fulfill. Current setup requires `aws.amazon.com/neuron: 8` which is half od `trn1.32xlarge` so expect two `trn1.xlarge` to be launched. 

* Deploy the Flux serving backend that loads the model from HuggingFace and uses the preloaded neuron model graph from S3 and standby for inference requests. The backend includes Deployment Pods and Services that route inference requests to the Pods so each model-api shapes scales horizontly.
```bash
kubectl apply -f specs/flux-neuron-1024x576-model-api.yaml
kubectl apply -f specs/flux-neuron-256x144-model-api.yaml
kubectl apply -f specs/flux-neuron-512x512-model-api.yaml
```
Note the three pending Deployment Pods that Karpenter seeks to fulfill. Current setup requires `aws.amazon.com/neuron: 8` which is half od `trn1.32xlarge` so expect two `trn1.xlarge` to be launched. 

* Deploy the Flux serving frontend that includes Gradio app. 
```bash
kubectl apply -f specs/flux-neuron-gradio.yaml
```

