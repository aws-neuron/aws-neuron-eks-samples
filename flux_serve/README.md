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
* [Enable Amazon S3 objects with Mountpoint for Amazon S3 CSI driver](https://docs.aws.amazon.com/eks/latest/userguide/s3-csi.html)
* [Deploy the OCI image pipeline](./oci-image-build)
* 
