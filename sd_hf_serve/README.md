## Deploy sd2_512 inference endpoint

This is a StableDiffusionPipeline based on `stabilityai/stable-diffusion-2-1-base`. Updated compile and benchmark code is in [sd2_512_benchmark](https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/benchmark/pytorch/sd2_512_benchmark.py) and [sd2_512_compile](https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/benchmark/pytorch/sd2_512_compile.py)

* [Create cluster with Karpenter node pools that provisions `inf2` instances](https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/)
*  Deploy karpenter nodepool for inferentia
```bash
  cat inf2-nodepool.yaml | envsubst | kubectl apply -f -  
```
* Deploy the Neuron plugin 
```bash
  kubectl apply -f k8s-neuron-device-plugin-rbac.yml
  kubectl apply -f k8s-neuron-device-plugin.yml
  kubectl apply -f k8s-neuron-scheduler-eks.yml
  kubectl apply -f my-scheduler.yml 
```
* [Deploy the OCI image pipeline](./oci-image-build)
* Allow access to model assets S3 bucket using pod identity
```bash
kubectl apply -f sd21-sa.yaml
aws iam create-policy --policy-name allow-access-to-model-assets --policy-document file://allow-access-to-model-assets.json
aws iam create-role --role-name allow-access-to-model-assets --assume-role-policy-document file://trust-relationship.json --description "allow-access-to-model-assets"
aws iam attach-role-policy --role-name allow-access-to-model-assets --policy-arn=arn:aws:iam::133573815653:policy/allow-access-to-model-assets
aws eks create-pod-identity-association --cluster-name yahavb-neuron-demo --role-arn arn:aws:iam::133573815653:role/allow-access-to-model-assets --namespace default --service-account sd21-sa
```
* Deploy a job that compiles the model with Neuron SDK and stage it in S3
```bash

```
