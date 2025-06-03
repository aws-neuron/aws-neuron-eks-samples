
* Fork https://github.com/aws-samples/scalable-hw-agnostic-inference and populate the `GITHUB_USER` and `GITHUB_OAUTH_TOKEN` based on `Settings/Developer Settings/Personal access tokens`.
* Check the latest [DLC](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) for `BASE_IMAGE_AMD_XLA_TAG` and `BASE_IMAGE_AMD_CUD_TAG` values.
* Export the following variables:
```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)
export AWS_REGION=us-west-2
export BASE_IMAGE_AMD_XLA_TAG=2.5.1-neuronx-py310-sdk2.21.0-ubuntu22.04
export IMAGE_AMD_XLA_TAG=amd64-neuron
export BASE_REPO=model
export BASE_TAG=multiarch-ubuntu
export BASE_AMD_TAG=amd64
export GITHUB_BRANCH=master
export GITHUB_USER=yahavb
export GITHUB_REPO=aws-neuron-eks-samples
export CF_STACK=flux-oci-image-inference-cdk
```
* Install needed packages

```bash
npm uninstall -g aws-cdk
npm install -g aws-cdk
```

* Deploy the pipeline

```bash
./deploy-pipeline.sh
```
