# Disaggregated Inference on AWS Neuron (Trn2 / Trn3) with EKS

> **Accompanying manifests.** Every `*.yaml` referenced in this guide lives in
> this directory — the cluster + nodegroup config, the ResourceClaimTemplate, the
> prefill/decode Deployments, and both router setups. Clone the repo and apply
> them from here.

## 1. What's new

Disaggregated inference (DI) is now available on **AWS Neuron** — both **Trn2**
(`trn2.48xlarge`) and **Trn3** instances. DI splits the two phases of LLM serving
into independently scalable pools:

- **Prefill** (`kv_producer`) — processes the prompt and produces the KV cache.
- **Decode** (`kv_consumer`) — pulls that KV cache and generates tokens.

KV cache moves from prefill to decode over **NIXL on the `LIBFABRIC` backend**,
riding **EFA** (Elastic Fabric Adapter) for high-bandwidth, low-latency transfer
between pods. Prefill and decode can use different parallelism, letting you scale
each pool to its own bottleneck (prefill is compute-bound, decode is memory-bound).

This guide shows how to implement **dynamic xPyD** on EKS — start from a 1P1D
(one prefill, one decode) skeleton and grow or shrink each pool independently to
match business needs. Because prefill and decode are separate Deployments behind
an open-source routing layer ([**vLLM production-stack**](https://github.com/vllm-project/production-stack)
or [**AIBrix**](https://github.com/vllm-project/aibrix)), you can
scale toward **prefill-dominant** topologies (e.g. 3P1D for long-prompt,
low-generation traffic) or **decode-dominant** ones (e.g. 1P4D for short-prompt,
long-generation traffic) simply by changing replica counts — the router
discovers new pods by label and rebalances automatically, no redeploy of the
serving stack required.

## 2. Device allocation on EKS with DRA (Neuron + EFA)

NIXL/LIBFABRIC needs **both** a Neuron allocation and an EFA device in the same
pod, and — critically — from the **same PCIe/NUMA group** so the fabric path is
valid. We use Kubernetes **Dynamic Resource Allocation (DRA)** to request them
together.

### Prerequisite: an EFA-enabled Trn nodegroup

First create an EKS nodegroup of `trn2.48xlarge` (or `trn3-dev1.48xlarge`) with
EFA enabled. The details that matter (see
[`trn2-48xl-efa-ng.yaml`](./trn2-48xl-efa-ng.yaml) for a full example):

```yaml
nodeGroups:
  - name: trn2-48xl-efa
    instanceType: trn2.48xlarge      # or trn3-dev1.48xlarge
    privateNetworking: true          # EFA requires private networking
    efaEnabled: true                 # attaches the EFA interfaces
    capacityReservation:             # Trn EFA capacity is provisioned via an ODCR
      capacityReservationTarget:
        capacityReservationID: <your-capacity-reservation-id>
```

`efaEnabled: true` is what puts the EFA NICs on the nodes (without it there is no
fabric for NIXL to use); `privateNetworking: true` is required alongside it; and
Trn EFA capacity is typically obtained through an on-demand **capacity
reservation** (`capacityReservationID`). Create it with
`eksctl create nodegroup -f trn2-48xl-efa-ng.yaml`.

### Install the DRA drivers

Two DRA drivers must be installed; follow the upstream docs for the current,
authoritative steps rather than pinning commands here:

- **Neuron DRA driver** — publishes `neuron.aws.com` devices. Install via the
  Neuron DRA install script (do **not** also run the Neuron device plugin on the
  same cluster). See the
  [Neuron DRA guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.27.0/containers/neuron-dra.html).
- **EFA DRA driver (`aws-dranet`)** — publishes `efa.networking.k8s.aws` devices.
  Install per the
  [Amazon EKS DRA device-management guide](https://docs.aws.amazon.com/eks/latest/userguide/device-management-efa.html)
  (`eks/aws-dranet` Helm chart in `kube-system`).

Verify both device classes and their per-node devices are discovered:

```bash
kubectl get deviceclass                       # neuron.aws.com + efa.networking.k8s.aws
kubectl get resourceslice -o wide             # neuron + EFA devices per Neuron node
```

### The ResourceClaimTemplate

We pair the two device classes in one claim, constrained to a single
`devicegroup8_id` so the 8 NeuronCores and 8 EFA devices are co-located and
mutually routable. See [`xl-lnc2-trn2-efa-rct.yaml`](./xl-lnc2-trn2-efa-rct.yaml):

```yaml
spec:
  spec:
    devices:
      constraints:
      # Same PCIe/NUMA group for neurons AND efas — required for the NIXL EFA path.
      - matchAttribute: resource.aws.com/devicegroup8_id
        requests: [neurons, efas]
      requests:
      - name: neurons
        exactly:
          deviceClassName: neuron.aws.com
          allocationMode: ExactCount
          count: 8                                   # 8 chips (half a 16-chip node)
          selectors:
          - cel:
              expression: device.attributes['neuron.aws.com'].instanceType == 'trn2.48xlarge'
      - name: efas
        exactly:
          deviceClassName: efa.networking.k8s.aws
          allocationMode: ExactCount
          count: 8
      config:
      - requests: [neurons]
        opaque:
          driver: neuron.aws.com
          parameters:
            apiVersion: neuron.aws.com/v1
            kind: NeuronConfig
            logicalNeuronCore: 2                     # LNC=2
```

A pod references it via `resourceClaims` + `resources.claims`. On a 16-chip node,
two such claims (one per pod) let prefill and decode co-locate, each on its own
aligned neuron+EFA group. For Trn3, use the equivalent `trn3-dev1.48xlarge`
selector.

> **Tip:** validate the fabric before serving — run `/opt/amazon/efa/bin/fi_pingpong -p efa`
> (server) in the decode pod and `fi_pingpong -p efa <decode-pod-ip>` (client) in
> the prefill pod. A bandwidth table confirms EFA works pod-to-pod.

## 3. Deploy the 1P1D skeleton

Deploy prefill and decode as two Deployments —
[`prefill-deploy.yaml`](./prefill-deploy.yaml) and
[`decode-deploy.yaml`](./decode-deploy.yaml) — on EFA-enabled Trn2 nodes (pin both
to one host for single-node 1P1D, or spread across nodes for a cross-instance
topology):

```bash
kubectl apply -f prefill-deploy.yaml   # kv_producer, NIXL side-channel 5559
kubectl apply -f decode-deploy.yaml    # kv_consumer, NIXL side-channel 5659
kubectl get pods -l 'app in (prefill,decode)'
```

Each server runs `vllm serve` with:

```
--kv-transfer-config '{"kv_connector":"NeuronNixlConnector","kv_role":"kv_producer|kv_consumer",
                       "kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
```

For the full parameter walkthrough, xPyD scaling, and read-mode transfer details,
see the upstream tutorial:
<https://github.com/aws-neuron/private-vllm-neuron/blob/neuron-staging/docs/tutorials/tutorial-di-1p1d-xpyd.md>

### Scale the topology (dynamic xPyD)

Grow or shrink each pool independently with `kubectl scale` — the router
discovers the new pods by label and rebalances automatically. No change to the
serving config or the router is needed.

```bash
# Decode-dominant (short prompts, long generation) → 1P4D
kubectl scale deployment/decode --replicas=4

# Prefill-dominant (long prompts, short generation) → 3P1D
kubectl scale deployment/prefill --replicas=3

# Back to balanced 1P1D
kubectl scale deployment/prefill --replicas=1
kubectl scale deployment/decode  --replicas=1
```

Each new replica consumes its own aligned Neuron + EFA claim, so ensure the
cluster has enough Neuron capacity (and nodes) for the target replica count —
otherwise the extra pods stay `Pending` until DRA can satisfy their claims.

## 4. Front it with an open-source router

The prefill/decode pods carry labels for **both vLLM production-stack and
AIBrix**, so you can put either framework in front without redeploying the
servers:

```yaml
app: prefill|decode                 # production-stack
model: prefill|decode               # production-stack
role-name: prefill|decode           # AIBrix
model.aibrix.ai/name: gpt-oss-20b   # AIBrix
model.aibrix.ai/port: "8000"        # AIBrix
```

### vLLM production-stack

The production-stack router discovers the pods via Kubernetes labels and
orchestrates prefill→decode. See
[`production-stack-router-deploy.yaml`](./production-stack-router-deploy.yaml);
it runs `python -m vllm_router.app` with:

```
--routing-logic=disaggregated_prefill_orchestrated
--service-discovery=k8s
--k8s-label-selector="app in (prefill,decode)"
--prefill-model-labels=prefill  --decode-model-labels=decode
```

### AIBrix

AIBrix routing is the cluster gateway (not a router pod). Install the latest
release and select the prefill/decode router — **the `pd` routing algorithm ships
in the v0.7.0 release**, so no custom build is needed:

```bash
kubectl apply -f https://github.com/vllm-project/aibrix/releases/download/v0.7.0/aibrix-dependency-v0.7.0.yaml --server-side
kubectl apply -f https://github.com/vllm-project/aibrix/releases/download/v0.7.0/aibrix-core-crds-v0.7.0.yaml --server-side
kubectl apply -f https://github.com/vllm-project/aibrix/releases/download/v0.7.0/aibrix-core-v0.7.0.yaml

kubectl set env deployment/aibrix-gateway-plugins -n aibrix-system ROUTING_ALGORITHM=pd
kubectl set env deployment/aibrix-gateway-plugins -n aibrix-system AIBRIX_PREFILL_REQUEST_TIMEOUT=600
```

`ROUTING_ALGORITHM=pd` selects AIBrix's built-in **`pd` (Prefill-Decode)
disaggregation router** — it routes each request to a prefill pod first, then
hands the KV cache off to a decode pod. See the algorithm's reference in the
AIBrix repo:
[`pd_readme.md`](https://github.com/vllm-project/aibrix/blob/main/pkg/plugins/gateway/algorithms/pd_readme.md)
(and the `ROUTING_ALGORITHM` entry in
[`ENV_VARS.md`](https://github.com/vllm-project/aibrix/blob/main/pkg/plugins/gateway/ENV_VARS.md)).
Neuron/EFA support for that path landed via
[aibrix#1894](https://github.com/vllm-project/aibrix/pull/1894). See
[`aibrix-router-deploy.yaml`](./aibrix-router-deploy.yaml) for the model
registration (`ModelAdapter`) and Envoy timeout resources.

## 5. Send a request

**AIBrix** — through the Envoy gateway service
(`kubectl get svc -n envoy-gateway-system`):

```bash
kubectl run test --rm -it --image=curlimages/curl --restart=Never -- \
  curl -sS -X POST \
  "http://envoy-aibrix-system-aibrix-eg-903790dc.envoy-gateway-system:80/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20b","prompt":"Count the numbers 1, 2, 3","max_tokens":50}'
```

**production-stack** — through the router service on port 8000:

```bash
kubectl run test --rm -it --image=curlimages/curl --restart=Never -- \
  curl -sS -X POST \
  "http://router.default:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20b","prompt":"Count the numbers 1, 2, 3","max_tokens":50}'
```

Both return an OpenAI-compatible completion — the router sent the prompt to a
prefill pod, which produced the KV cache; a decode pod pulled it over NIXL/LIBFABRIC/EFA
and generated the tokens.

## Conclusion

Disaggregated inference on AWS Neuron is available today on **Trn2 and Trn3**
instances, deployable on EKS with DRA-based Neuron + EFA allocation, and routable
through open-source platforms — **vLLM production-stack** and **AIBrix**. Support
for additional serving platforms is in progress.

## Get started

- **Try it now** — clone this repo, create an EFA-enabled Trn nodegroup, and
  deploy the 1P1D skeleton with the prefill/decode manifests in this directory.
- **Scale to your workload** — use `kubectl scale` to grow into prefill-dominant
  (xP1D) or decode-dominant (1PyD) topologies as your traffic mix demands.
- **Pick your router** — front the pools with vLLM production-stack or AIBrix; the
  pods already carry labels for both, so switching is a one-line change.
- **Go deeper** — see the upstream DI tutorial for xPyD, parallelism, and
  read-mode transfer:
  <https://github.com/aws-neuron/private-vllm-neuron/blob/neuron-staging/docs/tutorials/tutorial-di-1p1d-xpyd.md>
- **Tell us what to enable next** — file an issue on the AWS Neuron repo with the
  serving platform, model, or topology you want supported, and contributions to
  add more platforms are welcome.
