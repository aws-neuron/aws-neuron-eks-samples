# TP QK-Norm Bug Fix

## Issue
WAN2.1-T2V-1.3B with TP4 produced visibly degraded video quality compared to the no-TP reference (same model, single device).

## Root Cause
**Local RMSNorm on sharded Q/K features gave wrong normalization.**

In WAN's cross-attention, Q and K are normalized by `WanRMSNorm(dim=1536)` before the attention kernel. With TP4, Q/K projections are column-parallel (each rank holds 384 features = 3 heads). The old `shard_qkv_norm` created a plain `WanRMSNorm(384)` that computed:

```
rms = sqrt(mean(x² over 384 local features))
```

The correct computation requires the mean over all **1536** features globally, because different heads have vastly different magnitudes (observed stds: 0.004 to 0.047 across ranks). Local normalization artificially equalizes all ranks, boosting near-zero heads and causing ~2× amplification in cross-attention output.

## Fix
Introduced `TPRMSNorm` in `models/tp_utils.py` which:
1. Computes `sum(x²)` locally (384 features)
2. All-reduces the sum across 4 TP ranks
3. Divides by `global_dim=1536` to get the correct global mean
4. Applies `rsqrt(global_mean + eps)` as the normalization factor

This gives mathematically identical results to the non-TP `WanRMSNorm(1536)`.

## Commit
`004ab8c` on branch `wan1.3b-tp`
