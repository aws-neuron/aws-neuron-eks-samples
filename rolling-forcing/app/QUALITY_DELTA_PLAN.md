# WAN2.1-T2V-1.3B DiT QUALITY ISSUE - Debugging Analysis

## Branch: `wan1.3b-tp`

## Problem Summary

The capacity tracking code in `inference_neuron_tp.py` has a **structural bug** in the streaming endpoint that can cause the server to permanently reject requests with HTTP 429 after the first streaming call.

## Root Cause

The issue is in the `/generate/stream` endpoint's interaction between the capacity tracking lock and Starlette's `StreamingResponse`:

```python
@app.post("/generate/stream")
async def generate_video_streaming(request):
    # Lock acquired HERE (in outer async function)
    _inference_lock.acquire(blocking=False)
    _is_busy = True
    
    async def generate_frames():
        try:
            # ... streaming logic ...
        finally:
            # Lock released HERE (in inner generator)
            _is_busy = False
            _inference_lock.release()
    
    return StreamingResponse(generate_frames(), ...)
```

### Why This Is Broken

1. **Lock acquired in outer function, released in inner generator**: The lock is acquired in the `generate_video_streaming()` async function, but released inside the `generate_frames()` async generator's `finally` block.

2. **Generator execution is deferred**: With Starlette's `StreamingResponse`, the generator (`generate_frames()`) doesn't actually start running until the response is being sent to the client. This creates a gap between lock acquisition and the start of the work protected by the lock.

3. **Lock may never be released**: If anything goes wrong (client disconnect, timeout, internal error during generation), the lock may never be properly released — meaning the next call to that pod gets a 429 forever.

4. **`threading.Lock` in async context is fragile**: Using a `threading.Lock` in an async (uvicorn) context is inherently fragile and can lead to deadlocks or race conditions.

5. **Redundancy**: The lock and the readiness probe logic (503 when busy) are **redundant** anyway because the `dist.broadcast` calls inside the inference pipeline are inherently serialized across all TP ranks — only one request can physically run at a time.

## Affected Code

The capacity tracking variables and logic:

```python
# ── Capacity tracking: one request at a time ──────────────────────────
# When busy, readiness probe returns 503 → K8s removes pod from Service
# endpoints → new requests route to other available replicas.
_inference_lock = threading.Lock()
_is_busy = False
```

### In `/generate` endpoint:
- Lock acquisition with `_inference_lock.acquire(blocking=False)`
- 429 rejection if lock can't be acquired
- Lock release in `finally` block

### In `/generate/stream` endpoint:
- Lock acquisition in outer function
- 429 rejection if lock can't be acquired
- Lock release in inner generator's `finally` block (THE BUG)

### In `/readiness` endpoint:
- Returns 503 when `_is_busy` is True → K8s removes pod from Service endpoints

## The Fix

**Remove the capacity tracking logic entirely.** Specifically:

1. Remove `_inference_lock` and `_is_busy` variables
2. Remove the 429 rejection logic from `/generate` and `/generate/stream`
3. Remove the 503 logic from `/readiness`
4. Restore the original try/except structure

### Why Removal Is Safe

- The `dist.broadcast` calls inside the inference pipeline are inherently serialized across all TP ranks — only one request can physically run at a time on the Neuron hardware.
- The server naturally handles requests one at a time because the blocking collective operations (`dist.broadcast`) prevent concurrent execution.
- The Gradio app handles 429 from the service (checking `response.status_code == 429`), but that code path simply won't be triggered anymore.

## Impact

- The server will return to its previous working state where it handles requests one at a time naturally (since `dist.broadcast` is blocking anyway — there's no actual concurrency happening).
- The `/readiness` probe will always report ready (as long as the model is loaded), which is correct since K8s service-level load balancing with multiple replicas handles the "busy" case.

## Steps to Fix

1. Revert the capacity tracking logic from `inference_neuron_tp.py`
2. Keep the gradio files and other configurations untouched
