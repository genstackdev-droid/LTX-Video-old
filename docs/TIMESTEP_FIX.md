# Timestep Validation Fix

## Problem

The LTX-Video pipeline was failing with the following error:

```
AssertionError: Invalid inference timestep 0.9994. Allowed timesteps are [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219].
```

This error occurred when using the `LTXVFullPipeline` node in ComfyUI.

## Root Cause

The LTX-Video checkpoint (`ltxv-13b-0.9.8-distilled.safetensors`) includes metadata that defines a strict set of allowed inference timesteps. These timesteps are optimized for the multi-scale pipeline architecture.

When the pipeline was called with `num_inference_steps` (e.g., 80 steps), the scheduler would generate its own timesteps using a uniform or linear-quadratic distribution. These generated timesteps did not match the checkpoint's allowed timesteps, causing the validation to fail.

### Why This Happens

1. The checkpoint has `allowed_inference_steps` metadata: `[1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219]`
2. The pipeline validates timesteps against this list (line 952-956 in `pipeline_ltx_video.py`)
3. When `num_inference_steps=80` is passed, the scheduler generates 80 evenly-spaced timesteps
4. These generated timesteps (e.g., 0.9994) don't match the allowed list
5. The assertion fails

## Solution

The fix involves two changes:

### 1. Use Explicit Timesteps

Instead of passing `num_inference_steps` to the pipeline, we now check if the pipeline has `allowed_inference_steps` and use those explicitly:

```python
if hasattr(self.pipeline, 'allowed_inference_steps') and self.pipeline.allowed_inference_steps is not None:
    # Use checkpoint's predefined timesteps
    output = self.pipeline(
        timesteps=self.pipeline.allowed_inference_steps,
        # ... other parameters
    )
else:
    # Fallback to num_inference_steps for flexible checkpoints
    output = self.pipeline(
        num_inference_steps=optimized_steps,
        # ... other parameters
    )
```

### 2. Load Allowed Timesteps from Checkpoint

The `_load_pipeline` method already loads `allowed_inference_steps` from the checkpoint metadata:

```python
with safe_open(ckpt_path, framework="pt") as f:
    metadata = f.metadata()
    config_str = metadata.get("config")
    if config_str:
        configs = json.loads(config_str)
        allowed_inference_steps = configs.get("allowed_inference_steps", None)
```

This value is then passed to the pipeline during initialization.

## Impact

### Before Fix
- ❌ Pipeline would fail with timestep validation error
- ❌ Could not generate videos with the distilled checkpoint
- ❌ User had no control over timesteps

### After Fix
- ✅ Pipeline uses checkpoint's optimized timesteps
- ✅ Works seamlessly with distilled checkpoints
- ✅ Falls back to flexible timesteps for other checkpoints
- ✅ User sees which mode is being used in console logs

## Console Output

With the fix, users will see informative messages:

```
[LTX-Video] Using checkpoint timesteps: [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219]
```

Or for checkpoints without predefined timesteps:

```
[LTX-Video] Using num_inference_steps: 80
```

## Technical Details

### Checkpoint Metadata

The distilled checkpoint includes this configuration:

```yaml
allowed_inference_steps:
  - 1.0
  - 0.9937
  - 0.9875
  - 0.9812
  - 0.975
  - 0.9094
  - 0.725
  - 0.4219
```

These 8 timesteps are carefully chosen for:
1. **First pass**: High-quality coarse generation (6 timesteps)
2. **Second pass**: Detail refinement (3 timesteps with 1 overlap)
3. **Multi-scale rendering**: Optimal for the spatial upscaler

### Why Predefined Timesteps?

The multi-scale pipeline architecture benefits from:
- **Non-uniform spacing**: More timesteps at critical phases
- **Two-pass strategy**: Separate coarse and fine generation
- **Computational efficiency**: Fewer steps while maintaining quality
- **Consistency**: Reproducible results across different runs

### Backward Compatibility

The fix maintains backward compatibility:
- Checkpoints **with** `allowed_inference_steps`: Uses predefined timesteps
- Checkpoints **without** `allowed_inference_steps`: Uses `num_inference_steps`
- Both modes work seamlessly

## Related Changes

As part of this fix, we also:

1. Added adjustable FPS parameter (12-120 FPS)
2. Improved frame count calculations based on FPS
3. Enhanced console logging for better debugging
4. Created comprehensive FPS documentation

## Testing

To verify the fix works:

1. Load the workflow with `LTXVFullPipeline` node
2. Check console for: `[LTX-Video] Using checkpoint timesteps: [...]`
3. Verify video generates without errors
4. Test with different FPS settings (12-120)

## References

- `nodes.py`: Lines 277-304 (timestep selection logic)
- `ltx_video/pipelines/pipeline_ltx_video.py`: Lines 952-956 (validation)
- `configs/ltxv-13b-0.9.8-distilled.yaml`: Pipeline configuration
- `docs/FPS_GUIDE.md`: FPS feature documentation
