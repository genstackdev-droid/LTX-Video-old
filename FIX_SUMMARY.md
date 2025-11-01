# Fix Summary: LTXVFullPipeline Parameter Validation Errors

## Problem Statement

Users were experiencing validation errors when using LTXVFullPipeline node:

```
Failed to validate prompt for output 5:
* LTXVFullPipeline 2:
  - Value not in list: quality_mode: '80' not in ['Standard', 'Ultra']
  - Failed to convert an input value to a INT value: seed, blurry, low quality, distorted, watermark, text, invalid literal for int() with base 10: 'blurry, low quality, distorted, watermark, text'
  - Value 8 smaller than min of 20: steps
  - Value 42.0 bigger than max of 20.0: cfg_scale
```

## Root Cause

The `fps` parameter was added to the **required** section of INPUT_TYPES, but existing workflow JSON files were created before this parameter existed. This caused a parameter position mismatch:

### Before the Fix
- **Required parameters (9):** prompt, duration, resolution, prompt_mode, quality_mode, steps, cfg_scale, seed, **fps**
- **Old workflows provided (8):** prompt, duration, resolution, prompt_mode, quality_mode, steps, cfg_scale, seed
- **Missing:** fps parameter

### Parameter Shift Issue
When ComfyUI tried to map the 8 provided values to 9 expected required parameters:
- Position 0-7 mapped correctly (prompt through seed)
- Position 8 (fps) was missing, so ComfyUI shifted optional parameters into required positions
- Position 8 (expected fps) received negative_prompt value
- Position 9 (expected negative_prompt) received model_path value
- And so on...

This caused validation errors like:
- `quality_mode: '80'` (received steps value instead)
- `seed: 'blurry, low quality...'` (received negative_prompt value instead)

## Solution

Moved `fps` parameter from **required** to **optional** section with a default value of 25.

### Changes Made

#### 1. nodes.py
```python
# Before (fps in required)
"required": {
    ...
    "seed": ("INT", {...}),
    "fps": ("INT", {"default": 25, ...}),  # ❌ In required section
},
"optional": {
    "negative_prompt": (...),
    ...
}

# After (fps in optional)
"required": {
    ...
    "seed": ("INT", {...}),
},
"optional": {
    "negative_prompt": (...),
    "fps": ("INT", {"default": 25, ...}),  # ✅ In optional section
    ...
}
```

#### 2. Function Signature
```python
# Before
def generate_video(
    self, prompt, duration, resolution, prompt_mode, quality_mode,
    steps, cfg_scale, seed, fps,  # ❌ Required parameter
    negative_prompt="", model_path="...", sampler_name="..."
):

# After
def generate_video(
    self, prompt, duration, resolution, prompt_mode, quality_mode,
    steps, cfg_scale, seed,
    negative_prompt="", model_path="...", sampler_name="...",
    fps=25  # ✅ Optional with default
):
```

#### 3. Workflow Files
Updated all workflow JSON files to explicitly include fps=25:
- `ltx_simple_quickstart.json`
- `ltx_unified_production.json`
- `ltx_advanced_modular.json`

## Benefits

### ✅ Backward Compatibility
- Old workflows (11 widget values) work correctly with default fps=25
- No need to regenerate existing workflows

### ✅ Forward Compatibility
- New workflows can specify custom fps values (12-120)
- Existing functionality preserved

### ✅ Parameter Validation
- All parameters map to correct positions
- Validation errors resolved

## Testing

Created comprehensive tests in `tests/test_backward_compatibility.py`:
- Validates INPUT_TYPES structure
- Checks fps is in optional section with default value of 25
- Verifies workflow files have correct parameter counts
- Ensures parameter mapping is correct

All tests pass successfully:
```
✅ Parameter structure test passed!
✅ Workflow compatibility test passed!
✅ ALL TESTS PASSED!
```

## Migration Guide

### For Users with Existing Workflows
**No action required!** Old workflows will continue to work with fps defaulting to 25.

### For Users Creating New Workflows
New workflows should include fps in the widget values (position 11):
```json
"widgets_values": [
    "",                  // 0: prompt
    "10s",              // 1: duration
    "1080p",            // 2: resolution
    "Basic",            // 3: prompt_mode
    "Standard",         // 4: quality_mode
    120,                // 5: steps
    10.0,               // 6: cfg_scale
    -1,                 // 7: seed
    "blurry...",        // 8: negative_prompt
    "Lightricks/...",   // 9: model_path
    "DPM++...",         // 10: sampler_name
    25                  // 11: fps (NEW)
]
```

## Summary

The fix ensures backward compatibility by making the `fps` parameter optional, preventing parameter position mismatches that caused validation errors. All existing workflows continue to work, while new workflows can optionally specify custom fps values.
