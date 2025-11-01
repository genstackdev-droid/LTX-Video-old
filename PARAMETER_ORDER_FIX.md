# Parameter Order Fix Documentation

## Issue Summary

**Error Message:**
```
Failed to validate prompt for output 3:
* LTXVFullPipeline 2:
  - Failed to convert an input value to a INT value: fps, DPM++ 3M SDE Karras, 
    invalid literal for int() with base 10: 'DPM++ 3M SDE Karras'
```

## Root Cause

ComfyUI serializes and deserializes workflow parameters in the exact order they appear in the `INPUT_TYPES` dictionary. When the parameter order in `INPUT_TYPES` doesn't match the order in workflow files, ComfyUI incorrectly maps values to parameters.

### Before Fix (INCORRECT)

**INPUT_TYPES Definition:**
```python
"optional": {
    "negative_prompt": ...,  # Position 8
    "fps": ...,              # Position 9  <-- INT expected here
    "model_path": ...,       # Position 10
    "sampler_name": ...,     # Position 11 <-- STRING value here
}
```

**Workflow Serialization:**
```json
"widgets_values": [
    ...,                                    // Positions 0-7 (required params)
    "blurry, low quality, distorted",      // Position 8: negative_prompt
    "Lightricks/LTX-Video",                // Position 9: model_path
    "DPM++ 3M SDE Karras",                 // Position 10: sampler_name
    25                                      // Position 11: fps
]
```

**Result:** ComfyUI tried to pass `"DPM++ 3M SDE Karras"` (position 10) to `fps` (position 9), causing the type conversion error.

### After Fix (CORRECT)

**INPUT_TYPES Definition:**
```python
"optional": {
    "negative_prompt": ...,  # Position 8
    "model_path": ...,       # Position 9  <-- STRING
    "sampler_name": ...,     # Position 10 <-- STRING
    "fps": ...,              # Position 11 <-- INT
}
```

**Workflow Serialization:** (unchanged)
```json
"widgets_values": [
    ...,                                    // Positions 0-7 (required params)
    "blurry, low quality, distorted",      // Position 8: negative_prompt ✓
    "Lightricks/LTX-Video",                // Position 9: model_path ✓
    "DPM++ 3M SDE Karras",                 // Position 10: sampler_name ✓
    25                                      // Position 11: fps ✓
]
```

**Result:** All parameters now map correctly, and type validation passes.

## Changes Made

### 1. Updated `nodes.py` (Line 69-83)

**Changed:**
```python
"optional": {
    "negative_prompt": (...),
    "fps": ("INT", {"default": 25, "min": 12, "max": 120, "step": 1}),
    "model_path": ("STRING", {"default": "Lightricks/LTX-Video"}),
    "sampler_name": ([...], {"default": "DPM++ 3M SDE Karras"}),
}
```

**To:**
```python
"optional": {
    "negative_prompt": (...),
    "model_path": ("STRING", {"default": "Lightricks/LTX-Video"}),
    "sampler_name": ([...], {"default": "DPM++ 3M SDE Karras"}),
    "fps": ("INT", {"default": 25, "min": 12, "max": 120, "step": 1}),
}
```

### 2. Updated Tests

Added validation to `tests/test_nodes_pipeline_loading.py` to verify parameter order:

```python
# Verify parameter order (critical for ComfyUI workflow compatibility)
optional_keys = list(input_types["optional"].keys())
expected_order = ["negative_prompt", "model_path", "sampler_name", "fps"]
assert optional_keys == expected_order
```

## Affected Workflows

All workflow files were already using the correct order and required no changes:
- ✅ `workflows/ltx_simple_quickstart.json`
- ✅ `workflows/ltx_unified_production.json`
- ✅ `workflows/ltx_advanced_modular.json`

## Validation

Run the validation test:
```bash
PYTHONPATH=/path/to/repo python3 tests/test_nodes_pipeline_loading.py
```

Expected output:
```
✓ Node input types are correctly defined
✓ Optional parameter order is correct for workflow compatibility
```

## ComfyUI Parameter Mapping Reference

### Complete Parameter Order

| Position | Parameter Name    | Type   | Category |
|----------|-------------------|--------|----------|
| 0        | prompt            | STRING | Required |
| 1        | duration          | COMBO  | Required |
| 2        | resolution        | COMBO  | Required |
| 3        | prompt_mode       | COMBO  | Required |
| 4        | quality_mode      | COMBO  | Required |
| 5        | steps             | INT    | Required |
| 6        | cfg_scale         | FLOAT  | Required |
| 7        | seed              | INT    | Required |
| 8        | negative_prompt   | STRING | Optional |
| 9        | model_path        | STRING | Optional |
| 10       | sampler_name      | COMBO  | Optional |
| 11       | fps               | INT    | Optional |

### Key Principle

**The order of parameters in `INPUT_TYPES` must exactly match the order in which ComfyUI serializes them to workflow JSON files.**

Python dictionaries maintain insertion order (Python 3.7+), so the order you define parameters in `INPUT_TYPES` is the order ComfyUI will use.

## Best Practices

1. **Never reorder parameters** in an existing node's `INPUT_TYPES` without updating all workflow files
2. **Always append new optional parameters** to the end to maintain backward compatibility
3. **Test with actual workflows** after modifying `INPUT_TYPES`
4. **Add validation tests** for parameter order to catch regressions

## Related Issues

- ComfyUI parameter validation
- Workflow compatibility
- Type conversion errors in custom nodes

## Resolution Status

✅ **FIXED** - The parameter order in `INPUT_TYPES` now matches all workflow files.

---

**Date Fixed:** November 2024  
**Verified:** All tests passing
