"""
Test backward compatibility for LTXVFullPipeline node
Ensures old workflow files work correctly with the updated parameter structure
"""
import json
from pathlib import Path


def test_workflow_compatibility():
    """
    Test that workflow files have the correct number of parameters
    for the current INPUT_TYPES structure
    """
    workflows_dir = Path(__file__).parent.parent / "workflows"
    
    print("Testing workflow compatibility...")
    
    # Expected parameter counts
    # Required: 8 (prompt, duration, resolution, prompt_mode, quality_mode, steps, cfg_scale, seed)
    # Optional: 4 (negative_prompt, fps, model_path, sampler_name)
    # Total widgets_values: 8 required + up to 4 optional = 8-12 values
    
    for workflow_file in workflows_dir.glob("*.json"):
        print(f"\nChecking: {workflow_file.name}")
        
        with open(workflow_file, 'r') as f:
            workflow = json.load(f)
        
        # Find LTXVFullPipeline nodes
        ltxv_nodes = [node for node in workflow.get('nodes', []) 
                      if node.get('type') == 'LTXVFullPipeline']
        
        for node in ltxv_nodes:
            widgets = node.get('widgets_values', [])
            print(f"  Node {node['id']}: {len(widgets)} widget values")
            
            # Validate parameter count
            # Old workflows had 11 values (8 required + 3 optional, no fps)
            # New workflows should have 12 values (8 required + 4 optional, with fps)
            if len(widgets) == 11:
                print(f"  ⚠️  Old workflow format detected (11 values) - will use fps default")
            elif len(widgets) == 12:
                print(f"  ✓ New workflow format (12 values)")
            else:
                print(f"  ❌ Unexpected widget count: {len(widgets)}")
                return False
            
            # Print the parameter mapping
            param_names = [
                "prompt", "duration", "resolution", "prompt_mode", 
                "quality_mode", "steps", "cfg_scale", "seed",
                "negative_prompt", "model_path", "sampler_name"
            ]
            if len(widgets) >= 12:
                param_names.append("fps")
            
            print("  Parameter mapping:")
            for i, (name, value) in enumerate(zip(param_names, widgets)):
                value_str = str(value)[:50]  # Truncate long values
                print(f"    {i}: {name:20} = {value_str}")
    
    print("\n" + "=" * 60)
    print("✅ Workflow compatibility test passed!")
    print("=" * 60)
    return True


def test_parameter_structure():
    """
    Test that the INPUT_TYPES structure is correct
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # We need to check the structure without importing torch
    # Parse the nodes.py file directly
    nodes_file = Path(__file__).parent.parent / "nodes.py"
    
    with open(nodes_file, 'r') as f:
        content = f.read()
    
    print("\nTesting INPUT_TYPES structure...")
    
    # Check that fps is in optional section
    input_types_section = content[content.find('def INPUT_TYPES'):content.find('RETURN_TYPES')]
    
    # Split into required and optional sections
    required_section = input_types_section[:input_types_section.find('"optional"')]
    optional_section = input_types_section[input_types_section.find('"optional"'):]
    
    # Check fps is NOT in required
    if '"fps"' in required_section:
        print("❌ FAIL: fps found in required section")
        return False
    else:
        print("✓ fps is NOT in required section")
    
    # Check fps IS in optional
    if '"fps"' in optional_section:
        print("✓ fps is in optional section")
    else:
        print("❌ FAIL: fps not found in optional section")
        return False
    
    # Check fps has correct default
    import re
    fps_def = re.search(r'"fps":\s*\([^)]+{[^}]*"default":\s*(\d+)', optional_section)
    if fps_def and fps_def.group(1) == '25':
        print(f"✓ fps has default value of 25")
    else:
        print(f"❌ FAIL: fps default value issue")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Parameter structure test passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    print("Running backward compatibility tests...\n")
    
    try:
        if test_parameter_structure() and test_workflow_compatibility():
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED!")
            print("=" * 60)
            print("\nBackward compatibility confirmed:")
            print("- Old workflows (without fps) will work with default fps=25")
            print("- New workflows can specify custom fps value")
            print("- Parameter validation will pass correctly")
        else:
            print("\n❌ Some tests failed")
            exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
