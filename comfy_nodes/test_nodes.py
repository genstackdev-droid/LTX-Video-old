"""
Simple validation tests for LTX-Video ComfyUI nodes
Tests node structure, parameters, and basic functionality
"""

import sys


def test_node_imports():
    """Test that all nodes can be imported"""
    try:
        from comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        print("✓ Nodes import successfully")

        expected_nodes = [
            "LTXVFullPipeline",
            "LTXVSampler",
            "LTXVUpscaler",
            "LTXVFrameInterpolator",
            "LTXVPromptEnhancer",
        ]

        for node_name in expected_nodes:
            assert (
                node_name in NODE_CLASS_MAPPINGS
            ), f"Missing node: {node_name}"
            assert (
                node_name in NODE_DISPLAY_NAME_MAPPINGS
            ), f"Missing display name for: {node_name}"
            print(f"  ✓ {node_name}: {NODE_DISPLAY_NAME_MAPPINGS[node_name]}")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_node_structure():
    """Test that nodes have required methods and attributes"""
    from comfy_nodes import NODE_CLASS_MAPPINGS

    print("\n✓ Testing node structure...")

    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        # Check INPUT_TYPES
        assert hasattr(
            node_class, "INPUT_TYPES"
        ), f"{node_name} missing INPUT_TYPES"
        assert callable(
            node_class.INPUT_TYPES
        ), f"{node_name}.INPUT_TYPES not callable"

        # Check RETURN_TYPES
        assert hasattr(
            node_class, "RETURN_TYPES"
        ), f"{node_name} missing RETURN_TYPES"

        # Check FUNCTION
        assert hasattr(node_class, "FUNCTION"), f"{node_name} missing FUNCTION"

        # Check CATEGORY
        assert hasattr(node_class, "CATEGORY"), f"{node_name} missing CATEGORY"
        assert "LTX-Video" in node_class.CATEGORY, f"{node_name} wrong category"

        print(f"  ✓ {node_name} structure valid")

    return True


def test_full_pipeline_node():
    """Test LTXVFullPipeline node specifically"""
    from comfy_nodes.nodes import LTXVFullPipeline

    print("\n✓ Testing LTXVFullPipeline node...")

    # Get input types
    input_types = LTXVFullPipeline.INPUT_TYPES()

    # Check required inputs
    required = input_types["required"]
    assert "prompt" in required, "Missing prompt input"
    assert "duration" in required, "Missing duration input"
    assert "resolution" in required, "Missing resolution input"
    assert "steps" in required, "Missing steps input"
    assert "cfg_scale" in required, "Missing cfg_scale input"
    assert "seed" in required, "Missing seed input"

    print("  ✓ Required inputs present")

    # Check optional inputs
    optional = input_types.get("optional", {})
    assert "negative_prompt" in optional, "Missing negative_prompt"
    assert "model_path" in optional, "Missing model_path"

    print("  ✓ Optional inputs present")

    # Check return types
    assert LTXVFullPipeline.RETURN_TYPES == (
        "IMAGE",
        "INT",
        "INT",
        "INT",
    ), "Wrong return types"
    print("  ✓ Return types correct")

    # Check function exists
    node = LTXVFullPipeline()
    assert hasattr(node, "generate_video"), "Missing generate_video method"
    print("  ✓ Generate method exists")

    # Test prompt enhancement
    assert hasattr(node, "_enhance_prompt"), "Missing _enhance_prompt method"
    enhanced = node._enhance_prompt("a cat walking", "Basic")
    assert "photorealistic" in enhanced.lower(), "Enhancement not working"
    print("  ✓ Prompt enhancement working")

    # Test resolution params
    assert hasattr(node, "_get_resolution_params"), "Missing resolution method"
    width, height = node._get_resolution_params("1080p")
    assert width == 1920 and height == 1080, "Wrong 1080p dimensions"
    print("  ✓ Resolution params correct")

    # Test frame count
    assert hasattr(node, "_get_frame_count"), "Missing frame count method"
    frames = node._get_frame_count("8s")
    assert frames == 200, "Wrong frame count for 8s"
    print("  ✓ Frame count calculation correct")

    return True


def test_prompt_enhancer_node():
    """Test LTXVPromptEnhancer node specifically"""
    from comfy_nodes.nodes import LTXVPromptEnhancer

    print("\n✓ Testing LTXVPromptEnhancer node...")

    # Get input types
    input_types = LTXVPromptEnhancer.INPUT_TYPES()
    required = input_types["required"]

    # Check inputs
    assert "prompt" in required, "Missing prompt input"
    assert "enhancement_level" in required, "Missing enhancement_level"
    assert "style" in required, "Missing style"

    print("  ✓ Inputs present")

    # Check return types
    assert LTXVPromptEnhancer.RETURN_TYPES == (
        "STRING",
    ), "Wrong return types"
    print("  ✓ Return types correct")

    # Test enhancement
    node = LTXVPromptEnhancer()
    enhanced, = node.enhance("a cat", "Maximum", "Realistic")
    assert len(enhanced) > len("a cat"), "Enhancement didn't add text"
    assert "photorealistic" in enhanced.lower(), "Missing quality keywords"
    print("  ✓ Enhancement working")

    return True


def test_upscaler_node():
    """Test LTXVUpscaler node"""
    from comfy_nodes.nodes import LTXVUpscaler

    print("\n✓ Testing LTXVUpscaler node...")

    input_types = LTXVUpscaler.INPUT_TYPES()
    required = input_types["required"]

    assert "frames" in required, "Missing frames input"
    assert "upscale_method" in required, "Missing upscale_method"
    assert "scale_factor" in required, "Missing scale_factor"
    assert "tile_size" in required, "Missing tile_size"

    print("  ✓ Node structure valid")
    return True


def test_interpolator_node():
    """Test LTXVFrameInterpolator node"""
    from comfy_nodes.nodes import LTXVFrameInterpolator

    print("\n✓ Testing LTXVFrameInterpolator node...")

    input_types = LTXVFrameInterpolator.INPUT_TYPES()
    required = input_types["required"]

    assert "frames" in required, "Missing frames input"
    assert "target_fps" in required, "Missing target_fps"
    assert "interpolation_mode" in required, "Missing interpolation_mode"

    print("  ✓ Node structure valid")
    return True


def test_sampler_node():
    """Test LTXVSampler node"""
    from comfy_nodes.nodes import LTXVSampler

    print("\n✓ Testing LTXVSampler node...")

    input_types = LTXVSampler.INPUT_TYPES()
    required = input_types["required"]

    assert "model" in required, "Missing model input"
    assert "steps" in required, "Missing steps"
    assert "cfg" in required, "Missing cfg"
    assert "sampler_name" in required, "Missing sampler_name"

    print("  ✓ Node structure valid")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("=" * 60)
    print("LTX-Video ComfyUI Nodes - Validation Tests")
    print("=" * 60)

    tests = [
        ("Import Test", test_node_imports),
        ("Structure Test", test_node_structure),
        ("Full Pipeline Test", test_full_pipeline_node),
        ("Prompt Enhancer Test", test_prompt_enhancer_node),
        ("Upscaler Test", test_upscaler_node),
        ("Interpolator Test", test_interpolator_node),
        ("Sampler Test", test_sampler_node),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
