import pytest
from validation import (
    validate_master_config,
    validate_matrix_output,
    validate_runner_config,
    Fields,
    SingleNodeMatrixEntry,
    MultiNodeMatrixEntry,
    WorkerConfig,
    SingleNodeSearchSpaceEntry,
    MultiNodeSearchSpaceEntry,
    SingleNodeSeqLenConfig,
    MultiNodeSeqLenConfig,
    SingleNodeMasterConfigEntry,
    MultiNodeMasterConfigEntry,
)


# ============================================================================
# Tests for validate_master_config - Single Node
# ============================================================================

class TestValidateMasterConfigSingleNode:
    """Tests for validate_master_config with single-node configurations."""

    def test_valid_single_node_config(self):
        """Test validation of a valid single-node config."""
        config = {
            "test-fp8-h200-vllm": {
                "image": "vllm/vllm-openai:v0.11.0",
                "model": "meta-llama/Llama-3-70b",
                "model-prefix": "llama70b",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 4, "conc-start": 4, "conc-end": 64},
                            {"tp": 8, "conc-start": 4, "conc-end": 64, "ep": 2, "dp-attn": True}
                        ]
                    }
                ]
            }
        }
        result = validate_master_config(config)
        assert result == config

    def test_valid_single_node_with_disagg(self):
        """Test validation with disagg field."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "disagg": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 4, "conc-start": 4, "conc-end": 64}
                        ]
                    }
                ]
            }
        }
        result = validate_master_config(config)
        assert result == config

    def test_valid_single_node_with_conc_list(self):
        """Test validation with conc-list instead of conc-start/end."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 4, "conc-list": [1, 2, 4, 8, 16]}
                        ]
                    }
                ]
            }
        }
        result = validate_master_config(config)
        assert result == config

    def test_missing_required_field_image(self):
        """Test validation fails when image is missing."""
        config = {
            "test-config": {
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_required_field_model(self):
        """Test validation fails when model is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_required_field_model_prefix(self):
        """Test validation fails when model-prefix is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_required_field_precision(self):
        """Test validation fails when precision is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_required_field_framework(self):
        """Test validation fails when framework is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_required_field_runner(self):
        """Test validation fails when runner is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_required_field_multinode(self):
        """Test validation fails when multinode is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_required_field_seq_len_configs(self):
        """Test validation fails when seq-len-configs is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_wrong_type_image(self):
        """Test validation fails when image has wrong type."""
        config = {
            "test-config": {
                "image": 123,
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_empty_seq_len_configs(self):
        """Test that empty seq-len-configs is allowed by validation.

        Note: Pydantic allows empty lists by default. This may produce
        no output at runtime but is not a validation error.
        """
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": []
            }
        }
        # This is allowed - Pydantic doesn't enforce non-empty lists by default
        result = validate_master_config(config)
        assert result == config

    def test_missing_isl_in_seq_len_config(self):
        """Test validation fails when isl is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_osl_in_seq_len_config(self):
        """Test validation fails when osl is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_search_space(self):
        """Test validation fails when search-space is missing."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_empty_search_space(self):
        """Test that empty search-space is allowed by validation.

        Note: Pydantic allows empty lists by default. This may produce
        no output at runtime but is not a validation error.
        """
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": []
                    }
                ]
            }
        }
        # This is allowed - Pydantic doesn't enforce non-empty lists by default
        result = validate_master_config(config)
        assert result == config

    def test_missing_tp_in_search_space(self):
        """Test validation fails when tp is missing in search-space."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_missing_both_conc_range_and_list(self):
        """Test validation fails when neither conc-start/end nor conc-list is provided."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_both_conc_range_and_list_provided(self):
        """Test validation fails when both conc-start/end and conc-list are provided."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 4, "conc-start": 4, "conc-end": 64, "conc-list": [1, 2, 4]}
                        ]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_conc_start_greater_than_end(self):
        """Test validation fails when conc-start > conc-end."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 64, "conc-end": 4}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_conc_list_with_zero_value(self):
        """Test validation fails when conc-list contains zero."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-list": [0, 1, 2]}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_wrong_type_tp(self):
        """Test validation fails when tp has wrong type."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": "four", "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_wrong_type_ep(self):
        """Test validation fails when ep has wrong type."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "ep": "two", "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_wrong_type_dp_attn(self):
        """Test validation fails when dp-attn has a truly invalid type.

        Note: Pydantic coerces some string values to bools (e.g., "yes" -> True).
        We test with a value that cannot be coerced.
        """
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "dp-attn": [1, 2, 3], "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_extra_field_in_top_level(self):
        """Test validation fails when extra field is present at top level."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "extra-field": "not-allowed",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-start": 4, "conc-end": 64}]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_extra_field_in_search_space(self):
        """Test validation fails when extra field is present in search-space."""
        config = {
            "test-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "runner": "h200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 4, "conc-start": 4, "conc-end": 64, "invalid-field": "value"}
                        ]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)


# ============================================================================
# Tests for validate_master_config - Multi Node
# ============================================================================

class TestValidateMasterConfigMultiNode:
    """Tests for validate_master_config with multinode configurations."""

    def test_valid_multinode_config(self):
        """Test validation of a valid multinode config."""
        config = {
            "test-multinode": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "disagg": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "spec-decoding": "mtp",
                                "conc-list": [1, 2, 4, 8],
                                "prefill": {
                                    "num-worker": 1,
                                    "tp": 4,
                                    "ep": 4,
                                    "dp-attn": False,
                                    "additional-settings": ["PREFILL_MAX_NUM_TOKENS=4608"]
                                },
                                "decode": {
                                    "num-worker": 4,
                                    "tp": 8,
                                    "ep": 8,
                                    "dp-attn": False,
                                    "additional-settings": ["DECODE_MAX_NUM_TOKENS=128"]
                                }
                            }
                        ]
                    }
                ]
            }
        }
        result = validate_master_config(config)
        assert result == config

    def test_valid_multinode_with_conc_range(self):
        """Test validation of multinode config with conc-start/conc-end."""
        config = {
            "test-multinode": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "conc-start": 4,
                                "conc-end": 64,
                                "prefill": {
                                    "num-worker": 1,
                                    "tp": 4,
                                    "ep": 4,
                                    "dp-attn": False
                                },
                                "decode": {
                                    "num-worker": 4,
                                    "tp": 8,
                                    "ep": 8,
                                    "dp-attn": False
                                }
                            }
                        ]
                    }
                ]
            }
        }
        result = validate_master_config(config)
        assert result == config

    def test_multinode_missing_prefill(self):
        """Test validation fails when prefill is missing in multinode config."""
        config = {
            "test-multinode": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "conc-list": [1, 2, 4],
                                "decode": {
                                    "num-worker": 4,
                                    "tp": 8,
                                    "ep": 8,
                                    "dp-attn": False
                                }
                            }
                        ]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_multinode_missing_decode(self):
        """Test validation fails when decode is missing in multinode config."""
        config = {
            "test-multinode": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "conc-list": [1, 2, 4],
                                "prefill": {
                                    "num-worker": 1,
                                    "tp": 4,
                                    "ep": 4,
                                    "dp-attn": False
                                }
                            }
                        ]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_multinode_invalid_spec_decoding(self):
        """Test validation fails when spec-decoding has invalid value."""
        config = {
            "test-multinode": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "spec-decoding": "invalid-value",
                                "conc-list": [1, 2, 4],
                                "prefill": {
                                    "num-worker": 1,
                                    "tp": 4,
                                    "ep": 4,
                                    "dp-attn": False
                                },
                                "decode": {
                                    "num-worker": 4,
                                    "tp": 8,
                                    "ep": 8,
                                    "dp-attn": False
                                }
                            }
                        ]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_multinode_worker_config_missing_num_worker(self):
        """Test validation fails when num-worker is missing in worker config."""
        config = {
            "test-multinode": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "conc-list": [1, 2, 4],
                                "prefill": {
                                    "tp": 4,
                                    "ep": 4,
                                    "dp-attn": False
                                },
                                "decode": {
                                    "num-worker": 4,
                                    "tp": 8,
                                    "ep": 8,
                                    "dp-attn": False
                                }
                            }
                        ]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)

    def test_multinode_worker_config_missing_tp(self):
        """Test validation fails when tp is missing in worker config."""
        config = {
            "test-multinode": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "conc-list": [1, 2, 4],
                                "prefill": {
                                    "num-worker": 1,
                                    "ep": 4,
                                    "dp-attn": False
                                },
                                "decode": {
                                    "num-worker": 4,
                                    "tp": 8,
                                    "ep": 8,
                                    "dp-attn": False
                                }
                            }
                        ]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_master_config(config)


# ============================================================================
# Tests for validate_matrix_output - Single Node
# ============================================================================

class TestValidateMatrixOutputSingleNode:
    """Tests for validate_matrix_output with single-node entries."""

    def test_valid_single_node_entry(self):
        """Test validation of a valid single-node matrix entry."""
        entry = {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "isl": 1024,
            "osl": 1024,
            "tp": 8,
            "ep": 1,
            "dp-attn": False,
            "conc": 4,
            "max-model-len": 2248,
            "exp-name": "test_1k1k",
            "disagg": False
        }
        result = validate_matrix_output(entry, is_multinode=False)
        assert result == entry

    def test_single_node_missing_field(self):
        """Test validation fails when required field is missing."""
        entry = {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "isl": 1024,
            "osl": 1024,
            # Missing tp
            "ep": 1,
            "dp-attn": False,
            "conc": 4,
            "max-model-len": 2248,
            "exp-name": "test_1k1k",
            "disagg": False
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_matrix_output(entry, is_multinode=False)

    def test_single_node_wrong_type(self):
        """Test validation fails when field has wrong type."""
        entry = {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "isl": "not-an-int",
            "osl": 1024,
            "tp": 8,
            "ep": 1,
            "dp-attn": False,
            "conc": 4,
            "max-model-len": 2248,
            "exp-name": "test_1k1k",
            "disagg": False
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_matrix_output(entry, is_multinode=False)

    def test_single_node_extra_field(self):
        """Test validation fails when extra field is present."""
        entry = {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "isl": 1024,
            "osl": 1024,
            "tp": 8,
            "ep": 1,
            "dp-attn": False,
            "conc": 4,
            "max-model-len": 2248,
            "exp-name": "test_1k1k",
            "disagg": False,
            "extra-field": "not-allowed"
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_matrix_output(entry, is_multinode=False)


# ============================================================================
# Tests for validate_matrix_output - Multi Node
# ============================================================================

class TestValidateMatrixOutputMultiNode:
    """Tests for validate_matrix_output with multinode entries."""

    def test_valid_multinode_entry(self):
        """Test validation of a valid multinode matrix entry."""
        entry = {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp4",
            "framework": "dynamo-trt",
            "spec-decoding": "mtp",
            "runner": "gb200",
            "isl": 1024,
            "osl": 1024,
            "prefill": {
                "num-worker": 1,
                "tp": 4,
                "ep": 4,
                "dp-attn": False
            },
            "decode": {
                "num-worker": 4,
                "tp": 8,
                "ep": 8,
                "dp-attn": False
            },
            "conc": [1, 2, 4, 8],
            "max-model-len": 2248,
            "exp-name": "test_1k1k",
            "disagg": True
        }
        result = validate_matrix_output(entry, is_multinode=True)
        assert result == entry

    def test_multinode_missing_spec_decoding(self):
        """Test validation fails when spec-decoding is missing."""
        entry = {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp4",
            "framework": "dynamo-trt",
            "runner": "gb200",
            "isl": 1024,
            "osl": 1024,
            "prefill": {
                "num-worker": 1,
                "tp": 4,
                "ep": 4,
                "dp-attn": False
            },
            "decode": {
                "num-worker": 4,
                "tp": 8,
                "ep": 8,
                "dp-attn": False
            },
            "conc": [1, 2, 4, 8],
            "max-model-len": 2248,
            "exp-name": "test_1k1k",
            "disagg": True
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_matrix_output(entry, is_multinode=True)

    def test_multinode_conc_not_list(self):
        """Test validation fails when conc is not a list in multinode entry."""
        entry = {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp4",
            "framework": "dynamo-trt",
            "spec-decoding": "mtp",
            "runner": "gb200",
            "isl": 1024,
            "osl": 1024,
            "prefill": {
                "num-worker": 1,
                "tp": 4,
                "ep": 4,
                "dp-attn": False
            },
            "decode": {
                "num-worker": 4,
                "tp": 8,
                "ep": 8,
                "dp-attn": False
            },
            "conc": 4,  # Should be a list
            "max-model-len": 2248,
            "exp-name": "test_1k1k",
            "disagg": True
        }
        with pytest.raises(ValueError, match="failed validation"):
            validate_matrix_output(entry, is_multinode=True)


# ============================================================================
# Tests for validate_runner_config
# ============================================================================

class TestValidateRunnerConfig:
    """Tests for validate_runner_config function."""

    def test_valid_runner_config(self):
        """Test validation of a valid runner config."""
        config = {
            "h200": ["h200-nv_1", "h200-nv_2"],
            "b200": ["b200-nv_1"],
            "gb200": ["gb200-nv_1", "gb200-nv_2", "gb200-nv_3"]
        }
        result = validate_runner_config(config)
        assert result == config

    def test_runner_config_value_not_list(self):
        """Test validation fails when runner value is not a list."""
        config = {
            "h200": "h200-nv_1"  # Should be a list
        }
        with pytest.raises(ValueError, match="must be a list"):
            validate_runner_config(config)

    def test_runner_config_list_not_strings(self):
        """Test validation fails when list contains non-strings."""
        config = {
            "h200": ["h200-nv_1", 123]  # Contains non-string
        }
        with pytest.raises(ValueError, match="must contain only strings"):
            validate_runner_config(config)

    def test_runner_config_empty_list(self):
        """Test validation fails when runner list is empty."""
        config = {
            "h200": []  # Empty list
        }
        with pytest.raises(ValueError, match="cannot be an empty list"):
            validate_runner_config(config)


# ============================================================================
# Tests for Pydantic Models - Unit Tests
# ============================================================================

class TestWorkerConfigModel:
    """Tests for WorkerConfig Pydantic model."""

    def test_valid_worker_config(self):
        """Test valid WorkerConfig."""
        config = WorkerConfig(**{
            "num-worker": 4,
            "tp": 8,
            "ep": 8,
            "dp-attn": False,
            "additional-settings": ["SETTING1=value1"]
        })
        assert config.num_worker == 4
        assert config.tp == 8

    def test_worker_config_without_additional_settings(self):
        """Test WorkerConfig without additional-settings."""
        config = WorkerConfig(**{
            "num-worker": 4,
            "tp": 8,
            "ep": 8,
            "dp-attn": False
        })
        assert config.additional_settings is None


class TestSingleNodeSearchSpaceEntry:
    """Tests for SingleNodeSearchSpaceEntry Pydantic model."""

    def test_valid_with_range(self):
        """Test valid entry with conc-start/conc-end."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 4,
            "conc-start": 4,
            "conc-end": 64
        })
        assert entry.tp == 4
        assert entry.conc_start == 4
        assert entry.conc_end == 64

    def test_valid_with_list(self):
        """Test valid entry with conc-list."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 4,
            "conc-list": [1, 2, 4, 8]
        })
        assert entry.tp == 4
        assert entry.conc_list == [1, 2, 4, 8]

    def test_valid_with_optional_fields(self):
        """Test valid entry with optional fields."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 4,
            "ep": 2,
            "dp-attn": True,
            "spec-decoding": "mtp",
            "conc-start": 4,
            "conc-end": 64
        })
        assert entry.ep == 2
        assert entry.dp_attn == True
        assert entry.spec_decoding == "mtp"


class TestMultiNodeSearchSpaceEntry:
    """Tests for MultiNodeSearchSpaceEntry Pydantic model."""

    def test_valid_multinode_entry(self):
        """Test valid multinode search-space entry."""
        entry = MultiNodeSearchSpaceEntry(**{
            "conc-list": [1, 2, 4],
            "prefill": {
                "num-worker": 1,
                "tp": 4,
                "ep": 4,
                "dp-attn": False
            },
            "decode": {
                "num-worker": 4,
                "tp": 8,
                "ep": 8,
                "dp-attn": False
            }
        })
        assert entry.prefill.num_worker == 1
        assert entry.decode.num_worker == 4


# ============================================================================
# Tests for Fields Enum
# ============================================================================

class TestFieldsEnum:
    """Tests for the Fields enum."""

    def test_field_values(self):
        """Test that Fields enum has expected values."""
        assert Fields.IMAGE.value == 'image'
        assert Fields.MODEL.value == 'model'
        assert Fields.MODEL_PREFIX.value == 'model-prefix'
        assert Fields.PRECISION.value == 'precision'
        assert Fields.FRAMEWORK.value == 'framework'
        assert Fields.RUNNER.value == 'runner'
        assert Fields.SEQ_LEN_CONFIGS.value == 'seq-len-configs'
        assert Fields.MULTINODE.value == 'multinode'
        assert Fields.ISL.value == 'isl'
        assert Fields.OSL.value == 'osl'
        assert Fields.SEARCH_SPACE.value == 'search-space'
        assert Fields.TP.value == 'tp'
        assert Fields.EP.value == 'ep'
        assert Fields.CONC_START.value == 'conc-start'
        assert Fields.CONC_END.value == 'conc-end'
        assert Fields.CONC_LIST.value == 'conc-list'
        assert Fields.DP_ATTN.value == 'dp-attn'
        assert Fields.DISAGG.value == 'disagg'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
