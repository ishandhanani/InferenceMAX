import pytest
import yaml
from unittest.mock import patch
from generate_sweep_configs import (
    seq_len_to_str,
    generate_full_sweep,
    generate_runner_model_sweep_config,
    load_config_files,
    load_runner_file,
    main,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_single_node_config():
    """Sample master config with single-node entries."""
    return {
        "dsr1-fp8-h200-sglang": {
            "image": "lmsysorg/sglang:v0.5.5-cu129-amd64",
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "model-prefix": "dsr1",
            "runner": "h200",
            "precision": "fp8",
            "framework": "sglang",
            "multinode": False,
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 4, "conc-start": 4, "conc-end": 64},
                        {"tp": 8, "conc-start": 4, "conc-end": 64, "ep": 2, "dp-attn": True}
                    ]
                },
                {
                    "isl": 1024,
                    "osl": 8192,
                    "search-space": [
                        {"tp": 8, "conc-start": 4, "conc-end": 32}
                    ]
                },
                {
                    "isl": 8192,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 8, "conc-start": 4, "conc-end": 16}
                    ]
                }
            ]
        },
        "gptoss-fp4-b200-vllm": {
            "image": "vllm/vllm-openai:v0.11.0",
            "model": "openai/gpt-oss-120b",
            "model-prefix": "gptoss",
            "runner": "b200",
            "precision": "fp4",
            "framework": "vllm",
            "multinode": False,
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 1, "conc-start": 4, "conc-end": 128},
                        {"tp": 2, "conc-start": 4, "conc-end": 128},
                        {"tp": 4, "conc-start": 4, "conc-end": 64},
                        {"tp": 8, "conc-start": 4, "conc-end": 8}
                    ]
                }
            ]
        }
    }


@pytest.fixture
def sample_multinode_config():
    """Sample master config with multinode entries."""
    return {
        "dsr1-fp4-gb200-dynamo-trt": {
            "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3",
            "model": "deepseek-r1-fp4",
            "model-prefix": "dsr1",
            "runner": "gb200",
            "precision": "fp4",
            "framework": "dynamo-trt",
            "multinode": True,
            "disagg": True,
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {
                            "spec-decoding": "mtp",
                            "conc-list": [1, 2, 4, 8, 16, 36],
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
                        },
                        {
                            "conc-list": [64, 128],
                            "prefill": {
                                "num-worker": 1,
                                "tp": 4,
                                "ep": 4,
                                "dp-attn": True,
                                "additional-settings": []
                            },
                            "decode": {
                                "num-worker": 1,
                                "tp": 16,
                                "ep": 16,
                                "dp-attn": True,
                                "additional-settings": []
                            }
                        }
                    ]
                }
            ]
        }
    }


@pytest.fixture
def sample_runner_config():
    """Sample runner config."""
    return {
        "h200": ["h200-nv_1", "h200-nv_2"],
        "b200": ["b200-nv_1"],
        "gb200": ["gb200-nv_1", "gb200-nv_2", "gb200-nv_3"],
        "h100": ["h100-aws_1"]
    }


@pytest.fixture
def temp_config_files(tmp_path, sample_single_node_config, sample_runner_config):
    """Create temporary config files for single-node tests."""
    master_file = tmp_path / "master.yaml"
    runner_file = tmp_path / "runners.yaml"

    with open(master_file, 'w') as f:
        yaml.dump(sample_single_node_config, f)

    with open(runner_file, 'w') as f:
        yaml.dump(sample_runner_config, f)

    return str(master_file), str(runner_file)


@pytest.fixture
def temp_multinode_config_files(tmp_path, sample_multinode_config, sample_runner_config):
    """Create temporary config files for multinode tests."""
    master_file = tmp_path / "master.yaml"
    runner_file = tmp_path / "runners.yaml"

    with open(master_file, 'w') as f:
        yaml.dump(sample_multinode_config, f)

    with open(runner_file, 'w') as f:
        yaml.dump(sample_runner_config, f)

    return str(master_file), str(runner_file)


# ============================================================================
# Helper class for mocking args
# ============================================================================

class MockArgs:
    """Mock args object for testing functions."""
    def __init__(self, **kwargs):
        # Defaults
        self.model_prefix = None
        self.precision = None
        self.framework = None
        self.runner_type = None
        self.seq_lens = None
        self.step_size = 2
        self.max_conc = None
        self.max_tp = None
        self.max_ep = None
        self.single_node = False
        self.multi_node = False
        self.runner_config = None
        self.runner_node_filter = None

        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


# ============================================================================
# Tests for seq_len_to_str
# ============================================================================

class TestSeqLenToStr:
    """Tests for the seq_len_to_str function."""

    def test_known_mapping_1k1k(self):
        assert seq_len_to_str(1024, 1024) == "1k1k"

    def test_known_mapping_1k8k(self):
        assert seq_len_to_str(1024, 8192) == "1k8k"

    def test_known_mapping_8k1k(self):
        assert seq_len_to_str(8192, 1024) == "8k1k"

    def test_unknown_mapping_fallback(self):
        assert seq_len_to_str(2048, 4096) == "2048_4096"

    def test_unknown_mapping_small_values(self):
        assert seq_len_to_str(512, 512) == "512_512"


# ============================================================================
# Tests for load_config_files
# ============================================================================

class TestLoadConfigFiles:
    """Tests for the load_config_files function."""

    def test_load_single_valid_file(self, temp_config_files):
        master_file, _ = temp_config_files
        result = load_config_files([master_file])
        assert len(result) == 2
        assert "dsr1-fp8-h200-sglang" in result
        assert "gptoss-fp4-b200-vllm" in result

    def test_load_multiple_files(self, tmp_path, sample_single_node_config):
        file1 = tmp_path / "config1.yaml"
        file2 = tmp_path / "config2.yaml"

        config1 = {"dsr1-fp8-h200-sglang": sample_single_node_config["dsr1-fp8-h200-sglang"]}
        config2 = {"gptoss-fp4-b200-vllm": sample_single_node_config["gptoss-fp4-b200-vllm"]}

        with open(file1, 'w') as f:
            yaml.dump(config1, f)
        with open(file2, 'w') as f:
            yaml.dump(config2, f)

        result = load_config_files([str(file1), str(file2)])
        assert len(result) == 2

    def test_load_nonexistent_file(self):
        with pytest.raises(ValueError, match="does not exist"):
            load_config_files(["/nonexistent/file.yaml"])

    def test_load_files_with_duplicate_keys(self, tmp_path, sample_single_node_config):
        file1 = tmp_path / "config1.yaml"
        file2 = tmp_path / "config2.yaml"

        config = {"dsr1-fp8-h200-sglang": sample_single_node_config["dsr1-fp8-h200-sglang"]}

        with open(file1, 'w') as f:
            yaml.dump(config, f)
        with open(file2, 'w') as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Duplicate configuration keys"):
            load_config_files([str(file1), str(file2)])


# ============================================================================
# Tests for load_runner_file
# ============================================================================

class TestLoadRunnerFile:
    """Tests for the load_runner_file function."""

    def test_load_valid_runner_file(self, temp_config_files):
        _, runner_file = temp_config_files
        result = load_runner_file(runner_file)
        assert "h200" in result
        assert "b200" in result

    def test_load_nonexistent_runner_file(self):
        with pytest.raises(ValueError, match="does not exist"):
            load_runner_file("/nonexistent/runners.yaml")


# ============================================================================
# Tests for generate_full_sweep - Single Node
# ============================================================================

class TestGenerateFullSweepSingleNode:
    """Tests for generate_full_sweep with single-node configurations."""

    def test_basic_sweep(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert len(result) > 0
        assert all(entry['isl'] == 1024 and entry['osl'] == 1024 for entry in result)

    def test_sweep_no_filters(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(single_node=True)
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert len(result) > 0

    def test_sweep_returns_empty_when_no_matches(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            model_prefix=["nonexistent"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert result == []

    def test_filter_by_precision(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            precision=["fp8"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(entry['precision'] == 'fp8' for entry in result)

    def test_filter_by_framework(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            framework=["vllm"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(entry['framework'] == 'vllm' for entry in result)

    def test_filter_by_runner_type(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            runner_type=["h200"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(entry['runner'] == 'h200' for entry in result)

    def test_invalid_runner_type_raises_error(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            runner_type=["invalid-runner"],
            single_node=True
        )
        with pytest.raises(ValueError, match="Invalid runner type"):
            generate_full_sweep(args, sample_single_node_config, sample_runner_config)

    def test_multiple_runner_types(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            runner_type=["h200", "b200"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        runners = set(entry['runner'] for entry in result)
        assert 'h200' in runners or 'b200' in runners

    def test_filter_by_seq_lens(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k8k"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(entry['isl'] == 1024 and entry['osl'] == 8192 for entry in result)

    def test_filter_multiple_seq_lens(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k", "8k1k"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        seq_lens = set((e['isl'], e['osl']) for e in result)
        assert (1024, 1024) in seq_lens
        assert (8192, 1024) in seq_lens

    def test_step_size(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            model_prefix=["gptoss"],
            seq_lens=["1k1k"],
            step_size=4,
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        # With step_size=4, starting from 4: 4, 16, 64, 128 (or clamped)
        conc_values = set(e['conc'] for e in result)
        assert 4 in conc_values

    def test_max_conc_filter(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k"],
            max_conc=16,
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(entry['conc'] <= 16 for entry in result)

    def test_max_tp_filter(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k"],
            max_tp=4,
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(entry['tp'] <= 4 for entry in result)

    def test_max_ep_filter(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k"],
            max_ep=1,
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        # Should exclude entries with ep > 1
        assert all(entry['ep'] <= 1 for entry in result)

    def test_concurrency_overshoot_clamped(self, sample_runner_config):
        """Test that concurrency values are clamped to conc-end."""
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
                            {"tp": 4, "conc-start": 1, "conc-end": 5}
                        ]
                    }
                ]
            }
        }
        args = MockArgs(step_size=3, single_node=True)
        result = generate_full_sweep(args, config, sample_runner_config)
        conc_values = sorted(set(e['conc'] for e in result))
        # 1, 3, 9 -> clamped to 5
        assert conc_values == [1, 3, 5]

    def test_default_ep_dp_attn_values(self, sample_single_node_config, sample_runner_config):
        """Test that entries without ep/dp-attn get default values."""
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k"],
            max_tp=4,  # Filter to tp=4 which doesn't have ep/dp-attn
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        # tp=4 entries should have default ep=1 and dp-attn=False
        for entry in result:
            if entry['tp'] == 4:
                assert entry['ep'] == 1
                assert entry['dp-attn'] == False

    def test_explicit_ep_dp_attn_values(self, sample_single_node_config, sample_runner_config):
        """Test that entries with explicit ep/dp-attn use those values."""
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        # tp=8 entries should have ep=2 and dp-attn=True
        tp8_entries = [e for e in result if e['tp'] == 8]
        assert all(e['ep'] == 2 for e in tp8_entries)
        assert all(e['dp-attn'] == True for e in tp8_entries)

    def test_max_model_len_calculation(self, sample_single_node_config, sample_runner_config):
        """Test that max-model-len is calculated as isl + osl + 200."""
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k8k"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        # isl=1024, osl=8192 -> max-model-len = 1024 + 8192 + 200 = 9416
        assert all(e['max-model-len'] == 9416 for e in result)

    def test_exp_name_format(self, sample_single_node_config, sample_runner_config):
        """Test that exp-name follows the expected format."""
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(e['exp-name'] == 'dsr1_1k1k' for e in result)

    def test_disagg_defaults_to_false(self, sample_single_node_config, sample_runner_config):
        """Test that disagg defaults to False when not specified."""
        args = MockArgs(single_node=True)
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(e['disagg'] == False for e in result)

    def test_skips_multinode_configs_in_single_node_mode(self, sample_multinode_config, sample_runner_config):
        """Test that multinode configs are skipped when --single-node is specified."""
        args = MockArgs(single_node=True)
        result = generate_full_sweep(args, sample_multinode_config, sample_runner_config)
        assert result == []


# ============================================================================
# Tests for generate_full_sweep - Multi Node
# ============================================================================

class TestGenerateFullSweepMultiNode:
    """Tests for generate_full_sweep with multinode configurations."""

    def test_basic_multinode_sweep(self, sample_multinode_config, sample_runner_config):
        args = MockArgs(multi_node=True)
        result = generate_full_sweep(args, sample_multinode_config, sample_runner_config)
        assert len(result) > 0

    def test_multinode_conc_is_list(self, sample_multinode_config, sample_runner_config):
        """Test that multinode entries have conc as a list."""
        args = MockArgs(multi_node=True)
        result = generate_full_sweep(args, sample_multinode_config, sample_runner_config)
        for entry in result:
            assert isinstance(entry['conc'], list)

    def test_multinode_has_prefill_decode(self, sample_multinode_config, sample_runner_config):
        """Test that multinode entries have prefill and decode configs."""
        args = MockArgs(multi_node=True)
        result = generate_full_sweep(args, sample_multinode_config, sample_runner_config)
        for entry in result:
            assert 'prefill' in entry
            assert 'decode' in entry

    def test_multinode_spec_decoding_defaults_to_none(self, sample_multinode_config, sample_runner_config):
        """Test that spec-decoding defaults to 'none' if not specified."""
        args = MockArgs(multi_node=True)
        result = generate_full_sweep(args, sample_multinode_config, sample_runner_config)
        # The second search-space entry doesn't specify spec-decoding
        for entry in result:
            assert entry['spec-decoding'] in ['mtp', 'none']

    def test_multinode_disagg_value(self, sample_multinode_config, sample_runner_config):
        """Test that disagg is properly passed through."""
        args = MockArgs(multi_node=True)
        result = generate_full_sweep(args, sample_multinode_config, sample_runner_config)
        # The sample config has disagg=True
        assert all(e['disagg'] == True for e in result)

    def test_multinode_max_conc_filter(self, sample_multinode_config, sample_runner_config):
        """Test max_conc filter works with multinode conc lists."""
        args = MockArgs(multi_node=True, max_conc=8)
        result = generate_full_sweep(args, sample_multinode_config, sample_runner_config)
        for entry in result:
            assert all(c <= 8 for c in entry['conc'])

    def test_multinode_max_conc_filters_out_empty(self, sample_multinode_config, sample_runner_config):
        """Test that entries with no valid conc values are filtered out."""
        args = MockArgs(multi_node=True, max_conc=0)
        result = generate_full_sweep(args, sample_multinode_config, sample_runner_config)
        assert result == []

    def test_skips_single_node_configs_in_multi_node_mode(self, sample_single_node_config, sample_runner_config):
        """Test that single-node configs are skipped when --multi-node is specified."""
        args = MockArgs(multi_node=True)
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert result == []


# ============================================================================
# Tests for generate_runner_model_sweep_config
# ============================================================================

class TestGenerateRunnerModelSweepConfig:
    """Tests for the generate_runner_model_sweep_config function."""

    def test_basic_runner_model_sweep(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(runner_type="h200")
        result = generate_runner_model_sweep_config(args, sample_single_node_config, sample_runner_config)
        assert len(result) > 0
        runners = set(entry['runner'] for entry in result)
        assert 'h200-nv_1' in runners
        assert 'h200-nv_2' in runners

    def test_invalid_runner_type(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(runner_type="invalid-runner")
        with pytest.raises(ValueError, match="does not exist in runner config"):
            generate_runner_model_sweep_config(args, sample_single_node_config, sample_runner_config)

    def test_runner_node_filter(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(runner_type="h200", runner_node_filter="nv_1")
        result = generate_runner_model_sweep_config(args, sample_single_node_config, sample_runner_config)
        runners = set(entry['runner'] for entry in result)
        assert 'h200-nv_1' in runners
        assert 'h200-nv_2' not in runners

    def test_runner_node_filter_multiple_matches(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(runner_type="h200", runner_node_filter="nv")
        result = generate_runner_model_sweep_config(args, sample_single_node_config, sample_runner_config)
        runners = set(entry['runner'] for entry in result)
        assert 'h200-nv_1' in runners
        assert 'h200-nv_2' in runners

    def test_runner_node_filter_no_matches(self, sample_single_node_config, sample_runner_config):
        args = MockArgs(runner_type="h200", runner_node_filter="nonexistent")
        with pytest.raises(ValueError, match="No runner nodes found matching filter"):
            generate_runner_model_sweep_config(args, sample_single_node_config, sample_runner_config)

    def test_uses_highest_tp_lowest_conc(self, sample_single_node_config, sample_runner_config):
        """Test that it uses highest TP with lowest concurrency."""
        args = MockArgs(runner_type="h200")
        result = generate_runner_model_sweep_config(args, sample_single_node_config, sample_runner_config)
        # dsr1 config has tp=4 (conc 4-64) and tp=8 (conc 4-64), should pick tp=8, conc=4
        for entry in result:
            assert entry['tp'] == 8
            assert entry['conc'] == 4

    def test_always_uses_1k1k(self, sample_single_node_config, sample_runner_config):
        """Test that it always uses 1k1k sequence lengths."""
        args = MockArgs(runner_type="h200")
        result = generate_runner_model_sweep_config(args, sample_single_node_config, sample_runner_config)
        assert all(entry['isl'] == 1024 and entry['osl'] == 1024 for entry in result)

    def test_exp_name_has_test_suffix(self, sample_single_node_config, sample_runner_config):
        """Test that exp-name has _test suffix."""
        args = MockArgs(runner_type="h200")
        result = generate_runner_model_sweep_config(args, sample_single_node_config, sample_runner_config)
        assert all('_test' in entry['exp-name'] for entry in result)


# ============================================================================
# Tests for main function
# ============================================================================

class TestMain:
    """Tests for the main function with CLI argument parsing."""

    def test_main_full_sweep_single_node(self, temp_config_files):
        master_file, runner_file = temp_config_files

        test_args = [
            "generate_sweep_configs.py",
            "full-sweep",
            "--config-files", master_file,
            "--runner-config", runner_file,
            "--seq-lens", "1k1k",
            "--model-prefix", "dsr1",
            "--step-size", "2",
            "--single-node"
        ]

        with patch('sys.argv', test_args):
            result = main()
            assert len(result) > 0

    def test_main_full_sweep_multi_node(self, temp_multinode_config_files):
        master_file, runner_file = temp_multinode_config_files

        test_args = [
            "generate_sweep_configs.py",
            "full-sweep",
            "--config-files", master_file,
            "--runner-config", runner_file,
            "--multi-node"
        ]

        with patch('sys.argv', test_args):
            result = main()
            assert len(result) > 0

    def test_main_full_sweep_with_filters(self, temp_config_files):
        master_file, runner_file = temp_config_files

        test_args = [
            "generate_sweep_configs.py",
            "full-sweep",
            "--config-files", master_file,
            "--runner-config", runner_file,
            "--model-prefix", "dsr1",
            "--precision", "fp8",
            "--framework", "sglang",
            "--single-node"
        ]

        with patch('sys.argv', test_args):
            result = main()
            assert len(result) > 0
            assert all(entry['precision'] == 'fp8' for entry in result)
            assert all(entry['framework'] == 'sglang' for entry in result)

    def test_main_full_sweep_empty_result(self, temp_config_files):
        """Test that empty results are returned without error."""
        master_file, runner_file = temp_config_files

        test_args = [
            "generate_sweep_configs.py",
            "full-sweep",
            "--config-files", master_file,
            "--runner-config", runner_file,
            "--model-prefix", "nonexistent",
            "--single-node"
        ]

        with patch('sys.argv', test_args):
            result = main()
            assert result == []

    def test_main_runner_model_sweep(self, temp_config_files):
        master_file, runner_file = temp_config_files

        test_args = [
            "generate_sweep_configs.py",
            "runner-model-sweep",
            "--config-files", master_file,
            "--runner-config", runner_file,
            "--runner-type", "h200"
        ]

        with patch('sys.argv', test_args):
            result = main()
            assert len(result) > 0

    def test_main_runner_model_sweep_with_filter(self, temp_config_files):
        master_file, runner_file = temp_config_files

        test_args = [
            "generate_sweep_configs.py",
            "runner-model-sweep",
            "--config-files", master_file,
            "--runner-config", runner_file,
            "--runner-type", "h200",
            "--runner-node-filter", "nv_1"
        ]

        with patch('sys.argv', test_args):
            result = main()
            assert len(result) > 0
            runners = set(entry['runner'] for entry in result)
            assert 'h200-nv_1' in runners
            assert 'h200-nv_2' not in runners


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_concurrency_range_equals_start_end(self, sample_runner_config):
        """Test when conc-start equals conc-end."""
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
                            {"tp": 4, "conc-start": 8, "conc-end": 8}
                        ]
                    }
                ]
            }
        }
        args = MockArgs(single_node=True)
        result = generate_full_sweep(args, config, sample_runner_config)
        assert len(result) == 1
        assert result[0]['conc'] == 8

    def test_multiple_model_prefixes(self, sample_single_node_config, sample_runner_config):
        """Test filtering with multiple model prefixes."""
        args = MockArgs(
            model_prefix=["dsr1", "gptoss"],
            seq_lens=["1k1k"],
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        exp_names = [e['exp-name'] for e in result]
        assert any('dsr1' in name for name in exp_names)
        assert any('gptoss' in name for name in exp_names)

    def test_combined_max_filters(self, sample_single_node_config, sample_runner_config):
        """Test combining max_tp, max_ep, and max_conc filters."""
        args = MockArgs(
            model_prefix=["dsr1"],
            seq_lens=["1k1k"],
            max_tp=4,
            max_conc=8,
            single_node=True
        )
        result = generate_full_sweep(args, sample_single_node_config, sample_runner_config)
        assert all(entry['tp'] <= 4 for entry in result)
        assert all(entry['conc'] <= 8 for entry in result)

    def test_multinode_conc_range_instead_of_list(self, sample_runner_config):
        """Test multinode config with conc-start/conc-end instead of conc-list."""
        config = {
            "multinode-config": {
                "image": "test:latest",
                "model": "test/model",
                "model-prefix": "test",
                "runner": "gb200",
                "precision": "fp8",
                "framework": "dynamo-trt",
                "multinode": True,
                "disagg": True,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "conc-start": 4,
                                "conc-end": 16,
                                "prefill": {
                                    "num-worker": 1,
                                    "tp": 4,
                                    "ep": 4,
                                    "dp-attn": False
                                },
                                "decode": {
                                    "num-worker": 1,
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
        args = MockArgs(multi_node=True, step_size=2)
        result = generate_full_sweep(args, config, sample_runner_config)
        assert len(result) == 1
        # conc should be [4, 8, 16]
        assert result[0]['conc'] == [4, 8, 16]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
