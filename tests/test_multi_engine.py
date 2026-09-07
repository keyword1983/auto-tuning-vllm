"""Unit tests for multi-engine serving adapters and configuration."""

from auto_tune_vllm.engines import (
    get_engine_adapter,
    VLLMEngineAdapter,
    SGLangEngineAdapter,
    LlamaCppEngineAdapter,
)
from auto_tune_vllm.core.config import StudyConfig


def test_engine_factory():
    assert isinstance(get_engine_adapter("vllm"), VLLMEngineAdapter)
    assert isinstance(get_engine_adapter("VLLM"), VLLMEngineAdapter)
    assert isinstance(get_engine_adapter("sglang"), SGLangEngineAdapter)
    assert isinstance(get_engine_adapter("SGLang"), SGLangEngineAdapter)
    assert isinstance(get_engine_adapter("llamacpp"), LlamaCppEngineAdapter)
    assert isinstance(get_engine_adapter("llama.cpp"), LlamaCppEngineAdapter)
    assert isinstance(get_engine_adapter(None), VLLMEngineAdapter)


def test_vllm_adapter_mapping():
    adapter = VLLMEngineAdapter()
    params = {
        "batch_size": 32,
        "gpu_memory_utilization": 0.8,
        "max_num_batched_tokens": 4096,
        "tp": 2,
        "enforce_eager": True,
    }
    patch = adapter.map_to_serving_patch(params)
    assert patch["batchSize"] == "32"
    assert patch["gpuMemoryUtilization"] == "0.8"
    assert patch["prefillSettings"]["maxBatchTokens"] == "4096"
    assert patch["parallelism"]["tp"] == 2
    assert "--enforce-eager" in patch["extraCommand"]


def test_sglang_adapter_mapping():
    adapter = SGLangEngineAdapter()
    params = {
        "max_running_requests": 64,
        "mem_fraction_static": 0.85,
        "chunked_prefill_size": 1024,
        "schedule_policy": "lpm",
        "attention_backend": "flashinfer",
        "tp_size": 4,
    }
    patch = adapter.map_to_serving_patch(params)
    assert patch["batchSize"] == "64"
    assert patch["gpuMemoryUtilization"] == "0.85"
    assert patch["parallelism"]["tp"] == 4
    assert "--chunked-prefill-size=1024" in patch["extraCommand"]
    assert "--schedule-policy=lpm" in patch["extraCommand"]
    assert "--attention-backend=flashinfer" in patch["extraCommand"]


def test_llamacpp_adapter_mapping():
    adapter = LlamaCppEngineAdapter()
    params = {
        "parallel": 8,
        "ctx_size": 4096,
        "threads": 16,
        "ubatch_size": 256,
        "n_gpu_layers": 33,
        "flash_attn": True,
    }
    patch = adapter.map_to_serving_patch(params)
    assert patch["batchSize"] == "8"
    assert patch["contextLength"] == "4096"
    assert "--threads=16" in patch["extraCommand"]
    assert "--ubatch-size=256" in patch["extraCommand"]
    assert "--n-gpu-layers=33" in patch["extraCommand"]
    assert "--flash-attn" in patch["extraCommand"]


def test_baseline_extraction():
    vllm_tmpl = {"batchSize": "32", "gpuMemoryUtilization": "0.8", "contextLength": "2048"}
    assert VLLMEngineAdapter().extract_baseline_parameters(vllm_tmpl) == {
        "batch_size": 32,
        "gpu_memory_utilization": 0.8,
        "context_length": 2048,
    }

    sglang_tmpl = {
        "batchSize": "64",
        "gpuMemoryUtilization": "0.85",
        "extraCommand": ["--schedule-policy=lpm", "--chunked-prefill-size=1024"],
    }
    sglang_baseline = SGLangEngineAdapter().extract_baseline_parameters(sglang_tmpl)
    assert sglang_baseline["batch_size"] == 64
    assert sglang_baseline["gpu_memory_utilization"] == 0.85
    assert sglang_baseline["schedule_policy"] == "lpm"
    assert sglang_baseline["chunked_prefill_size"] == 1024

    llamacpp_tmpl = {
        "batchSize": "16",
        "contextLength": "4096",
        "extraCommand": ["--threads=8", "--ubatch-size=512", "--n-gpu-layers=20"],
    }
    llamacpp_baseline = LlamaCppEngineAdapter().extract_baseline_parameters(llamacpp_tmpl)
    assert llamacpp_baseline["batch_size"] == 16
    assert llamacpp_baseline["context_length"] == 4096
    assert llamacpp_baseline["threads"] == 8
    assert llamacpp_baseline["ubatch_size"] == 512
    assert llamacpp_baseline["n_gpu_layers"] == 20


def test_study_config_from_yaml():
    sglang_cfg = StudyConfig.from_file("examples/study_config_sglang.yaml")
    assert sglang_cfg.engine == "sglang"
    assert "batch_size" in sglang_cfg.parameters
    assert "chunked_prefill_size" in sglang_cfg.parameters
    assert "schedule_policy" in sglang_cfg.parameters

    llamacpp_cfg = StudyConfig.from_file("examples/study_config_llamacpp.yaml")
    assert llamacpp_cfg.engine == "llamacpp"
    assert "batch_size" in llamacpp_cfg.parameters
    assert "threads" in llamacpp_cfg.parameters
    assert "ubatch_size" in llamacpp_cfg.parameters


def test_cr_synthesis_multi_engine():
    from unittest.mock import MagicMock, patch
    from auto_tune_vllm.execution.afsbox import synthesize_study_config_from_cr

    mock_cr_sglang = {
        "spec": {
            "optimization": {"nTrials": 5, "sampler": "tpe"},
            "servingTemplate": {
                "batchSize": "32",
                "gpuMemoryUtilization": "0.8",
                "engine": {"type": "sglang", "servicePort": 30000},
            },
            "testSuite": [{"params": {"concurrency": 4, "requestCount": 20}}],
        }
    }

    with patch("kubernetes.client.CustomObjectsApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = mock_cr_sglang
        mock_api_cls.return_value = mock_api
        with patch("kubernetes.config.load_incluster_config"):
            tmp_path = synthesize_study_config_from_cr("mock-sglang-test", "default")
            sglang_study = StudyConfig.from_file(tmp_path)
            assert sglang_study.engine == "sglang"
            assert "batch_size" in sglang_study.parameters
            assert sglang_study.baseline.parameters["batch_size"] == 32
            assert sglang_study.baseline.parameters["gpu_memory_utilization"] == 0.8


if __name__ == "__main__":
    print("Running test_engine_factory()...")
    test_engine_factory()
    print("Running test_vllm_adapter_mapping()...")
    test_vllm_adapter_mapping()
    print("Running test_sglang_adapter_mapping()...")
    test_sglang_adapter_mapping()
    print("Running test_llamacpp_adapter_mapping()...")
    test_llamacpp_adapter_mapping()
    print("Running test_baseline_extraction()...")
    test_baseline_extraction()
    print("Running test_study_config_from_yaml()...")
    test_study_config_from_yaml()
    print("Running test_cr_synthesis_multi_engine()...")
    test_cr_synthesis_multi_engine()
    print("\nALL MULTI-ENGINE UNIT TESTS (CR + CONFIG) PASSED SUCCESSFULLY!")
