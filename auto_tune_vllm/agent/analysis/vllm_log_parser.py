"""
vLLM Server Log Parser

Parses raw vLLM server log text into structured sections using regex patterns.
Extracts engine configuration, compilation settings, memory allocation, timing
information, and warnings/errors.

SOURCE: AI-Analysis-Agent/psap-mcp-server/psap_mcp_server/src/tools/vllm_log_tool.py
Converted to sync, no S3/MCP/async dependencies.

Public API:
    parse_vllm_log(raw: str) -> dict
        Parse raw vLLM log text into structured sections.
"""

import re
from typing import Any, Dict, Optional


def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _safe_int(text: str) -> Optional[int]:
    try:
        return int(text)
    except (ValueError, TypeError):
        return None


def parse_vllm_log(raw: str) -> Dict[str, Any]:
    """Parse a vLLM server log into structured sections.

    Extracts configuration, compilation, memory, and timing information
    using regex patterns matched against known vLLM log output formats.

    Args:
        raw: Raw log text (e.g. from ``cat /tmp/vllm*.log`` or process stdout).

    Returns:
        Dict with keys: server_config, engine_config, compilation, memory,
        timing, warnings_errors. Empty sections are omitted.
    """
    result: Dict[str, Any] = {
        "server_config": {},
        "engine_config": {},
        "compilation": {},
        "memory": {},
        "timing": {},
        "warnings_errors": [],
    }

    lines = raw.splitlines()

    for line in lines:
        # --- Server config ---
        m = re.search(r"vLLM API server version (\S+)", line)
        if m:
            result["server_config"]["vllm_version"] = m.group(1)

        m = re.search(r"non-default args:\s*(\{.+\})", line)
        if m:
            try:
                import ast

                args = ast.literal_eval(m.group(1))
                result["server_config"]["non_default_args"] = args
                for key in (
                    "model",
                    "max_model_len",
                    "gpu_memory_utilization",
                    "enable_prefix_caching",
                    "trust_remote_code",
                    "port",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "enable_chunked_prefill",
                    "enforce_eager",
                    "tensor_parallel_size",
                    "quantization",
                    "scheduling_policy",
                ):
                    if key in args:
                        result["server_config"][key] = args[key]
            except Exception:
                result["server_config"]["non_default_args_raw"] = m.group(1)

        # --- Architecture ---
        m = re.search(r"Resolved architecture:\s*(\S+)", line)
        if m:
            result["engine_config"]["architecture"] = m.group(1)

        # --- Engine config from init log line ---
        if (
            "Initializing a V1 LLM engine" in line
            or "Initializing an LLM engine" in line
        ):
            for field, pattern in [
                ("dtype", r"dtype=(\S+?)(?:,|$)"),
                ("quantization", r"quantization=(\S+?)(?:,|$)"),
                ("tensor_parallel_size", r"tensor_parallel_size=(\d+)"),
                ("pipeline_parallel_size", r"pipeline_parallel_size=(\d+)"),
                ("data_parallel_size", r"data_parallel_size=(\d+)"),
                ("max_seq_len", r"max_seq_len=(\d+)"),
                ("enforce_eager", r"enforce_eager=(\S+?)(?:,|$)"),
                ("enable_prefix_caching", r"enable_prefix_caching=(\S+?)(?:,|$)"),
                ("enable_chunked_prefill", r"enable_chunked_prefill=(\S+?)(?:,|$)"),
                ("load_format", r"load_format=(\S+?)(?:,|$)"),
                ("kv_cache_dtype", r"kv_cache_dtype=(\S+?)(?:,|$)"),
            ]:
                em = re.search(pattern, line)
                if em:
                    val = em.group(1)
                    if val.isdigit():
                        result["engine_config"][field] = int(val)
                    elif val in ("True", "False"):
                        result["engine_config"][field] = val == "True"
                    else:
                        result["engine_config"][field] = val

            # CUDA graph mode
            em = re.search(r"cudagraph_mode=<CUDAGraphMode\.(\S+?):", line)
            if em:
                result["engine_config"]["cudagraph_mode"] = em.group(1)

            # Max cudagraph capture size
            em = re.search(r"max_cudagraph_capture_size['\"]?:\s*(\d+)", line)
            if em:
                result["engine_config"]["max_cudagraph_capture_size"] = int(em.group(1))

            # Count cudagraph capture sizes
            em = re.search(r"cudagraph_capture_sizes['\"]?:\s*\[([^\]]+)\]", line)
            if em:
                sizes = [s.strip() for s in em.group(1).split(",") if s.strip()]
                result["engine_config"]["num_cudagraph_capture_sizes"] = len(sizes)

        # --- Chunked prefill ---
        m = re.search(
            r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)", line
        )
        if m:
            result["engine_config"]["chunked_prefill"] = True
            result["engine_config"]["max_num_batched_tokens"] = int(m.group(1))

        m = re.search(r"Overriding max cuda graph capture size to (\d+)", line)
        if m:
            result["engine_config"]["max_cudagraph_capture_size_override"] = int(
                m.group(1)
            )

        # --- Attention backend ---
        m = re.search(r"Using (\S+) attention backend", line)
        if m:
            result["compilation"]["attention_backend"] = m.group(1)

        # --- MoE stream ---
        if "separate cuda stream for MoE shared_experts" in line:
            result["compilation"]["moe_shared_experts_stream"] = True

        # --- Quantization backend ---
        m = re.search(r"\[mxfp4\.py:\d+\]\s*Using (\S+) backend", line)
        if m:
            result["compilation"]["quantization_backend"] = m.group(1)

        # --- Model loading ---
        m = re.search(r"Loading weights took (\S+) seconds", line)
        if m:
            result["memory"]["weights_load_time_s"] = _safe_float(m.group(1))

        m = re.search(r"Model loading took (\S+) GiB memory and (\S+) seconds", line)
        if m:
            result["memory"]["model_memory_gib"] = _safe_float(m.group(1))
            result["memory"]["model_load_time_s"] = _safe_float(m.group(2))

        # --- torch.compile ---
        m = re.search(r"Dynamo bytecode transform time:\s*(\S+)\s*s", line)
        if m:
            result["compilation"]["dynamo_transform_time_s"] = _safe_float(m.group(1))

        m = re.search(r"Compiling a graph .+ takes (\S+)\s*s", line)
        if m:
            result["compilation"]["graph_compile_time_s"] = _safe_float(m.group(1))

        m = re.search(r"torch\.compile takes (\S+)\s*s in total", line)
        if m:
            result["compilation"]["torch_compile_time_s"] = _safe_float(m.group(1))

        # --- KV cache ---
        m = re.search(r"Available KV cache memory:\s*(\S+)\s*GiB", line)
        if m:
            result["memory"]["kv_cache_memory_gib"] = _safe_float(m.group(1))

        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", line)
        if m:
            result["memory"]["kv_cache_tokens"] = _safe_int(m.group(1).replace(",", ""))

        m = re.search(r"Maximum concurrency for .+ per request:\s*(\S+)x", line)
        if m:
            result["memory"]["max_concurrency"] = _safe_float(m.group(1))

        # --- CUDA graph capture ---
        m = re.search(
            r"Graph capturing finished in (\d+)\s*secs?, took (\S+)\s*GiB", line
        )
        if m:
            result["compilation"]["cuda_graph_capture_time_s"] = _safe_int(m.group(1))
            result["compilation"]["cuda_graph_memory_gib"] = _safe_float(m.group(2))

        # --- Engine init ---
        m = re.search(r"init engine .+ took (\S+) seconds", line)
        if m:
            result["timing"]["engine_init_time_s"] = _safe_float(m.group(1))

        # --- TP/PP rank info ---
        m = re.search(r"world_size=(\d+)\s+rank=(\d+)", line)
        if m:
            result["engine_config"]["world_size"] = int(m.group(1))

        m = re.search(r"TP rank (\d+), EP rank (\d+)", line)
        if m:
            result["engine_config"]["tp_rank"] = int(m.group(1))
            result["engine_config"]["ep_rank"] = int(m.group(2))

        # --- Serving info ---
        m = re.search(r"Uvicorn running on (\S+)", line)
        if m:
            result["server_config"]["uvicorn_url"] = m.group(1)

        m = re.search(r"Starting vLLM API server on (\S+):(\d+)", line)
        if m:
            result["server_config"]["host"] = m.group(1)
            result["server_config"]["port"] = int(m.group(2))

        # --- Warnings and errors ---
        if "WARNING" in line or "ERROR" in line:
            cleaned = line.strip()
            if (
                cleaned
                and "Loading safetensors" not in cleaned
                and "Capturing CUDA" not in cleaned
            ):
                result["warnings_errors"].append(cleaned)

    # Remove empty sections
    return {k: v for k, v in result.items() if v}
