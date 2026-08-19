"""
Kernel-to-Source-Code Mapper

Maps CUDA kernel names to their vLLM/PyTorch source code locations
and provides human-readable descriptions of kernel functionality.

SOURCE: AI-Analysis-Agent/psap-mcp-server/psap_mcp_server/src/tools/kernel_code_mapper_tool.py

Converted to sync plain Python functions. No MCP, async, httpx, or logger.

Public API:
    map_kernel(kernel_name: str) -> dict
        Main entry point. Returns source info, category, description.

    find_kernel_mapping(kernel_name: str) -> list[dict]
    is_pytorch_stdlib(kernel_name: str) -> bool

Constants:
    KERNEL_MAPPINGS
    PYTORCH_STDLIB_OPS
"""

import re
from typing import Dict, List

# ===================================================================== #
#  Curated kernel-to-source mappings                                     #
# ===================================================================== #
# Each entry is (regex_pattern, [(path, confidence, description), ...]).

KERNEL_MAPPINGS = [
    # Flash Attention kernels
    (
        r"flash_attn.*|FlashAttn.*",
        [
            (
                "vllm/attention/backends/flash_attn.py",
                "high",
                "Flash Attention backend implementation",
            ),
            ("vllm/attention/ops/", "medium", "Attention operation implementations"),
            ("csrc/attention/", "medium", "C++/CUDA attention kernels"),
        ],
    ),
    # PagedAttention kernels
    (
        r"paged_attention.*|PagedAttention.*",
        [
            ("vllm/attention/backends/", "high", "Paged attention backend"),
            (
                "csrc/attention/attention_kernels.cu",
                "high",
                "CUDA paged attention kernels",
            ),
        ],
    ),
    # MoE (Mixture of Experts) kernels -- important for DeepSeek
    (
        r".*moe.*|.*expert.*|.*MoE.*|fused_moe.*",
        [
            (
                "vllm/model_executor/layers/fused_moe/",
                "high",
                "Fused MoE layer implementations",
            ),
            ("csrc/moe/", "high", "C++/CUDA MoE kernels"),
            ("vllm/model_executor/layers/moe/", "medium", "MoE layer Python code"),
        ],
    ),
    # Quantization kernels
    (
        r".*awq.*|.*gptq.*|.*fp8.*|.*int8.*|.*quant.*",
        [
            (
                "vllm/model_executor/layers/quantization/",
                "high",
                "Quantization implementations",
            ),
            ("csrc/quantization/", "high", "CUDA quantization kernels"),
        ],
    ),
    # GEMM / Matrix multiplication
    (
        r"aten::mm|aten::bmm|aten::matmul|aten::linear|cutlass.*gemm.*|cublas.*gemm.*",
        [
            (
                "vllm/model_executor/layers/linear.py",
                "medium",
                "Linear layer implementations",
            ),
            ("csrc/", "low", "Custom CUDA kernels may override"),
        ],
    ),
    # Activation functions
    (
        r"aten::silu|aten::gelu|aten::relu|.*activation.*|silu_and_mul.*",
        [
            (
                "vllm/model_executor/layers/activation.py",
                "high",
                "Activation function implementations",
            ),
            ("csrc/activation_kernels.cu", "medium", "CUDA activation kernels"),
        ],
    ),
    # LayerNorm / RMSNorm
    (
        r".*layer_norm.*|.*rms_norm.*|.*LayerNorm.*|.*RMSNorm.*|aten::layer_norm",
        [
            (
                "vllm/model_executor/layers/layernorm.py",
                "high",
                "LayerNorm implementations",
            ),
            ("csrc/layernorm_kernels.cu", "medium", "CUDA LayerNorm kernels"),
        ],
    ),
    # Rotary embeddings
    (
        r".*rotary.*|.*rope.*|.*RoPE.*",
        [
            (
                "vllm/model_executor/layers/rotary_embedding.py",
                "high",
                "Rotary embedding implementations",
            ),
            (
                "csrc/pos_encoding_kernels.cu",
                "medium",
                "CUDA position encoding kernels",
            ),
        ],
    ),
    # Sampling / Top-k/p
    (
        r".*sample.*|.*topk.*|.*topp.*|.*argmax.*",
        [
            (
                "vllm/model_executor/layers/sampler.py",
                "high",
                "Sampling implementations",
            ),
            ("csrc/", "low", "CUDA sampling kernels"),
        ],
    ),
    # Communication / NCCL
    (
        r"nccl.*|ncclAllReduce.*|ncclAllGather.*|c10d.*",
        [
            ("vllm/distributed/", "high", "Distributed communication code"),
            ("vllm/executor/", "medium", "Executor implementations"),
        ],
    ),
    # Memory operations
    (
        r"aten::copy_|aten::clone|aten::contiguous|aten::to|cudaMemcpy.*",
        [
            ("vllm/worker/", "medium", "Worker implementations handle memory"),
            ("vllm/attention/backends/", "low", "Attention backends may copy tensors"),
        ],
    ),
    # KV Cache operations
    (
        r".*kv_cache.*|.*cache_copy.*|.*reshape_and_cache.*",
        [
            (
                "vllm/attention/backends/",
                "high",
                "KV cache operations in attention backends",
            ),
            ("csrc/cache_kernels.cu", "high", "CUDA cache kernels"),
            ("vllm/worker/cache_engine.py", "medium", "Cache engine implementation"),
        ],
    ),
    # Embedding operations
    (
        r"aten::embedding|.*embed.*",
        [
            (
                "vllm/model_executor/layers/vocab_parallel_embedding.py",
                "high",
                "Embedding layer implementations",
            ),
        ],
    ),
    # Softmax
    (
        r"aten::softmax|aten::_softmax|.*softmax.*",
        [
            ("vllm/attention/", "medium", "Softmax used in attention"),
            ("csrc/attention/", "medium", "CUDA attention with fused softmax"),
        ],
    ),
    # Custom vLLM ops
    (
        r"vllm::.*",
        [
            ("csrc/", "high", "Custom vLLM CUDA ops"),
            ("vllm/", "medium", "Python wrapper code"),
        ],
    ),
    # Triton kernels
    (
        r"triton.*|Triton.*",
        [
            ("vllm/attention/ops/", "high", "Triton attention kernels"),
            ("vllm/model_executor/layers/fused_moe/", "medium", "Triton MoE kernels"),
        ],
    ),
]


# ===================================================================== #
#  PyTorch standard-library ops (not vLLM-specific)                      #
# ===================================================================== #

PYTORCH_STDLIB_OPS = [
    r"aten::empty.*",
    r"aten::zeros.*",
    r"aten::ones.*",
    r"aten::view.*",
    r"aten::reshape.*",
    r"aten::transpose.*",
    r"aten::permute.*",
    r"aten::squeeze.*",
    r"aten::unsqueeze.*",
    r"aten::cat.*",
    r"aten::stack.*",
    r"aten::split.*",
    r"aten::chunk.*",
    r"aten::select.*",
    r"aten::slice.*",
    r"aten::index.*",
    r"aten::as_strided.*",
    r"aten::expand.*",
    r"aten::repeat.*",
    r"aten::fill_.*",
    r"aten::zero_.*",
    r"aten::add.*",
    r"aten::sub.*",
    r"aten::mul.*",
    r"aten::div.*",
    r"aten::pow.*",
    r"aten::sqrt.*",
    r"aten::rsqrt.*",
    r"aten::exp.*",
    r"aten::log.*",
    r"aten::abs.*",
    r"aten::neg.*",
    r"aten::sum.*",
    r"aten::mean.*",
    r"aten::max.*",
    r"aten::min.*",
    r"aten::where.*",
    r"aten::masked.*",
    r"aten::scatter.*",
    r"aten::gather.*",
]


# ===================================================================== #
#  Core functions                                                        #
# ===================================================================== #


def is_pytorch_stdlib(kernel_name: str) -> bool:
    """Check whether a kernel name is a standard PyTorch ATen operation.

    These operations are implemented in PyTorch core rather than in
    vLLM custom code.

    Args:
        kernel_name: The kernel / operation name from a profiler trace.

    Returns:
        ``True`` if the kernel matches any ``PYTORCH_STDLIB_OPS`` pattern.
    """
    for pattern in PYTORCH_STDLIB_OPS:
        if re.match(pattern, kernel_name, re.IGNORECASE):
            return True
    return False


def find_kernel_mapping(kernel_name: str) -> List[Dict[str, str]]:
    """Find likely vLLM source-file locations for a kernel name.

    Matches *kernel_name* against ``KERNEL_MAPPINGS`` patterns and returns
    all matching entries with path, confidence level, description, and the
    pattern that matched.

    Args:
        kernel_name: The kernel / operation name from a profiler trace.

    Returns:
        List of dicts, each with keys ``path``, ``confidence``,
        ``description``, ``matched_pattern``.  May be empty if no
        pattern matches.
    """
    results = []
    for pattern, mappings in KERNEL_MAPPINGS:
        if re.search(pattern, kernel_name, re.IGNORECASE):
            for path, confidence, description in mappings:
                results.append(
                    {
                        "path": path,
                        "confidence": confidence,
                        "description": description,
                        "matched_pattern": pattern,
                    }
                )
    return results


def _infer_category(kernel_name: str, mappings: List[Dict[str, str]]) -> str:
    """Infer a human-readable category from the kernel name or its mappings.

    Uses simple heuristics on the file paths / kernel name to assign a
    broad category label.
    """
    lower = kernel_name.lower()

    # Check paths from mappings first
    all_paths = " ".join(m.get("path", "") for m in mappings).lower()

    if "attention" in lower or "attention" in all_paths:
        return "attention"
    if "moe" in lower or "expert" in lower or "fused_moe" in all_paths:
        return "moe"
    if "quant" in lower or "fp8" in lower or "quantization" in all_paths:
        return "quantization"
    if "nccl" in lower or "allreduce" in lower or "distributed" in all_paths:
        return "communication"
    if "norm" in lower or "layernorm" in all_paths:
        return "normalization"
    if "activation" in lower or "silu" in lower or "gelu" in lower:
        return "activation"
    if "gemm" in lower or "matmul" in lower or "linear" in all_paths:
        return "gemm_linear"
    if "cache" in lower:
        return "kv_cache"
    if "embed" in lower:
        return "embedding"
    if "sample" in lower or "topk" in lower:
        return "sampling"
    if "rotary" in lower or "rope" in lower:
        return "rotary_embedding"
    if "softmax" in lower:
        return "softmax"
    if "triton" in lower:
        return "triton"
    if lower.startswith("vllm::"):
        return "vllm_custom"
    if lower.startswith("aten::"):
        return "pytorch_aten"
    if lower.startswith("cuda"):
        return "cuda_runtime"

    return "other"


# ===================================================================== #
#  Main entry point                                                      #
# ===================================================================== #


def map_kernel(kernel_name: str) -> dict:
    """Map a kernel / operation name to its source info, category, and description.

    This is the primary entry point. It combines ``find_kernel_mapping``,
    ``is_pytorch_stdlib``, and category inference into a single convenient
    call.

    Args:
        kernel_name: The kernel name from a PyTorch profiler trace
            (e.g. ``"flash_attn_v2_fwd"``, ``"fused_moe_kernel"``,
            ``"aten::mm"``).

    Returns:
        Dict with keys:
        - ``kernel_name``: The input name.
        - ``is_pytorch_stdlib``: Whether it is a standard PyTorch op.
        - ``category``: Inferred broad category string.
        - ``likely_source_files``: List of source-location dicts (may be
          empty).
        - ``description``: Human-readable summary.
        - ``suggestions``: List of follow-up suggestions.
    """
    stdlib = is_pytorch_stdlib(kernel_name)
    mappings = find_kernel_mapping(kernel_name)
    category = _infer_category(kernel_name, mappings)

    # Build a concise description
    if stdlib:
        description = (
            f"'{kernel_name}' is a standard PyTorch ATen operation "
            "implemented in PyTorch core, not vLLM-specific."
        )
    elif mappings:
        top = mappings[0]
        description = (
            f"'{kernel_name}' likely maps to {top['path']} "
            f"({top['confidence']} confidence): {top['description']}."
        )
    else:
        description = (
            f"No curated mapping found for '{kernel_name}'. "
            "It may be from a third-party library or a newly added kernel."
        )

    # Suggestions for further investigation
    suggestions: List[str] = []
    if stdlib:
        suggestions.append(
            "This is a standard PyTorch operation. Performance may be "
            "affected by how vLLM uses this op (tensor shapes, dtypes)."
        )
    elif not mappings:
        suggestions.append(
            f"Try searching the vLLM repo for '{kernel_name}' to find "
            "the implementation."
        )
        suggestions.append(
            "Check if this kernel is from a third-party library "
            "(e.g., flash-attn, triton, deep_gemm)."
        )
    else:
        suggestions.append(
            "Review the source files listed above to understand the "
            "kernel implementation."
        )

    return {
        "kernel_name": kernel_name,
        "is_pytorch_stdlib": stdlib,
        "category": category,
        "likely_source_files": mappings,
        "description": description,
        "suggestions": suggestions,
    }
