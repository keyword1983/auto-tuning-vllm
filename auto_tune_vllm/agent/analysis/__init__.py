"""Analysis helpers used by the agentic tuner.

Extracted from the PSAP analysis MCP tools and converted to sync functions.
"""

from .cost import (
    ACCELERATOR_PRICING,
    calculate_cost,
    calculate_cpmt,
    compare_cost_efficiency,
    filter_by_slo,
)
from .kernel_mapper import (
    KERNEL_MAPPINGS,
    PYTORCH_STDLIB_OPS,
    find_kernel_mapping,
    is_pytorch_stdlib,
    map_kernel,
)
from .regression import detect_regression
from .trace_analyzer import (
    CATEGORY_GROUPS,
    FUNCTIONAL_PIPELINES,
    analyze_trace,
    classify_kernel,
    extract_kernel_stats,
    get_category_breakdown,
    get_pipeline_breakdown,
    get_top_kernels,
    merge_stats,
)
from .vllm_log_parser import parse_vllm_log

__all__ = [
    "analyze_trace",
    "extract_kernel_stats",
    "merge_stats",
    "get_top_kernels",
    "get_category_breakdown",
    "get_pipeline_breakdown",
    "classify_kernel",
    "FUNCTIONAL_PIPELINES",
    "CATEGORY_GROUPS",
    "map_kernel",
    "find_kernel_mapping",
    "is_pytorch_stdlib",
    "KERNEL_MAPPINGS",
    "PYTORCH_STDLIB_OPS",
    "detect_regression",
    "parse_vllm_log",
    "calculate_cost",
    "calculate_cpmt",
    "filter_by_slo",
    "compare_cost_efficiency",
    "ACCELERATOR_PRICING",
]
