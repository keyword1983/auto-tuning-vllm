"""
PyTorch Profiler Trace Analyzer

Parses Chrome trace JSON files from PyTorch profiler and extracts
kernel-level performance statistics.

SOURCE: AI-Analysis-Agent/psap-mcp-server/psap_mcp_server/src/tools/pytorch_profile_tool.py

Converted to sync plain Python functions. No MCP, S3, async, or logger.

Public API:
    analyze_trace(trace_json: dict) -> dict
        Main entry point. Returns kernel stats, top kernels, category breakdown.

    extract_kernel_stats(trace: dict) -> dict
    merge_stats(stats_list: list) -> dict
    get_top_kernels(stats: dict, n: int = 20) -> list
    get_category_breakdown(stats: dict) -> dict
    classify_kernel(kernel_name: str) -> str | None

Constants:
    FUNCTIONAL_PIPELINES
    CATEGORY_GROUPS
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Functional pipeline grouping: maps high-level execution pipelines to kernel
# name patterns.  Used to produce a category-level breakdown comparable to
# what a human expert would write.
# ---------------------------------------------------------------------------
FUNCTIONAL_PIPELINES: Dict[str, Dict[str, Any]] = {
    "moe_execution": {
        "patterns": [
            "moe",
            "expert",
            "scatter",
            "gather",
            "fused_moe",
            "deep_gemm",
            "outplace_fused",
        ],
        "description": "MoE routing + expert GEMM execution pipeline",
    },
    "attention": {
        "patterns": [
            "attention",
            "flash",
            "mla",
            "paged",
            "kv_cache",
            "reshape_and_cache",
        ],
        "description": "Attention mechanism (MLA / MHA / GQA / Flash / Paged)",
    },
    "communication": {
        "patterns": [
            "nccl",
            "allreduce",
            "allgather",
            "multimem",
            "c10d",
            "all_reduce",
        ],
        "description": "Inter-GPU collective communication",
    },
    "quantization": {
        "patterns": [
            "quant",
            "fp8",
            "int8",
            "dequant",
            "scale",
            "per_token_group",
        ],
        "description": "Quantization / dequantization / scaling operations",
    },
    "normalization_activation": {
        "patterns": [
            "layernorm",
            "rmsnorm",
            "rms_norm",
            "silu",
            "gelu",
            "activation",
            "act_and_mul",
        ],
        "description": "Normalization (RMSNorm/LayerNorm) and activation functions",
    },
    "gemm_linear": {
        "patterns": [
            "gemm",
            "matmul",
            "aten::mm",
            "aten::bmm",
            "aten::linear",
            "cublas",
            "cutlass",
        ],
        "description": "Standalone matrix multiplication / linear layers (not MoE)",
    },
}

# Category groups for filtering trace events by their ``cat`` field.
CATEGORY_GROUPS: Dict[str, Optional[List[str]]] = {
    "kernel": ["kernel"],
    "cpu": ["cpu_op"],
    "cuda": ["cuda_runtime", "cuda_driver"],
    "communication": ["nccl", "c10d"],
    "memory": ["cuda_memory"],
    "all": None,
}


# ===================================================================== #
#  Core stat-extraction functions                                        #
# ===================================================================== #


def extract_kernel_stats(trace: dict) -> Dict[str, Dict]:
    """Extract kernel statistics from a Chrome trace JSON dict.

    Looks for complete-duration events (``ph == "X"``) and aggregates per
    unique operation name: invocation count, total / avg / min / max
    duration, and the trace category.

    Args:
        trace: Parsed Chrome trace JSON (must contain ``traceEvents``).

    Returns:
        Dict mapping ``kernel_name`` to a stats dict with keys:
        ``count``, ``total_dur``, ``avg_dur``, ``min_dur``, ``max_dur``,
        ``cat``.
    """
    stats: Dict[str, Dict] = defaultdict(
        lambda: {
            "count": 0,
            "total_dur": 0,
            "min_dur": float("inf"),
            "max_dur": 0,
            "cat": "",
        }
    )

    for event in trace.get("traceEvents", []):
        if event.get("ph") == "X":
            name = event.get("name", "unknown")
            dur = event.get("dur", 0)
            cat = event.get("cat", "")
            if dur > 0:
                stats[name]["count"] += 1
                stats[name]["total_dur"] += dur
                stats[name]["min_dur"] = min(stats[name]["min_dur"], dur)
                stats[name]["max_dur"] = max(stats[name]["max_dur"], dur)
                if not stats[name]["cat"]:
                    stats[name]["cat"] = cat

    for data in stats.values():
        data["avg_dur"] = data["total_dur"] / data["count"] if data["count"] > 0 else 0
        if data["min_dur"] == float("inf"):
            data["min_dur"] = 0

    return dict(stats)


def merge_stats(stats_list: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """Merge statistics from multiple ranks / traces into one view.

    Args:
        stats_list: List of per-rank stats dicts (as returned by
            ``extract_kernel_stats``).

    Returns:
        A single merged stats dict.
    """
    merged: Dict[str, Dict] = defaultdict(
        lambda: {
            "count": 0,
            "total_dur": 0,
            "min_dur": float("inf"),
            "max_dur": 0,
            "cat": "",
        }
    )

    for stats in stats_list:
        for name, data in stats.items():
            merged[name]["count"] += data["count"]
            merged[name]["total_dur"] += data["total_dur"]
            merged[name]["min_dur"] = min(
                merged[name]["min_dur"],
                data.get("min_dur", float("inf")),
            )
            merged[name]["max_dur"] = max(
                merged[name]["max_dur"],
                data.get("max_dur", 0),
            )
            if not merged[name]["cat"]:
                merged[name]["cat"] = data.get("cat", "")

    for data in merged.values():
        data["avg_dur"] = data["total_dur"] / data["count"] if data["count"] > 0 else 0
        if data["min_dur"] == float("inf"):
            data["min_dur"] = 0

    return dict(merged)


# ===================================================================== #
#  Formatting helper                                                     #
# ===================================================================== #


def _format_duration(us: float) -> str:
    """Format a duration given in microseconds to a human-readable string."""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    elif us >= 1_000:
        return f"{us / 1_000:.2f}ms"
    else:
        return f"{us:.2f}\u00b5s"


# ===================================================================== #
#  Top-kernels & category breakdown                                      #
# ===================================================================== #


def get_top_kernels(
    stats: Dict[str, Dict],
    n: int = 20,
    sort_by: str = "total_dur",
) -> List[Dict]:
    """Return the top *n* kernels sorted by the chosen metric.

    Args:
        stats: Kernel stats dict (from ``extract_kernel_stats`` or
            ``merge_stats``).
        n: Number of kernels to return (default 20).
        sort_by: Stats key to sort by (default ``"total_dur"``).

    Returns:
        List of dicts, each with keys: ``name``, ``category``, ``count``,
        ``total_dur_us``, ``total_dur_human``, ``avg_dur_us``,
        ``avg_dur_human``, ``min_dur_us``, ``max_dur_us``.
    """
    sorted_kernels = sorted(
        stats.items(),
        key=lambda x: x[1].get(sort_by, 0),
        reverse=True,
    )

    result = []
    for name, data in sorted_kernels[:n]:
        result.append(
            {
                "name": name,
                "category": data.get("cat", "unknown"),
                "count": data["count"],
                "total_dur_us": data["total_dur"],
                "total_dur_human": _format_duration(data["total_dur"]),
                "avg_dur_us": data.get("avg_dur", 0),
                "avg_dur_human": _format_duration(data.get("avg_dur", 0)),
                "min_dur_us": data.get("min_dur", 0),
                "max_dur_us": data.get("max_dur", 0),
            }
        )

    return result


def get_category_breakdown(stats: Dict[str, Dict]) -> Dict[str, Dict]:
    """Aggregate GPU time by trace-event category.

    Args:
        stats: Kernel stats dict.

    Returns:
        Dict mapping category name to a summary dict with keys:
        ``total_dur_us``, ``total_dur_human``, ``invocation_count``,
        ``unique_operations``.
    """
    cat_totals: Dict[str, Dict] = defaultdict(
        lambda: {"total_dur": 0, "count": 0, "kernel_count": 0}
    )

    for _name, data in stats.items():
        cat = data.get("cat", "other") or "other"
        cat_totals[cat]["total_dur"] += data["total_dur"]
        cat_totals[cat]["count"] += data["count"]
        cat_totals[cat]["kernel_count"] += 1

    result = {}
    for cat, totals in sorted(
        cat_totals.items(),
        key=lambda x: x[1]["total_dur"],
        reverse=True,
    ):
        result[cat] = {
            "total_dur_us": totals["total_dur"],
            "total_dur_human": _format_duration(totals["total_dur"]),
            "invocation_count": totals["count"],
            "unique_operations": totals["kernel_count"],
        }

    return result


# ===================================================================== #
#  Pipeline classification helper                                        #
# ===================================================================== #


def classify_kernel(kernel_name: str) -> Optional[str]:
    """Map a kernel name to its functional pipeline category.

    Matches against ``FUNCTIONAL_PIPELINES`` patterns (case-insensitive,
    substring match). Returns the first matching pipeline name, or
    ``None`` if no pattern matches.

    Args:
        kernel_name: The kernel / operation name from the trace.

    Returns:
        Pipeline name string (e.g. ``"attention"``, ``"moe_execution"``)
        or ``None``.
    """
    lower = kernel_name.lower()
    for pipeline_name, cfg in FUNCTIONAL_PIPELINES.items():
        if any(pat in lower for pat in cfg["patterns"]):
            return pipeline_name
    return None


def _filter_by_category(
    stats: Dict[str, Dict],
    category: Optional[str],
) -> Dict[str, Dict]:
    """Filter stats to only include a specified category group.

    Args:
        stats: Kernel stats dict.
        category: One of the keys in ``CATEGORY_GROUPS``, or ``None``
            (which means "all").

    Returns:
        Filtered stats dict.
    """
    if category is None or category == "all":
        return stats

    categories = CATEGORY_GROUPS.get(category.lower())
    if categories is None:
        return stats

    lower_cats = [c.lower() for c in categories]
    return {
        name: data
        for name, data in stats.items()
        if data.get("cat", "").lower() in lower_cats
    }


# ===================================================================== #
#  Pipeline breakdown (used in comparisons)                              #
# ===================================================================== #


def get_pipeline_breakdown(stats: Dict[str, Dict]) -> List[Dict]:
    """Compute a functional-pipeline breakdown for a single trace.

    Groups all kernels (events with ``cat == "kernel"``) into the
    pipelines defined in ``FUNCTIONAL_PIPELINES`` and returns per-pipeline
    totals.

    Args:
        stats: Kernel stats dict.

    Returns:
        List of dicts sorted by total time descending, each with:
        ``pipeline``, ``description``, ``time_ms``, ``calls``,
        ``unique_kernels``.
    """
    kernel_names = {n for n, d in stats.items() if d.get("cat") == "kernel"}

    assigned: Dict[str, str] = {}
    for name in kernel_names:
        nl = name.lower()
        for pipe_name, pipe_cfg in FUNCTIONAL_PIPELINES.items():
            if any(pat in nl for pat in pipe_cfg["patterns"]):
                assigned[name] = pipe_name
                break

    buckets: Dict[str, Dict[str, float]] = {}
    for pipe_name in list(FUNCTIONAL_PIPELINES.keys()) + ["other"]:
        buckets[pipe_name] = {"time_us": 0.0, "calls": 0, "kernels": 0}

    for name in kernel_names:
        pipe = assigned.get(name, "other")
        d = stats.get(name, {})
        buckets[pipe]["time_us"] += d.get("total_dur", 0)
        buckets[pipe]["calls"] += d.get("count", 0)
        if d.get("total_dur", 0) > 0:
            buckets[pipe]["kernels"] += 1

    result = []
    for pn, b in sorted(
        buckets.items(),
        key=lambda x: x[1]["time_us"],
        reverse=True,
    ):
        if b["time_us"] == 0:
            continue
        desc = FUNCTIONAL_PIPELINES.get(pn, {}).get("description", "Other kernels")
        result.append(
            {
                "pipeline": pn,
                "description": desc,
                "time_ms": round(b["time_us"] / 1e3, 1),
                "calls": int(b["calls"]),
                "unique_kernels": int(b["kernels"]),
            }
        )

    return result


# ===================================================================== #
#  Main entry point                                                      #
# ===================================================================== #


def analyze_trace(
    trace_json: dict,
    top_n: int = 20,
    category: Optional[str] = None,
) -> dict:
    """Analyze a PyTorch profiler Chrome trace JSON.

    This is the primary entry point for trace analysis. It extracts kernel
    stats, computes top kernels, category breakdown, and pipeline
    breakdown in one call.

    Args:
        trace_json: Parsed Chrome trace JSON dict (must contain
            ``traceEvents``).
        top_n: Number of top kernels to include (default 20).
        category: Optional category filter (see ``CATEGORY_GROUPS``).

    Returns:
        Dict with keys:
        - ``status``: ``"success"`` or ``"error"``
        - ``kernel_stats``: Full per-kernel statistics
        - ``top_kernels``: Top *n* kernels by total duration
        - ``category_breakdown``: Time aggregated by trace category
        - ``pipeline_breakdown``: Time aggregated by functional pipeline
        - ``summary``: Total traced time, unique operation count, etc.
    """
    try:
        stats = extract_kernel_stats(trace_json)

        if not stats:
            return {
                "status": "error",
                "message": "No complete-duration events found in trace",
            }

        filtered_stats = _filter_by_category(stats, category)
        top_kernels = get_top_kernels(filtered_stats, n=top_n)
        category_bkdn = get_category_breakdown(stats)
        pipeline_bkdn = get_pipeline_breakdown(stats)

        total_time = sum(d["total_dur"] for d in stats.values())
        filtered_time = sum(d["total_dur"] for d in filtered_stats.values())

        return {
            "status": "success",
            "kernel_stats": stats,
            "top_kernels": top_kernels,
            "category_breakdown": category_bkdn,
            "pipeline_breakdown": pipeline_bkdn,
            "summary": {
                "total_traced_time_us": total_time,
                "total_traced_time_human": _format_duration(total_time),
                "filtered_time_us": filtered_time,
                "filtered_time_human": _format_duration(filtered_time),
                "unique_operations": len(stats),
                "filtered_operations": len(filtered_stats),
                "category_filter": category or "all",
            },
        }

    except Exception as exc:
        return {"status": "error", "message": f"Failed to analyze trace: {exc}"}
