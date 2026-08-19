"""
Cost Efficiency Calculator

Computes Cost Per Million Tokens (CPMT) and other cost-efficiency metrics
based on benchmark throughput and accelerator pricing.

SOURCE: AI-Analysis-Agent/psap-mcp-server/psap_mcp_server/src/tools/cost_efficiency_tool.py

Converted to sync plain Python functions. No MCP, S3, async, pandas,
or logger.

Public API:
    calculate_cost(throughput: float, accelerator: str = "h100") -> dict
        Main entry point. Returns cost metrics including CPMT.

    calculate_cpmt(throughput_tok_per_sec: float, hourly_cost: float) -> float
    filter_by_slo(results: list, max_itl_p95_ms: float | None,
                  max_ttft_p95_ms: float | None) -> list
    compare_cost_efficiency(baseline_cpmt: float, current_cpmt: float) -> dict

Constants:
    ACCELERATOR_PRICING
"""

from typing import Any, Dict, List, Optional

# ===================================================================== #
#  Accelerator pricing (per hour, USD)                                   #
# ===================================================================== #
# Pricing as of October 2025. Full-instance costs for GPU accelerators;
# per-core pricing for TPUs.

ACCELERATOR_PRICING: Dict[str, Dict[str, Any]] = {
    "H200": {
        "hourly_cost": 41.62,
        "provider": "AWS",
        "instance": "p6en.48xlarge",
        "configuration": "8xNVIDIA H200-144GB",
        "gpus_per_instance": 8,
        "notes": "Full 8-GPU instance cost regardless of TP",
    },
    "MI300X": {
        "hourly_cost": 48.00,
        "provider": "Azure",
        "instance": "ND96isr_MI300X_v5",
        "configuration": "8xAMD MI300X-192GB",
        "gpus_per_instance": 8,
        "notes": "Full 8-GPU instance cost regardless of TP",
    },
    "TPU": {
        "hourly_cost": 2.70,
        "provider": "GCP",
        "instance": "TPU Trillium",
        "configuration": "Per-core pricing",
        "gpus_per_instance": 1,
        "notes": "Per-core pricing, multiply by TP count",
    },
    "H100": {
        "hourly_cost": 40.00,
        "provider": "AWS/Azure",
        "instance": "p5.48xlarge (approx)",
        "configuration": "8xNVIDIA H100-80GB",
        "gpus_per_instance": 8,
        "notes": "Full 8-GPU instance cost regardless of TP",
    },
    "A100": {
        "hourly_cost": 30.00,
        "provider": "AWS/Azure/GCP",
        "instance": "p4de.24xlarge (approx)",
        "configuration": "8xNVIDIA A100-80GB",
        "gpus_per_instance": 8,
        "notes": "Full 8-GPU instance cost regardless of TP",
    },
}


# ===================================================================== #
#  Core CPMT formula                                                     #
# ===================================================================== #


def calculate_cpmt(
    throughput_tok_per_sec: float,
    hourly_cost: float,
) -> float:
    """Compute Cost Per Million Tokens (CPMT).

    Formula::

        CPMT = (time_for_1M_tokens_in_seconds * hourly_cost) / 3600

    Which simplifies to::

        CPMT = (1_000_000 / throughput) * (hourly_cost / 3600)

    Args:
        throughput_tok_per_sec: Effective throughput in tokens per second.
        hourly_cost: Total hourly instance cost in USD.

    Returns:
        CPMT value in USD. Returns ``float('inf')`` if throughput is zero
        or negative.
    """
    if throughput_tok_per_sec <= 0:
        return float("inf")

    million_tokens = 1_000_000
    ttmt_seconds = million_tokens / throughput_tok_per_sec
    cpmt = (ttmt_seconds * hourly_cost) / 3600
    return cpmt


# ===================================================================== #
#  SLO filtering                                                         #
# ===================================================================== #


def filter_by_slo(
    results: List[Dict[str, Any]],
    max_itl_p95_ms: Optional[float] = None,
    max_ttft_p95_ms: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Filter a list of result dicts by latency SLO constraints.

    Each result dict is expected to have optional keys ``itl_p95_ms``
    and ``ttft_p95_ms``.

    Args:
        results: List of result dicts (each containing latency fields).
        max_itl_p95_ms: Maximum acceptable Inter-Token Latency P95 in
            milliseconds. Results exceeding this are filtered out.
        max_ttft_p95_ms: Maximum acceptable Time-to-First-Token P95 in
            milliseconds. Results exceeding this are filtered out.

    Returns:
        Filtered list of result dicts that meet all specified SLO
        constraints.
    """
    filtered = results

    if max_itl_p95_ms is not None:
        filtered = [
            r
            for r in filtered
            if r.get("itl_p95_ms") is None or r["itl_p95_ms"] <= max_itl_p95_ms
        ]

    if max_ttft_p95_ms is not None:
        filtered = [
            r
            for r in filtered
            if r.get("ttft_p95_ms") is None or r["ttft_p95_ms"] <= max_ttft_p95_ms
        ]

    return filtered


# ===================================================================== #
#  Cost comparison helper                                                #
# ===================================================================== #


def compare_cost_efficiency(
    baseline_cpmt: float,
    current_cpmt: float,
) -> dict:
    """Compare two CPMT values and return the percentage improvement.

    A negative CPMT change means the current configuration is cheaper
    (better).

    Args:
        baseline_cpmt: CPMT from the baseline run (USD).
        current_cpmt: CPMT from the current run (USD).

    Returns:
        Dict with ``baseline_cpmt``, ``current_cpmt``,
        ``absolute_change``, ``percent_change``, and ``direction``.
    """
    if baseline_cpmt <= 0 or baseline_cpmt == float("inf"):
        pct = 0.0
    else:
        pct = ((current_cpmt - baseline_cpmt) / baseline_cpmt) * 100.0

    abs_change = current_cpmt - baseline_cpmt

    if pct < -2:
        direction = "cheaper"
    elif pct > 2:
        direction = "more_expensive"
    else:
        direction = "similar"

    return {
        "baseline_cpmt": round(baseline_cpmt, 4),
        "current_cpmt": round(current_cpmt, 4),
        "absolute_change": round(abs_change, 4),
        "percent_change": round(pct, 2),
        "direction": direction,
    }


# ===================================================================== #
#  Internal helpers                                                      #
# ===================================================================== #


def _resolve_accelerator(name: str) -> Optional[str]:
    """Case-insensitive lookup into ACCELERATOR_PRICING.

    Returns the canonical key if found, else ``None``.
    """
    upper = name.upper().strip()
    for key in ACCELERATOR_PRICING:
        if key.upper() == upper:
            return key
    # Substring fallback (e.g. "h200" matches "H200")
    for key in ACCELERATOR_PRICING:
        if upper in key.upper() or key.upper() in upper:
            return key
    return None


def _get_hourly_cost(accelerator_key: str) -> float:
    """Return the hourly cost for a known accelerator key."""
    entry = ACCELERATOR_PRICING.get(accelerator_key, {})
    if isinstance(entry, dict):
        return entry.get("hourly_cost", 0.0)
    return float(entry)


def _get_gpus_per_instance(accelerator_key: str) -> int:
    """Return GPUs per instance for a known accelerator key."""
    entry = ACCELERATOR_PRICING.get(accelerator_key, {})
    if isinstance(entry, dict):
        return entry.get("gpus_per_instance", 8)
    return 8


# ===================================================================== #
#  Main entry point                                                      #
# ===================================================================== #


def calculate_cost(
    throughput: float,
    accelerator: str = "h100",
    tp: int = 1,
    hourly_cost_override: Optional[float] = None,
    itl_p95_ms: Optional[float] = None,
    ttft_p95_ms: Optional[float] = None,
) -> dict:
    """Calculate cost-efficiency metrics for an inference configuration.

    This is the primary entry point. Given a throughput measurement and an
    accelerator type (or custom hourly cost), it computes CPMT, time to
    million tokens, efficiency score, and embeds pricing metadata.

    For GPU accelerators (H200, MI300X, H100, A100) the calculation
    adjusts throughput to reflect the full instance: if you measure
    throughput with TP=2 on an 8-GPU box, the effective throughput is
    ``raw_throughput * (8 / 2)`` because you could run 4 independent
    replicas.

    For TPU the cost scales with the TP count instead (per-core pricing).

    Args:
        throughput: Measured output throughput in tokens per second.
        accelerator: Accelerator name (e.g. ``"h100"``, ``"H200"``,
            ``"MI300X"``). Case-insensitive.
        tp: Tensor-parallelism degree used during the benchmark
            (default 1).
        hourly_cost_override: If provided, use this hourly cost (USD)
            instead of looking up ``ACCELERATOR_PRICING``.
        itl_p95_ms: Optional measured ITL P95 latency in ms (included
            in output for SLO-aware analysis).
        ttft_p95_ms: Optional measured TTFT P95 latency in ms.

    Returns:
        Dict with keys:
        - ``status``: ``"success"`` or ``"error"``
        - ``accelerator``: Canonical accelerator name.
        - ``raw_throughput_tok_per_sec``
        - ``effective_throughput_tok_per_sec``
        - ``hourly_cost_usd``
        - ``cost_per_million_tokens_usd`` (CPMT)
        - ``time_to_million_tokens_seconds``
        - ``time_to_million_tokens_minutes``
        - ``efficiency_score`` (tokens per dollar-hour)
        - ``pricing_info``: Pricing metadata dict.
        - ``latency``: Optional latency values.
    """
    try:
        if throughput <= 0:
            return {
                "status": "error",
                "message": "Throughput must be a positive number.",
            }

        # Resolve accelerator
        acc_key = _resolve_accelerator(accelerator)
        if acc_key is None and hourly_cost_override is None:
            return {
                "status": "error",
                "message": (
                    f"Unknown accelerator '{accelerator}'. "
                    f"Available: {list(ACCELERATOR_PRICING.keys())}. "
                    "Or provide hourly_cost_override."
                ),
            }

        # Determine hourly cost
        if hourly_cost_override is not None:
            hourly_cost = hourly_cost_override
            pricing_info = {
                "source": "custom_override",
                "hourly_cost": hourly_cost,
            }
        else:
            hourly_cost = _get_hourly_cost(acc_key)
            pricing_info = dict(ACCELERATOR_PRICING.get(acc_key, {}))

        # Compute effective throughput and total hourly cost
        is_tpu = acc_key is not None and acc_key.upper() == "TPU"
        if is_tpu:
            # TPU: per-core pricing -- cost scales with TP
            total_hourly_cost = hourly_cost * max(tp, 1)
            effective_throughput = throughput
            throughput_note = "Raw throughput (per-core pricing)"
        else:
            # GPU: full-instance pricing -- throughput scales with instance
            total_hourly_cost = hourly_cost
            gpus_per_instance = _get_gpus_per_instance(acc_key) if acc_key else 8
            effective_throughput = throughput * (gpus_per_instance / max(tp, 1))
            throughput_note = (
                f"Adjusted throughput ({gpus_per_instance}-GPU instance, TP={tp})"
            )

        # CPMT
        cpmt = calculate_cpmt(effective_throughput, total_hourly_cost)

        # Time to million tokens
        million = 1_000_000
        if effective_throughput > 0:
            ttmt_sec = million / effective_throughput
            ttmt_min = ttmt_sec / 60
        else:
            ttmt_sec = float("inf")
            ttmt_min = float("inf")

        # Efficiency score (tokens per dollar-hour)
        efficiency = (
            effective_throughput / total_hourly_cost if total_hourly_cost > 0 else 0.0
        )

        result: Dict[str, Any] = {
            "status": "success",
            "accelerator": acc_key or accelerator,
            "tp": tp,
            "raw_throughput_tok_per_sec": round(throughput, 2),
            "effective_throughput_tok_per_sec": round(effective_throughput, 2),
            "throughput_note": throughput_note,
            "hourly_cost_usd": round(total_hourly_cost, 2),
            "cost_per_million_tokens_usd": (
                round(cpmt, 4) if cpmt != float("inf") else None
            ),
            "time_to_million_tokens_seconds": (
                round(ttmt_sec, 2) if ttmt_sec != float("inf") else None
            ),
            "time_to_million_tokens_minutes": (
                round(ttmt_min, 2) if ttmt_min != float("inf") else None
            ),
            "efficiency_score_tokens_per_dollar_hour": round(efficiency, 2),
            "pricing_info": pricing_info,
        }

        # Attach latency if provided
        if itl_p95_ms is not None or ttft_p95_ms is not None:
            result["latency"] = {
                "itl_p95_ms": itl_p95_ms,
                "ttft_p95_ms": ttft_p95_ms,
            }

        return result

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to calculate cost: {exc}",
        }
