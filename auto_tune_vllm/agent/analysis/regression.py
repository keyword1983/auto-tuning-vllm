"""
Performance Regression Detection

Compares benchmark results between two runs (baseline vs. current) and
detects performance regressions or improvements beyond a configurable
threshold.

SOURCE: AI-Analysis-Agent/psap-mcp-server/psap_mcp_server/src/tools/regression_analysis_tool.py

Converted to sync plain Python functions. No MCP, S3, async, pandas,
or logger.

Public API:
    detect_regression(baseline: dict, current: dict, threshold: float = 0.02) -> dict
        Main entry point. Compares two benchmark result dicts and returns
        per-metric regressions, improvements, and a summary.

Metric directionality:
    - Throughput metrics (tok/sec): higher is better.
    - Latency metrics (ttft, itl, tpot): lower is better.
"""

from typing import Any, Dict, List, Optional

# ===================================================================== #
#  Metric directionality                                                 #
# ===================================================================== #

# Keywords that indicate "higher is better" metrics.
_HIGHER_IS_BETTER_KEYWORDS = ["tok/sec", "throughput", "tokens_per_sec"]

# Keywords that indicate "lower is better" metrics.
_LOWER_IS_BETTER_KEYWORDS = [
    "ttft",
    "itl",
    "tpot",
    "latency",
    "p50",
    "p95",
    "p99",
    "median",
    "mean",
]


def _is_higher_better(metric_name: str) -> bool:
    """Return True if a higher value for *metric_name* means better performance."""
    lower = metric_name.lower()
    if any(kw in lower for kw in _HIGHER_IS_BETTER_KEYWORDS):
        return True
    if any(kw in lower for kw in _LOWER_IS_BETTER_KEYWORDS):
        return False
    # Default: assume higher is better (e.g. generic scores).
    return True


# ===================================================================== #
#  Core comparison logic                                                 #
# ===================================================================== #


def _compare_metric(
    metric_name: str,
    baseline_val: float,
    current_val: float,
    threshold: float,
) -> Dict[str, Any]:
    """Compare a single metric value between baseline and current.

    Args:
        metric_name: Name of the metric.
        baseline_val: Value from the baseline run.
        current_val: Value from the current run.
        threshold: Fractional threshold for flagging regression/improvement
            (e.g. 0.02 = 2%).

    Returns:
        Dict with comparison details including percentage change and
        direction.
    """
    if baseline_val == 0:
        pct_change = 100.0 if current_val != 0 else 0.0
    else:
        pct_change = ((current_val - baseline_val) / baseline_val) * 100.0

    abs_change = current_val - baseline_val
    higher_better = _is_higher_better(metric_name)

    # Determine direction
    if higher_better:
        if pct_change > threshold * 100:
            direction = "improvement"
        elif pct_change < -threshold * 100:
            direction = "regression"
        else:
            direction = "neutral"
    else:
        # Lower is better: a negative pct_change is improvement.
        if pct_change < -threshold * 100:
            direction = "improvement"
        elif pct_change > threshold * 100:
            direction = "regression"
        else:
            direction = "neutral"

    return {
        "metric": metric_name,
        "baseline_value": round(baseline_val, 4),
        "current_value": round(current_val, 4),
        "absolute_change": round(abs_change, 4),
        "percent_change": round(pct_change, 2),
        "direction": direction,
        "higher_is_better": higher_better,
    }


# ===================================================================== #
#  Main entry point                                                      #
# ===================================================================== #


def detect_regression(
    baseline: dict,
    current: dict,
    threshold: float = 0.02,
    metrics: Optional[List[str]] = None,
) -> dict:
    """Compare two benchmark result dicts and detect regressions.

    Both *baseline* and *current* should be flat dicts mapping metric names
    to numeric values.  For example::

        baseline = {
            "output_tok/sec": 1200.5,
            "ttft_median": 45.3,
            "ttft_p95": 78.1,
            "itl_median": 12.4,
            "itl_p95": 18.7,
        }
        current = {
            "output_tok/sec": 1150.0,
            "ttft_median": 48.9,
            "ttft_p95": 82.5,
            "itl_median": 11.8,
            "itl_p95": 19.2,
        }

    Args:
        baseline: Metric-name to value dict from the baseline run.
        current: Metric-name to value dict from the current run.
        threshold: Fractional threshold for flagging a change as
            regression or improvement.  Default ``0.02`` (2%).
        metrics: Optional explicit list of metric names to compare.
            If ``None``, all numeric keys common to both dicts are
            compared.

    Returns:
        Dict with keys:
        - ``status``: ``"success"`` or ``"error"``
        - ``threshold``: The threshold used (as a fraction).
        - ``comparisons``: List of per-metric comparison dicts.
        - ``regressions``: Subset of comparisons flagged as regressions.
        - ``improvements``: Subset of comparisons flagged as improvements.
        - ``no_change``: Subset within the neutral band.
        - ``summary``: High-level counts and overall verdict.
    """
    try:
        if not baseline or not current:
            return {
                "status": "error",
                "message": "Both baseline and current results are required.",
            }

        # Determine which metrics to compare
        if metrics is not None:
            keys_to_compare = [k for k in metrics if k in baseline and k in current]
        else:
            # Use all common numeric keys
            keys_to_compare = sorted(
                k
                for k in set(baseline.keys()) & set(current.keys())
                if isinstance(baseline[k], (int, float))
                and isinstance(current[k], (int, float))
            )

        if not keys_to_compare:
            return {
                "status": "error",
                "message": (
                    "No common numeric metrics found between baseline and "
                    "current results."
                ),
            }

        comparisons: List[Dict[str, Any]] = []
        regressions: List[Dict[str, Any]] = []
        improvements: List[Dict[str, Any]] = []
        no_change: List[Dict[str, Any]] = []

        for key in keys_to_compare:
            b_val = float(baseline[key])
            c_val = float(current[key])
            result = _compare_metric(key, b_val, c_val, threshold)
            comparisons.append(result)

            if result["direction"] == "regression":
                regressions.append(result)
            elif result["direction"] == "improvement":
                improvements.append(result)
            else:
                no_change.append(result)

        # Sort regressions by magnitude (worst first)
        regressions.sort(key=lambda x: abs(x["percent_change"]), reverse=True)
        improvements.sort(key=lambda x: abs(x["percent_change"]), reverse=True)

        # Overall verdict
        if regressions and not improvements:
            verdict = "regression"
        elif improvements and not regressions:
            verdict = "improvement"
        elif regressions and improvements:
            verdict = "mixed"
        else:
            verdict = "no_significant_change"

        total_compared = len(comparisons)
        summary = {
            "total_metrics_compared": total_compared,
            "regressions_count": len(regressions),
            "improvements_count": len(improvements),
            "no_change_count": len(no_change),
            "regression_rate": (
                round(len(regressions) / total_compared * 100, 2)
                if total_compared > 0
                else 0
            ),
            "improvement_rate": (
                round(len(improvements) / total_compared * 100, 2)
                if total_compared > 0
                else 0
            ),
            "verdict": verdict,
        }

        return {
            "status": "success",
            "threshold": threshold,
            "comparisons": comparisons,
            "regressions": regressions,
            "improvements": improvements,
            "no_change": no_change,
            "summary": summary,
            "message": (
                f"Compared {total_compared} metrics: "
                f"{len(regressions)} regressions, "
                f"{len(improvements)} improvements, "
                f"{len(no_change)} unchanged "
                f"(threshold={threshold * 100:.1f}%)"
            ),
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to detect regressions: {exc}",
        }
