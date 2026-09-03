"""Benchmark provider backed by vLLM's built-in `vllm bench serve` CLI.

Replaces the GuideLLM provider so that the benchmarking tool always ships
with the same version as the vLLM server under test, instead of pinning a
separate `guidellm` release whose CLI surface can drift out from under us
(the `guidellm benchmark --rate-type ...` syntax used by
:class:`auto_tune_vllm.benchmarks.providers.GuideLLMBenchmark` broke against
newer guidellm releases).

Key semantic differences from GuideLLMBenchmark, verified against
`vllm bench serve --help=options` on vllm 0.26.0/0.27.1:

- `vllm bench serve` is num-prompts bound, not duration bound: there is no
  `--max-seconds` equivalent, so `BenchmarkConfig.max_seconds` is ignored
  here. Use `BenchmarkConfig.samples` (--num-prompts) to size the run.
- GuideLLM's `--rate-type concurrent --rate N` is a closed-loop concurrency
  gate, so `BenchmarkConfig.rate` is mapped to `--max-concurrency` - except
  when `rate` is the string `"inf"`, in which case the flag is omitted
  entirely (vLLM's own default) so vLLM's own scheduler decides how much
  to batch, unconstrained by an arbitrary client-side ceiling.
  `BenchmarkConfig.request_rate` maps separately to `--request-rate` (the
  open-loop arrival rate, in requests/sec) and also defaults to `"inf"` -
  vLLM's own default, meaning every request is submitted at time 0. Note
  that `--max-concurrency`, when set to a real number, still caps
  concurrent in-flight requests on top of whatever `--request-rate` is
  doing - if you set `request_rate` to a real arrival rate to test
  open-loop behavior, make sure `rate` isn't left low enough to clip it
  back into an artificial closed-loop test.
- Throughput metrics (`request_throughput` / `output_throughput`) are single
  aggregate numbers with no percentile breakdown, unlike GuideLLM's output.
  The same value is reported for every percentile key so objective lookups
  never KeyError.
- `request_latency` is reported in seconds (matching GuideLLM's convention)
  by converting vLLM's `e2el` (end-to-end latency), which is in ms.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from ..config import BenchmarkConfig
from ..providers import BenchmarkProvider

# vLLM's percentile-metric attribute names -> GuideLLM-style base metric name.
# "tpot" (time per output token) has no GuideLLM equivalent in
# core.config.ObjectiveConfig.valid_metrics, so it isn't mapped here, but the
# raw mean/median/percentile values are still copied into detailed_metrics
# under their vLLM names for visibility.
_LATENCY_METRIC_MAP = {
    "ttft": ("time_to_first_token_ms", 1.0),
    "itl": ("inter_token_latency_ms", 1.0),
    "e2el": ("request_latency", 1.0 / 1000.0),  # ms -> seconds
}
_PERCENTILES = ["50", "90", "95", "99"]


class VllmbenchBenchmark(BenchmarkProvider):
    """Benchmark provider that shells out to `vllm bench serve`."""

    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        self._logger.info(f"Starting vllm bench serve for {config.model}")

        self._results_file = self._get_results_file_path()
        cmd = self._build_command(model_url, config, self._results_file)

        if shutil.which("vllm") is None:
            raise RuntimeError(
                "vllm CLI not found on PATH. `vllm bench serve` requires "
                "the vllm package to be installed."
            )
        if not (
            model_url.startswith("http://") or model_url.startswith("https://")
        ):
            raise ValueError(f"Invalid model_url: {model_url!r} (expected http/https)")

        self._logger.info(f"Running: {' '.join(cmd)}")
        self._logger.info(f"Results will be saved to: {self._results_file}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        self._process_pid = self._process.pid
        try:
            self._process_pgid = os.getpgid(self._process_pid)
        except (OSError, ProcessLookupError):
            self._logger.warning(
                f"Failed to get process group for vllm bench serve process "
                f"{self._process_pid}"
            )
            self._process_pgid = None

        return self._process

    def parse_results(self) -> Dict[str, Any]:
        results_file = self._results_file

        if not os.path.exists(results_file):
            raise RuntimeError(f"vllm bench serve results file not found: {results_file}")

        try:
            with open(results_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in results file: {e}")

        return self._parse_vllm_bench_results(data)

    def _get_results_file_path(self) -> str:
        """Mirrors GuideLLMBenchmark's permanent results file layout."""
        if self._trial_context is None:
            self._logger.warning("No trial context set, using temporary file")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name

        try:
            study_name = self._trial_context["study_name"]
            trial_id = self._trial_context["trial_id"]

            base_dir = Path("/tmp/auto-tune-vllm-local-run/logs")
            benchmark_dir = base_dir / study_name / "benchmark_results"
            benchmark_dir.mkdir(parents=True, exist_ok=True)

            permanent_file = benchmark_dir / f"{trial_id}_benchmark_results.json"
            return str(permanent_file)
        except Exception as e:
            self._logger.warning(
                f"Failed to create permanent results path: {e}, using temporary file"
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name

    def _build_command(
        self, model_url: str, config: BenchmarkConfig, results_file: str
    ) -> list[str]:
        """Build the `vllm bench serve` argument list."""
        base_url = model_url[:-3] if model_url.endswith("/v1") else model_url
        results_path = Path(results_file)

        cmd = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "openai",
            "--base-url",
            base_url,
            "--endpoint",
            "/v1/completions",
            "--model",
            config.model,
            "--num-prompts",
            str(config.samples),
            "--request-rate",
            str(config.request_rate),
            "--save-result",
            "--result-dir",
            str(results_path.parent),
            "--result-filename",
            results_path.name,
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            ",".join(_PERCENTILES),
        ]

        # "inf" means no client-side concurrency cap - vLLM's own scheduler
        # (bounded by whatever gpu_memory_utilization/max_num_seqs/
        # max_num_batched_tokens this trial is testing) decides how much it
        # actually batches, instead of an arbitrary client-side ceiling.
        # vllm bench serve's own default for --max-concurrency is "no cap"
        # (the flag is simply omitted), so we mirror that instead of passing
        # a literal "inf" the CLI wouldn't accept as a concurrency count.
        if str(config.rate).strip().lower() != "inf":
            cmd.extend(["--max-concurrency", str(config.rate)])

        if config.processor:
            cmd.extend(["--tokenizer", config.processor])

        if config.use_synthetic_data:
            cmd.extend(
                [
                    "--dataset-name",
                    "random",
                    "--input-len",
                    str(config.prompt_tokens),
                    "--output-len",
                    str(config.output_tokens),
                ]
            )
        elif config.dataset.startswith("hf://"):
            cmd.extend(["--dataset-name", "hf", "--dataset-path", config.dataset[5:]])
        else:
            if not os.path.exists(config.dataset):
                raise FileNotFoundError(f"Dataset file not found: {config.dataset}")
            # Best-effort: vllm bench serve has no generic "local file" dataset
            # loader equivalent to GuideLLM's --data-type file. "custom" expects
            # a specific JSONL shape (see vLLM benchmark dataset docs) - verify
            # against your dataset before relying on this path.
            cmd.extend(["--dataset-name", "custom", "--dataset-path", config.dataset])

        return cmd

    def _parse_vllm_bench_results(self, data: dict) -> Dict[str, Any]:
        """Map `vllm bench serve` JSON output to GuideLLM-compatible keys."""
        result: Dict[str, Any] = {}

        try:
            request_throughput = data["request_throughput"]
            output_throughput = data["output_throughput"]
        except KeyError as e:
            raise RuntimeError(f"Missing required field in vllm bench serve results: {e}")

        # Single aggregate numbers - no percentile breakdown available, so
        # every percentile/mean variant resolves to the same value.
        for base_name, value in (
            ("requests_per_second", request_throughput),
            ("output_tokens_per_second", output_throughput),
        ):
            result[base_name] = value
            result[f"{base_name}_mean"] = value
            for p in _PERCENTILES:
                result[f"{base_name}_p{p}"] = value

        for vllm_attr, (base_name, scale) in _LATENCY_METRIC_MAP.items():
            median_key = f"median_{vllm_attr}_ms"
            if median_key not in data:
                raise RuntimeError(
                    f"Required metric '{median_key}' not found in vllm bench "
                    f"serve results. Ensure --percentile-metrics includes "
                    f"'{vllm_attr}'."
                )

            result[base_name] = data[median_key] * scale

            mean_key = f"mean_{vllm_attr}_ms"
            if mean_key in data:
                result[f"{base_name}_mean"] = data[mean_key] * scale

            for p in _PERCENTILES:
                p_key = f"p{p}_{vllm_attr}_ms"
                if p_key in data:
                    result[f"{base_name}_p{p}"] = data[p_key] * scale

        # Keep raw vLLM fields around for debugging/analysis without
        # overwriting the GuideLLM-compatible keys above.
        for passthrough in (
            "duration",
            "completed",
            "failed",
            "total_input_tokens",
            "total_output_tokens",
            "total_token_throughput",
            "max_output_tokens_per_s",
            "max_concurrent_requests",
        ):
            if passthrough in data:
                result[f"vllm_bench_{passthrough}"] = data[passthrough]

        return result
