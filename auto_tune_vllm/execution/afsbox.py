"""AFSBox Kubernetes execution backend.

Orchestrates trials by updating AFSBox ModelServing experimental instances
and triggering AFSBox Benchmark (AIPerf) jobs via Kubernetes Custom Resources.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..core.trial import ExecutionInfo, TrialConfig, TrialResult
from .backends import ExecutionBackend, JobHandle

logger = logging.getLogger(__name__)

AFSBOX_GROUP = "afsbox.asus.com"
AFSBOX_VERSION = "v1beta1"
PLURAL_SERVINGS = "modelservings"
PLURAL_BENCHMARKS = "benchmarks"
PLURAL_TUNINGS = "modeltunings"


class AFSBoxK8sBackend(ExecutionBackend):
    """Execution backend interfacing with AFSBox Kubernetes CRDs (ModelServing and Benchmark)."""

    def __init__(
        self,
        tuning_name: Optional[str] = None,
        namespace: str = "default",
        deploy_timeout_seconds: int = 1800,
        poll_interval_seconds: int = 10,
    ):
        """Initialize AFSBox Kubernetes execution backend.

        Args:
            tuning_name: Name of the parent ModelTuning CR (if run as part of a ModelTuning session).
            namespace: Kubernetes namespace where ModelServing and Benchmark CRs reside.
            deploy_timeout_seconds: Max seconds to wait for ModelServing to become Ready after patch.
            poll_interval_seconds: Interval between status polls.
        """
        self.tuning_name = tuning_name
        self.namespace = namespace
        self.deploy_timeout_seconds = deploy_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.active_trials: Dict[str, Dict[str, Any]] = {}

        # Lazy load kubernetes client to avoid import errors when running in environments without it
        try:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            except Exception:
                config.load_kube_config()
                logger.info("Loaded local kube-config")

            self.custom_api = client.CustomObjectsApi()
        except ImportError:
            raise RuntimeError(
                "The 'kubernetes' package is required for the AFSBox backend. "
                "Please install it via 'pip install kubernetes'."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kubernetes client for AFSBox backend: {e}")

    def _get_experiment_serving_name(self) -> str:
        """Get the deterministic experiment serving name."""
        if self.tuning_name:
            return f"{self.tuning_name}-exp"
        return "optuna-tune-exp"

    def _map_parameters_to_serving_patch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map Optuna parameter dictionary to AFSBox ModelServing spec patch.

        Handles parallelism (TP/PP), memory utilization, batch size, context length,
        and generic CLI extraArgs.
        """
        spec_patch: Dict[str, Any] = {}
        parallelism: Dict[str, Any] = {}
        extra_args: List[str] = []

        for k, v in params.items():
            k_lower = k.lower().replace("-", "_")

            # Typed top-level fields
            if k_lower in ("batchsize", "batch_size", "max_num_seqs"):
                spec_patch["batchSize"] = str(v)
            elif k_lower in ("contextlength", "context_length", "max_model_len"):
                spec_patch["contextLength"] = str(v)
            elif k_lower in ("replicas", "replica_count"):
                spec_patch["replicas"] = int(v)

            # Parallelism
            elif k_lower in ("tp", "tensor_parallel_size"):
                parallelism["tp"] = int(v)
            elif k_lower in ("pp", "pipeline_parallel_size"):
                parallelism["pp"] = int(v)
            elif k_lower in ("dp", "data_parallel_size"):
                parallelism["dp"] = int(v)

            # GPU & Memory
            elif k_lower in ("gpu_memory_utilization", "gpu_mem_util"):
                extra_args.append(f"--gpu-memory-utilization={v}")
            elif k_lower in ("max_num_batched_tokens",):
                extra_args.append(f"--max-num-batched-tokens={v}")
            elif k_lower in ("kv_cache_dtype",):
                spec_patch["kvCacheDtype"] = str(v)
            elif k_lower in ("enable_cuda_graphs", "cuda_graph"):
                if isinstance(v, bool):
                    spec_patch["cudaGraph"] = {"enabled": v}

            # Direct values / params dot-path
            elif k.startswith("values.") or k.startswith("params."):
                extra_args.append(f"--{k}={v}")
            else:
                extra_args.append(f"--{k.replace('_', '-')}={v}")

        if parallelism:
            spec_patch["parallelism"] = parallelism

        if extra_args:
            spec_patch["extraArgs"] = extra_args

        return spec_patch

    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit a trial to AFSBox by updating ModelServing and triggering a Benchmark CR."""
        serving_name = self._get_experiment_serving_name()
        trial_id = trial_config.trial_id
        trial_num = trial_config.trial_number if trial_config.trial_number is not None else 0
        bench_name = f"{self.tuning_name or 'optuna'}-c{trial_num}"

        logger.info(
            "Submitting trial %s (trial #%d) to AFSBox ModelServing %s",
            trial_id,
            trial_num,
            serving_name,
        )

        exec_info = ExecutionInfo()
        exec_info.mark_vllm_started()

        # 1. Update ModelServing with candidate parameters
        spec_patch = self._map_parameters_to_serving_patch(trial_config.parameters)
        try:
            # Patch experiment ModelServing spec
            body = {"spec": spec_patch}
            self.custom_api.patch_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_SERVINGS,
                name=serving_name,
                body=body,
            )
            logger.info("Patched ModelServing %s with spec: %s", serving_name, spec_patch)
        except Exception as e:
            logger.error("Failed to patch ModelServing %s: %s", serving_name, e)
            raise RuntimeError(f"AFSBox ModelServing patch failed: {e}")

        # 2. Wait for ModelServing to become Ready
        start_wait = time.time()
        is_ready = False
        while time.time() - start_wait < self.deploy_timeout_seconds:
            try:
                serving_obj = self.custom_api.get_namespaced_custom_object(
                    group=AFSBOX_GROUP,
                    version=AFSBOX_VERSION,
                    namespace=self.namespace,
                    plural=PLURAL_SERVINGS,
                    name=serving_name,
                )
                status = serving_obj.get("status", {})
                phase = status.get("phase", "")
                if phase == "Ready":
                    is_ready = True
                    break
                elif phase == "Failed":
                    raise RuntimeError(
                        f"ModelServing {serving_name} failed: {status.get('message', 'Unknown error')}"
                    )
            except Exception as e:
                logger.warning("Checking ModelServing %s status warning: %s", serving_name, e)

            time.sleep(self.poll_interval_seconds)

        if not is_ready:
            raise TimeoutError(
                f"Timed out waiting {self.deploy_timeout_seconds}s for ModelServing {serving_name} to become Ready"
            )

        exec_info.mark_vllm_ready()
        logger.info("ModelServing %s is Ready. Launching Benchmark CR %s...", serving_name, bench_name)

        # 3. Create Benchmark CR to initiate AIPerf load test
        exec_info.mark_benchmark_started()
        bench_body = {
            "apiVersion": f"{AFSBOX_GROUP}/{AFSBOX_VERSION}",
            "kind": "Benchmark",
            "metadata": {
                "name": bench_name,
                "namespace": self.namespace,
                "labels": {
                    "afsbox.asus.com/benchmark-serving": serving_name,
                    "afsbox.asus.com/benchmark-candidate": trial_id,
                },
            },
            "spec": {
                "displayName": f"Optuna / {trial_id}",
                "target": {
                    "modelServingRef": {"name": serving_name},
                },
                "suite": [
                    {
                        "name": "perf-eval",
                        "params": {
                            "requestCount": 100,
                            "concurrency": 8,
                        },
                    }
                ],
            },
        }

        if self.tuning_name:
            bench_body["metadata"]["labels"]["afsbox.asus.com/model-tuning"] = self.tuning_name

        try:
            self.custom_api.create_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_BENCHMARKS,
                body=bench_body,
            )
            logger.info("Created Benchmark CR %s", bench_name)
        except Exception as e:
            # If already exists from previous attempt, ignore conflict
            if "AlreadyExists" not in str(e):
                logger.error("Failed to create Benchmark CR %s: %s", bench_name, e)
                raise RuntimeError(f"AFSBox Benchmark CR creation failed: {e}")

        handle = JobHandle(trial_id=trial_id, backend_job_id=bench_name, status="running")
        self.active_trials[trial_id] = {
            "handle": handle,
            "bench_name": bench_name,
            "exec_info": exec_info,
            "trial_config": trial_config,
        }
        return handle

    def poll_trials(
        self, job_handles: List[JobHandle]
    ) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll active Benchmark CRs and collect results."""
        completed_results: List[TrialResult] = []
        remaining_handles: List[JobHandle] = []

        for handle in job_handles:
            trial_id = handle.trial_id
            trial_info = self.active_trials.get(trial_id)
            if not trial_info:
                continue

            bench_name = trial_info["bench_name"]
            exec_info: ExecutionInfo = trial_info["exec_info"]
            trial_config: TrialConfig = trial_info["trial_config"]

            try:
                bench_obj = self.custom_api.get_namespaced_custom_object(
                    group=AFSBOX_GROUP,
                    version=AFSBOX_VERSION,
                    namespace=self.namespace,
                    plural=PLURAL_BENCHMARKS,
                    name=bench_name,
                )
                status = bench_obj.get("status", {})
                phase = status.get("phase", "")

                if phase == "Completed":
                    exec_info.mark_benchmark_completed()
                    exec_info.mark_completed("success")

                    # Extract metrics from results
                    results_list = status.get("results", [])
                    detailed_metrics = {}
                    if results_list and "metrics" in results_list[0]:
                        detailed_metrics = results_list[0]["metrics"]

                    # Extract objective values
                    objective_values = self._extract_objectives(
                        detailed_metrics, trial_config.optimization_config
                    )

                    result = TrialResult(
                        trial_id=trial_id,
                        trial_number=trial_config.trial_number,
                        trial_type=trial_config.trial_type,
                        objective_values=objective_values,
                        detailed_metrics=detailed_metrics,
                        execution_info=exec_info,
                        success=True,
                    )
                    completed_results.append(result)
                    logger.info("Trial %s Benchmark %s completed: %s", trial_id, bench_name, objective_values)

                elif phase == "Failed":
                    exec_info.mark_completed("failed")
                    result = TrialResult(
                        trial_id=trial_id,
                        trial_number=trial_config.trial_number,
                        trial_type=trial_config.trial_type,
                        objective_values=[],
                        execution_info=exec_info,
                        success=False,
                        error_message=status.get("message", "Benchmark failed"),
                    )
                    completed_results.append(result)
                    logger.warning("Trial %s Benchmark %s failed: %s", trial_id, bench_name, status.get("message"))

                else:
                    remaining_handles.append(handle)

            except Exception as e:
                logger.error("Error polling Benchmark %s: %s", bench_name, e)
                remaining_handles.append(handle)

        return completed_results, remaining_handles

    def _extract_objectives(
        self, metrics: Dict[str, Any], optimization_config: Any
    ) -> List[float]:
        """Extract objective values based on configured optimization targets."""
        values: List[float] = []

        if not optimization_config or not hasattr(optimization_config, "objectives"):
            # Default fallback: maximize throughput
            tps = metrics.get("output_tokens_per_sec_per_user") or metrics.get("output_token_throughput") or 0.0
            return [float(tps)]

        for obj in optimization_config.objectives:
            m_name = obj.metric.lower()
            if "token" in m_name and "second" in m_name:
                v = metrics.get("output_tokens_per_sec_per_user") or metrics.get("output_token_throughput") or 0.0
                values.append(float(v))
            elif "first_token" in m_name or "ttft" in m_name:
                percentile = getattr(obj, "percentile", "p50")
                key = f"ttft_{percentile}"
                v = metrics.get(key) or metrics.get("ttft_p50") or 0.0
                values.append(float(v))
            elif "inter_token" in m_name or "itl" in m_name:
                percentile = getattr(obj, "percentile", "p50")
                key = f"itl_{percentile}"
                v = metrics.get(key) or metrics.get("itl_p50") or 0.0
                values.append(float(v))
            elif "latency" in m_name or "e2e" in m_name:
                percentile = getattr(obj, "percentile", "p50")
                key = f"e2e_{percentile}"
                v = metrics.get(key) or metrics.get("e2e_p50") or 0.0
                values.append(float(v))
            else:
                v = metrics.get(obj.metric, 0.0)
                values.append(float(v))

        return values

    def shutdown(self):
        """Clean shutdown of backend resources."""
        logger.info("AFSBoxK8sBackend shutdown completed.")

    def cleanup_all_trials(self):
        """Clean up active benchmarks."""
        logger.info("Cleaning up active trials for AFSBox backend.")
        self.active_trials.clear()
