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


LABEL_SERVING = "afsbox.asus.com/serving"
LABEL_TUNING = "afsbox.asus.com/model-tuning"
LABEL_CANDIDATE = "afsbox.asus.com/candidate"


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
        self._cached_tuning_test_suite: Optional[List[Dict[str, Any]]] = None

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

    def _get_parent_tuning_suite(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch testSuite from parent ModelTuning CR if available."""
        if not self.tuning_name:
            return None
        if self._cached_tuning_test_suite is not None:
            return self._cached_tuning_test_suite
        try:
            tuning_obj = self.custom_api.get_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
            )
            suite = tuning_obj.get("spec", {}).get("testSuite")
            if suite:
                self._cached_tuning_test_suite = suite
                logger.info("Cached parent ModelTuning %s testSuite (%d items)", self.tuning_name, len(suite))
                return suite
        except Exception as e:
            logger.warning("Could not fetch testSuite from ModelTuning %s: %s", self.tuning_name, e)
        return None

    def _build_benchmark_suite(self, trial_config: TrialConfig) -> List[Dict[str, Any]]:
        """Construct Benchmark CR suite from ModelTuning CR or trial_config.benchmark_config."""
        parent_suite = self._get_parent_tuning_suite()
        if parent_suite:
            logger.info("Using testSuite inherited from ModelTuning CR %s", self.tuning_name)
            return parent_suite

        bc = trial_config.benchmark_config
        if not bc:
            return [
                {
                    "name": "perf-eval",
                    "type": "concurrency",
                    "timeoutSeconds": 300,
                    "params": {
                        "concurrency": 8,
                        "requestCount": 100,
                        "streaming": True,
                        "ignoreEOS": True,
                    },
                }
            ]

        # Determine benchmark item parameters
        concurrency = bc.rate or bc.concurrency or 8
        request_count = bc.samples or 100
        timeout_seconds = bc.max_seconds or 300

        params: Dict[str, Any] = {
            "streaming": True,
            "ignoreEOS": True,
        }

        # Input & output token distribution
        if bc.prompt_tokens:
            params["isl"] = {
                "mean": int(bc.prompt_tokens),
                "stddev": int(bc.prompt_tokens_stdev or 0),
            }
        if bc.output_tokens:
            params["osl"] = {
                "mean": int(bc.output_tokens),
                "stddev": int(bc.output_tokens_stdev or 0),
            }

        # Dataset replay vs request-rate vs fixed concurrency
        if bc.dataset and bc.dataset.lower() in ("sharegpt", "sonnet"):
            suite_type = "dataset-replay"
            params["dataset"] = bc.dataset.lower()
            params["concurrency"] = int(concurrency)
            params["requestCount"] = int(request_count)
        elif bc.request_rate and str(bc.request_rate).lower() != "inf":
            suite_type = "request-rate"
            params["requestRate"] = str(bc.request_rate)
            params["concurrency"] = int(concurrency)
            params["requestCount"] = int(request_count)
        elif concurrency == 1:
            suite_type = "latency"
            params["requestCount"] = int(request_count)
        else:
            suite_type = "concurrency"
            params["concurrency"] = int(concurrency)
            params["requestCount"] = int(request_count)

        return [
            {
                "name": "perf-eval",
                "type": suite_type,
                "timeoutSeconds": int(timeout_seconds),
                "params": params,
            }
        ]

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
                spec_patch["gpuMemoryUtilization"] = str(v)
            elif k_lower in ("max_num_batched_tokens",):
                if "prefillSettings" not in spec_patch:
                    spec_patch["prefillSettings"] = {}
                spec_patch["prefillSettings"]["maxBatchTokens"] = str(v)
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
            spec_patch["extraCommand"] = extra_args

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
            patched_obj = self.custom_api.patch_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_SERVINGS,
                name=serving_name,
                body=body,
            )
            target_gen = patched_obj.get("metadata", {}).get("generation", 0)
            logger.info(
                "Patched ModelServing %s with spec: %s (target generation: %s)",
                serving_name,
                spec_patch,
                target_gen,
            )
        except Exception as e:
            logger.error("Failed to patch ModelServing %s: %s", serving_name, e)
            raise RuntimeError(f"AFSBox ModelServing patch failed: {e}")

        # 2. Wait for ModelServing to become Ready
        time.sleep(2)
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
                observed_gen = status.get("observedGeneration", 0)
                if phase == "Ready" and observed_gen >= target_gen:
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
        suite = self._build_benchmark_suite(trial_config)
        endpoint_model = (
            trial_config.benchmark_config.model
            if trial_config.benchmark_config and trial_config.benchmark_config.model
            else "afsbox/opt-125m"
        )
        bench_body = {
            "apiVersion": f"{AFSBOX_GROUP}/{AFSBOX_VERSION}",
            "kind": "Benchmark",
            "metadata": {
                "name": bench_name,
                "namespace": self.namespace,
                "labels": {
                    LABEL_SERVING: serving_name,
                    LABEL_CANDIDATE: trial_id,
                },
            },
            "spec": {
                "displayName": f"Optuna / {trial_id}",
                "target": {
                    "modelServingRef": {"name": serving_name},
                    "endpoint": {
                        "url": f"http://{serving_name}.{self.namespace}.svc.cluster.local:8000/v1",
                        "modelName": endpoint_model,
                    },
                },
                "suite": suite,
            },
        }

        if self.tuning_name:
            bench_body["metadata"]["labels"][LABEL_TUNING] = self.tuning_name

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

                    # Extract metrics from results or BenchmarkReport
                    results_list = status.get("results", [])
                    detailed_metrics = {}
                    if results_list and "metrics" in results_list[0]:
                        detailed_metrics = results_list[0]["metrics"]

                    report_ref = status.get("reportRef", {}).get("name")
                    if not detailed_metrics and report_ref:
                        try:
                            report_obj = self.custom_api.get_namespaced_custom_object(
                                group=AFSBOX_GROUP,
                                version=AFSBOX_VERSION,
                                namespace=self.namespace,
                                plural="benchmarkreports",
                                name=report_ref,
                            )
                            items = report_obj.get("spec", {}).get("items", [])
                            if items and "metrics" in items[0]:
                                detailed_metrics = items[0]["metrics"]
                        except Exception as err:
                            logger.warning("Failed to fetch BenchmarkReport %s: %s", report_ref, err)

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
                    self._sync_candidate_status_to_tuning(trial_id, bench_name, "Completed")
                    logger.info("Trial %s Benchmark %s completed: %s", trial_id, bench_name, objective_values)

                elif phase == "Failed":
                    exec_info.mark_completed("failed")
                    err_msg = status.get("message", "Benchmark failed")
                    result = TrialResult(
                        trial_id=trial_id,
                        trial_number=trial_config.trial_number,
                        trial_type=trial_config.trial_type,
                        objective_values=[],
                        execution_info=exec_info,
                        success=False,
                        error_message=err_msg,
                    )
                    completed_results.append(result)
                    self._sync_candidate_status_to_tuning(trial_id, bench_name, "Failed", err_msg)
                    logger.warning("Trial %s Benchmark %s failed: %s", trial_id, bench_name, err_msg)

                else:
                    remaining_handles.append(handle)

            except Exception as e:
                logger.error("Error polling Benchmark %s: %s", bench_name, e)
                remaining_handles.append(handle)

        return completed_results, remaining_handles

    def _sync_candidate_status_to_tuning(
        self, candidate_name: str, bench_name: str, phase: str, message: str = ""
    ):
        """Sync trial candidate progress back to parent ModelTuning.status.candidates."""
        if not self.tuning_name:
            return
        try:
            tuning_obj = self.custom_api.get_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
            )
            status = tuning_obj.get("status", {})
            candidates = status.get("candidates", [])

            found = False
            for c in candidates:
                if c.get("name") == candidate_name:
                    c["phase"] = phase
                    c["benchmarkRef"] = bench_name
                    if message:
                        c["message"] = message
                    found = True
                    break
            if not found:
                candidates.append({
                    "name": candidate_name,
                    "benchmarkRef": bench_name,
                    "phase": phase,
                    "message": message,
                })

            status["candidates"] = candidates
            status["currentCandidate"] = candidate_name
            self.custom_api.patch_namespaced_custom_object_status(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
                body={"status": status},
            )
        except Exception as e:
            logger.debug("Failed to sync candidate status to ModelTuning %s: %s", self.tuning_name, e)

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
                v = (
                    metrics.get("outputTokensPerSec")
                    or metrics.get("outputTokensPerSecPerUser")
                    or metrics.get("output_tokens_per_sec_per_user")
                    or metrics.get("output_token_throughput")
                    or 0.0
                )
                values.append(float(v))
            elif "first_token" in m_name or "ttft" in m_name:
                percentile = getattr(obj, "percentile", "p50").lower()
                ttft_data = metrics.get("ttft")
                if isinstance(ttft_data, dict):
                    v = ttft_data.get(percentile) or ttft_data.get("p50") or ttft_data.get("avg") or 0.0
                else:
                    v = metrics.get(f"ttft_{percentile}") or metrics.get("ttft_p50") or 0.0
                values.append(float(v))
            elif "inter_token" in m_name or "itl" in m_name:
                percentile = getattr(obj, "percentile", "p50").lower()
                itl_data = metrics.get("itl")
                if isinstance(itl_data, dict):
                    v = itl_data.get(percentile) or itl_data.get("p50") or itl_data.get("avg") or 0.0
                else:
                    v = metrics.get(f"itl_{percentile}") or metrics.get("itl_p50") or 0.0
                values.append(float(v))
            elif "latency" in m_name or "e2e" in m_name:
                percentile = getattr(obj, "percentile", "p50").lower()
                e2e_data = metrics.get("e2e")
                if isinstance(e2e_data, dict):
                    v = e2e_data.get(percentile) or e2e_data.get("p50") or e2e_data.get("avg") or 0.0
                else:
                    v = metrics.get(f"e2e_{percentile}") or metrics.get("e2e_p50") or 0.0
                values.append(float(v))
            else:
                v = metrics.get(obj.metric, 0.0)
                values.append(float(v))

        return values

    def sync_final_results_to_tuning(self, results: Dict[str, Any]):
        """Sync final best candidate and pareto frontier to ModelTuning.status."""
        if not self.tuning_name:
            return
        try:
            tuning_obj = self.custom_api.get_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
            )
            status = tuning_obj.get("status", {})
            if results.get("type") == "multi_objective":
                pareto_candidates = []
                for p in results.get("pareto_front", []):
                    trial_num = p.get("trial")
                    if trial_num is not None:
                        pareto_candidates.append(f"trial_{trial_num}")
                status["paretoFrontier"] = pareto_candidates
                if pareto_candidates:
                    status["bestCandidate"] = pareto_candidates[0]
            else:
                best_num = results.get("best_trial_number")
                if best_num is not None:
                    status["bestCandidate"] = f"trial_{best_num}"

            status["currentCandidate"] = ""
            self.custom_api.patch_namespaced_custom_object_status(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
                body={"status": status},
            )
            logger.info(
                "Synced final results to ModelTuning %s status: best=%s, pareto=%s",
                self.tuning_name,
                status.get("bestCandidate"),
                status.get("paretoFrontier"),
            )
        except Exception as e:
            logger.warning("Failed to sync final results to ModelTuning %s: %s", self.tuning_name, e)

    def shutdown(self):
        """Clean shutdown of backend resources."""
        logger.info("AFSBoxK8sBackend shutdown completed.")

    def cleanup_all_trials(self):
        """Clean up active benchmarks."""
        logger.info("Cleaning up active trials for AFSBox backend.")
        self.active_trials.clear()


def synthesize_study_config_from_cr(tuning_name: str, namespace: str = "default") -> str:
    """Synthesize a study configuration YAML file from parent ModelTuning CR."""
    import tempfile
    import yaml
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    custom_api = client.CustomObjectsApi()
    tuning_obj = custom_api.get_namespaced_custom_object(
        group=AFSBOX_GROUP,
        version=AFSBOX_VERSION,
        namespace=namespace,
        plural=PLURAL_TUNINGS,
        name=tuning_name,
    )
    spec = tuning_obj.get("spec", {})
    opt_spec = spec.get("optimization", {}) or {}
    test_suite = spec.get("testSuite", [])

    objectives = []
    for obj in opt_spec.get("objectives", []):
        objectives.append({
            "metric": obj.get("metric", "output_tokens_per_second"),
            "direction": obj.get("direction", "maximize"),
            "percentile": obj.get("percentile", "p50"),
        })
    if not objectives:
        objectives = [
            {"metric": "output_tokens_per_second", "direction": "maximize"},
            {"metric": "time_to_first_token_ms", "direction": "minimize"},
        ]

    suite_params = test_suite[0].get("params", {}) if test_suite else {}
    isl = suite_params.get("isl", {})
    osl = suite_params.get("osl", {})
    bench_dict = {
        "benchmark_type": "aiperf",
        "model": spec.get("servingTemplate", {}).get("model", {}).get("uri", "default"),
        "samples": suite_params.get("requestCount", 100),
        "rate": suite_params.get("concurrency", 8),
        "prompt_tokens": isl.get("mean", 1000) if isinstance(isl, dict) else 1000,
        "output_tokens": osl.get("mean", 1000) if isinstance(osl, dict) else 1000,
        "dataset": suite_params.get("dataset"),
        "max_seconds": test_suite[0].get("timeoutSeconds", 300) if test_suite else 300,
    }

    params_dict = {
        "tensor_parallel_size": {
            "enabled": True,
            "options": [1, 2],
        },
        "max_num_batched_tokens": {
            "enabled": True,
            "options": [2048, 4096, 8192],
        },
        "gpu_memory_utilization": {
            "enabled": True,
            "min": 0.8,
            "max": 0.95,
            "step": 0.05,
        },
    }

    study_dict = {
        "study": {
            "name": tuning_name,
        },
        "backend": "afsbox",
        "afsbox": {
            "namespace": namespace,
            "tuning_name": tuning_name,
        },
        "optimization": {
            "approach": "multi_objective" if len(objectives) > 1 else "single_objective",
            "objectives": objectives,
            "sampler": opt_spec.get("sampler", "nsga2" if len(objectives) > 1 else "tpe"),
            "n_trials": opt_spec.get("nTrials", 20),
            "max_concurrent_trials": 1,
        },
        "benchmark": bench_dict,
        "parameters": params_dict,
    }

    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=f"_{tuning_name}.yaml", delete=False)
    yaml.safe_dump(study_dict, tmp_file, sort_keys=False)
    tmp_file.close()
    logger.info("Synthesized study configuration from ModelTuning CR: %s", tmp_file.name)
    return tmp_file.name

