# Container Image

This guide covers building and running auto-tune-vllm as a container image for
single-node/single-GPU use, and the `vllmbench` benchmark provider that makes
this practical to package.

## Why a new benchmark provider

The `guidellm` provider (`auto_tune_vllm/benchmarks/providers.py`) shells out
to `guidellm benchmark --rate-type concurrent ...`. That CLI shape has
changed across `guidellm` releases, so pinning `guidellm>=0.1.0` as a hard
dependency means a container build can silently pick up a version whose CLI
no longer matches what the provider calls.

`vllm bench serve` avoids this problem entirely: it ships inside the `vllm`
package itself, so it's always the exact version paired with the server
under test. `auto_tune_vllm/benchmarks/custom/vllmbench.py` implements
`VllmbenchBenchmark` on top of it, and it's now the default
(`benchmark_type: "vllmbench"` in `BenchmarkConfig`). `guidellm` is still
available as an optional provider - install it explicitly with
`pip install auto-tune-vllm[guidellm]` and set `benchmark_type: "guidellm"`
if you need it.

### Semantic differences from GuideLLM to be aware of

- **Duration vs. count-bound**: `vllm bench serve` has no `--max-seconds`
  equivalent - it runs until `samples` (`--num-prompts`) requests complete.
  `BenchmarkConfig.max_seconds` is ignored by this provider.
- **Concurrency**: GuideLLM's `--rate-type concurrent --rate N` is a
  closed-loop concurrency gate, so `rate` is mapped to `--max-concurrency`,
  not `--request-rate` (which is an open-loop arrival rate).
- **Throughput metrics have no percentile breakdown**: `requests_per_second`
  and `output_tokens_per_second` are single aggregate numbers in vLLM's
  output. Every percentile/mean variant resolves to the same value so
  objective lookups never fail.
- **Units**: `request_latency` is converted from vLLM's `e2el` (ms) to
  seconds, matching GuideLLM's convention.

See `examples/study_config_vllmbench.yaml` for a working example.

## Building the image

```bash
docker build -t auto-tune-vllm:v0.27.1 .
```

The `Dockerfile` builds on top of `vllm/vllm-openai:v0.27.1` so CUDA/torch/
vLLM stay a known-good, version-matched combination - only the optimization
stack (Ray, Optuna, BoTorch) is layered on top. `auto_tune_vllm` is installed
with `pip install --no-deps` so pip never tries to re-resolve `vllm`/`torch`
against `pyproject.toml`'s looser `vllm>=0.11.0` constraint.

To use a different vLLM release, override the base image tag:

```bash
docker build --build-arg VLLM_IMAGE_TAG=v0.28.0 -t auto-tune-vllm:v0.28.0 .
```

## Quick start: env-var-driven config (no YAML required)

Hand-writing a study config YAML is overkill if you just want to tune the
handful of parameters that actually matter for your model. The image's
entrypoint (`docker/entrypoint.sh`) has a second mode: run it with **no
arguments** and set `TUNE_MODEL` (plus whichever `TUNE_*` variables you
care about), and it renders a full study config
(`docker/generate_config.py`), validates it, and runs it - no config file
to write or mount.

```bash
docker run --rm --gpus all \
  -e TUNE_MODEL="facebook/opt-125m" \
  -e TUNE_PRESET="high_throughput" \
  -e TUNE_N_TRIALS="50" \
  -e TUNE_GPU_MEMORY_UTILIZATION="0.85-0.95:0.01" \
  -e TUNE_MAX_NUM_BATCHED_TOKENS="2048-32768:2048" \
  -e TUNE_KV_CACHE_DTYPE="auto,fp8" \
  -e TUNE_TENSOR_PARALLEL_SIZE="1" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  auto-tune-vllm:v0.27.1
```

Running the image with **no arguments and no `TUNE_MODEL`** prints
`auto-tune-vllm --help` instead of erroring - and passing explicit
arguments (as in [Running locally](#running-locally) below) always
passes through to the CLI unchanged, so both usage styles work with the
same image.

### `TUNE_*` variables

| Variable | Maps to | Default |
|---|---|---|
| `TUNE_MODEL` | `benchmark.model` | **required** |
| `TUNE_PRESET` | `optimization.preset` | `high_throughput` |
| `TUNE_SAMPLER` | `optimization.sampler` | `tpe` |
| `TUNE_N_TRIALS` | `optimization.n_trials` | `50` |
| `TUNE_MAX_CONCURRENT_TRIALS` | `optimization.max_concurrent_trials` and `--max-concurrent-trials` | `1` |
| `TUNE_RATE` | `benchmark.rate` (`--max-concurrency`) | `1` (baseline trials run at this concurrency too - raise it deliberately, not by accident) |
| `TUNE_SAMPLES` | `benchmark.samples` (`--num-prompts`) | `100` |
| `TUNE_WORKLOAD` | selects a named traffic shape (below) - sets `prompt_tokens`/`output_tokens` for you | unset |
| `TUNE_PROMPT_TOKENS` / `TUNE_OUTPUT_TOKENS` | `benchmark.prompt_tokens` / `output_tokens` - overrides `TUNE_WORKLOAD` if both are set | `1000` / `1000` |
| `TUNE_STUDY_NAME` / `TUNE_STUDY_PREFIX` | `study.name` / `study.prefix` | auto-generated prefix `auto_tune` (or `auto_tune_<workload>` if `TUNE_WORKLOAD` is set) |
| `TUNE_LOG_PATH` / `TUNE_LOG_LEVEL` | `logging.file_path` / `log_level` | `/tmp/auto-tune-vllm-local-run/logs` / `INFO` |
| `TUNE_TENSOR_PARALLEL_SIZE` / `TUNE_MAX_MODEL_LEN` | `static_parameters.*` | unset (vLLM default) |
| `TUNE_BACKEND` / `TUNE_PYTHON_EXECUTABLE` | `optimize` CLI flags | `ray` / `/usr/bin/python3` |
| `TUNE_CONFIG_OUTPUT` | where the rendered YAML is written | `/tmp/generated_study_config.yaml` |

### `TUNE_WORKLOAD`: named traffic shapes

Optimal vLLM parameters differ by traffic shape - a config tuned on short
prompts won't necessarily be optimal for long-context RAG traffic. Rather
than picking `prompt_tokens`/`output_tokens` by hand, set `TUNE_WORKLOAD` to
one of:

| `TUNE_WORKLOAD` | `prompt_tokens` | `output_tokens` | Represents |
|---|---|---|---|
| `short` | `100` | `100` | Short-form: classification, function calling |
| `chat` | `200` | `250` | Conversational turns |
| `long` | `TUNE_MAX_MODEL_LEN - 2000` | `2000` | Long-context: RAG, summarization - requires `TUNE_MAX_MODEL_LEN` so the prompt actually fills the model's context window |

Explicit `TUNE_PROMPT_TOKENS`/`TUNE_OUTPUT_TOKENS` override the profile if
both are set. The study name prefix also picks up the workload
(`auto_tune_chat_...`) so studies for different shapes don't collide.

Each `TUNE_WORKLOAD` is a **separate study** with its own fixed
prompt/output length - a single study still only optimizes against one
traffic shape at a time. To compare shapes, run the container three times
(once per workload) and compare the resulting best parameters: if they
converge, one shared config is fine; if they diverge significantly (common
between short prefill-bound traffic and long/high-concurrency
batching-bound traffic), pick the shape matching your dominant production
traffic, or run separate tuned deployments per traffic shape.

Per-parameter tunables are opt-in: a parameter is only added to the search
space if you set its variable, so unset ones use vLLM's own defaults -
same as leaving them out of a hand-written config.

**Range parameters** (`"min-max"` or `"min-max:step"`):

| Variable | Parameter |
|---|---|
| `TUNE_GPU_MEMORY_UTILIZATION` | `gpu_memory_utilization` |
| `TUNE_MAX_NUM_BATCHED_TOKENS` | `max_num_batched_tokens` |
| `TUNE_MAX_NUM_SEQS` | `max_num_seqs` |
| `TUNE_CUDA_GRAPH_SIZES` | `cuda_graph_sizes` |
| `TUNE_MAX_SEQ_LEN_TO_CAPTURE` | `max_seq_len_to_capture` |

**List parameters** (comma-separated, e.g. `"auto,fp8"`):

| Variable | Parameter |
|---|---|
| `TUNE_KV_CACHE_DTYPE` | `kv_cache_dtype` |
| `TUNE_BLOCK_SIZE` | `block_size` |

If you don't set *any* `TUNE_*` parameter variable, `generate_config.py`
falls back to a small default search space (`gpu_memory_utilization`,
`max_num_batched_tokens`, `kv_cache_dtype`) so you still get a useful run.

Need a parameter that isn't in this list, or a multi-objective/custom
`objectives` setup? Fall back to a hand-written config and the normal CLI
passthrough mode below - see the [Configuration Reference](configuration.md)
for the full schema. A ready-to-adapt Kubernetes `Job` using this quick
start is in [`examples/k8s-job-env-driven.yaml`](../examples/k8s-job-env-driven.yaml).

## Running locally

```bash
docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)/examples:/config" \
  -v /tmp/auto-tune-vllm-run:/tmp/auto-tune-vllm-local-run \
  auto-tune-vllm:v0.27.1 \
  optimize --config /config/study_config_vllmbench.yaml \
           --backend ray --start-ray-head \
           --python-executable /usr/bin/python3 \
           --max-concurrent-trials 1
```

Notes:

- The CLI's `optimize` command only supports `--backend ray` (see
  `auto_tune_vllm/cli/main.py`) - it always needs one of
  `--python-executable`, `--venv-path`, or `--conda-env`. Inside the
  container, `/usr/bin/python3` is the right value: there's only one Python
  environment, so Ray workers can't pick up a mismatched one - the
  environment-mismatch problems described in
  [Ray Cluster Setup](ray_cluster_setup.md) don't apply here.
- Mount a Hugging Face cache volume so models aren't re-downloaded on every
  run, and set `HF_TOKEN` for gated models.

## Running on Kubernetes

A minimal single-GPU pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: auto-tune-vllm
spec:
  runtimeClassName: nvidia   # only if your cluster requires it
  containers:
    - name: auto-tune-vllm
      image: auto-tune-vllm:v0.27.1
      command: ["auto-tune-vllm"]
      args: ["optimize", "--config", "/config/study.yaml",
             "--backend", "ray", "--start-ray-head",
             "--python-executable", "/usr/bin/python3",
             "--max-concurrent-trials", "1"]
      resources:
        limits: { nvidia.com/gpu: 1 }
        requests: { nvidia.com/gpu: 1 }
      volumeMounts:
        - { name: shm, mountPath: /dev/shm }
        - { name: config, mountPath: /config }
  volumes:
    - name: shm
      emptyDir: { medium: Memory, sizeLimit: 4Gi }
    - name: config
      configMap: { name: auto-tune-vllm-study-config }
```

`/dev/shm` needs a real size limit - the default Kubernetes allocation
(commonly 64Mi) is too small for Ray's object store and vLLM will warn about
falling back to `/tmp`, which hurts performance.

If your cluster's GPU nodes carry a taint (e.g. `nvidia.com/gpu:NoSchedule`),
add a matching toleration. If pods on your cluster don't have outbound
internet access, mount a local model cache (NFS, PVC, etc.) at
`/root/.cache/huggingface` or point `--model` at a local path instead of a
Hugging Face repo ID.

## Verified

Built and pushed to a registry, deployed as a pod on a real GPU node, and run
end-to-end with `facebook/opt-125m` through the `vllmbench` provider -
produced a real `output_tokens_per_second` objective value. See commit
history for the two bugs this surfaced and fixed along the way
(`_validate_environment()` unconditionally requiring `guidellm`, and an
indentation bug in `patch_transformers.py`).
