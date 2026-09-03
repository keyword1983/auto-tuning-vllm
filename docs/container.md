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
| `TUNE_RATE` | `benchmark.rate` (`--max-concurrency`, closed-loop cap) | `"inf"` (no cap - omits the flag, same as vLLM's own default) |
| `TUNE_REQUEST_RATE` | `benchmark.request_rate` (`--request-rate`, open-loop arrival rate, req/s) | `"inf"` (vLLM's own default - submit everything at time 0) |
| `TUNE_SAMPLES` | `benchmark.samples` (`--num-prompts`) | `100` (or the `TUNE_WORKLOAD` profile's value below, if set) |
| `TUNE_WORKLOAD` | selects a named traffic shape (below) - sets `prompt_tokens`/`output_tokens`/`samples` for you | unset |
| `TUNE_PROMPT_TOKENS` / `TUNE_OUTPUT_TOKENS` | `benchmark.prompt_tokens` / `output_tokens` - overrides `TUNE_WORKLOAD` if both are set | `1000` / `1000` |
| `TUNE_STUDY_NAME` / `TUNE_STUDY_PREFIX` | `study.name` / `study.prefix` | auto-generated prefix `auto_tune` (or `auto_tune_<workload>` if `TUNE_WORKLOAD` is set) |
| `TUNE_LOG_PATH` / `TUNE_LOG_LEVEL` | `logging.file_path` / `log_level` | `/tmp/auto-tune-vllm-local-run/logs` / `INFO` |
| `TUNE_DATABASE_URL` | `study.database_url` **and** `logging.database_url` | unset (SQLite + local log files) |
| `TUNE_TENSOR_PARALLEL_SIZE` / `TUNE_MAX_MODEL_LEN` | `static_parameters.*` | unset (vLLM default) |
| `TUNE_BACKEND` / `TUNE_PYTHON_EXECUTABLE` | `optimize` CLI flags | `ray` / `/usr/bin/python3` |
| `TUNE_CONFIG_OUTPUT` | where the rendered YAML is written | `/tmp/generated_study_config.yaml` |

### `TUNE_RATE` vs `TUNE_REQUEST_RATE`: closed-loop vs open-loop load

These control two different things and can fight each other if set
carelessly:

- **`TUNE_RATE`** (`--max-concurrency`): a hard cap on concurrent in-flight
  requests. Closed-loop - the benchmark client won't submit request N+1
  until one of the current N finishes.
- **`TUNE_REQUEST_RATE`** (`--request-rate`): requests/sec at which new
  requests *arrive*, independent of whether earlier ones finished yet.
  Open-loop - defaults to `"inf"` (everything arrives at time 0).

Picking a benchmark concurrency isn't just about avoiding "too low to see
any batching effect" - it's also easy to end up testing a number with no
real reference point. Finding your server's own saturation point (e.g. by
sweeping `baseline.concurrency_levels`) sounds like a way to pick one, but
it's circular: the very parameters you're searching (`max_num_seqs`,
`max_num_batched_tokens`, ...) change where that saturation point is, so a
"peak" measured against vLLM's defaults doesn't mean much once you start
testing other parameter combinations.

The way out is to anchor on something that *doesn't* depend on the
parameters under test: your actual (or target) production arrival rate.
That's what `TUNE_REQUEST_RATE` is for - set it to requests/sec matching
your real traffic, leave `TUNE_RATE` high enough that it doesn't clip the
arrival pattern back into an artificial closed-loop cap, and let each
parameter combination's queueing/latency behavior emerge naturally. A
config that handles your real arrival rate with low queueing is
genuinely better than one that doesn't - that comparison isn't circular,
because the arrival rate itself never changes between trials.

If you don't know your production arrival rate, `TUNE_RATE` alone
(closed-loop, `TUNE_REQUEST_RATE` left at `"inf"`) is a reasonable
fallback - just don't mistake "high concurrency" for "correct answer";
it's a *deliberately chosen* stress level, not a discovered one.

### `TUNE_DATABASE_URL`: making results survive the pod

Without this, trial results (Optuna study: parameters, objective values)
live in a local SQLite file and logs live under `TUNE_LOG_PATH` - both
inside the container's ephemeral filesystem. `kubectl exec`/`kubectl cp`
only work against a *running* container, and this entrypoint `exec`s
straight into `auto-tune-vllm optimize`, so the moment the run finishes (or
the pod is deleted, evicted, or crashes) that process exits and there is no
way to retrieve anything left behind - `kubectl logs` still works (pod
stdout is retained separately by the kubelet), but the study database and
log files are gone.

Set `TUNE_DATABASE_URL` to a PostgreSQL connection string
(`postgresql://user:pass@host:5432/dbname`) and both the Optuna study and
the trial logger write there in real time instead - safe against the pod
disappearing at any point, and queryable while the run is still in
progress. `--create-db` is passed automatically (idempotent - it only
creates the database if it doesn't already exist) so a fresh database name
doesn't need to be provisioned by hand first.

```bash
docker run --rm --gpus all \
  -e TUNE_MODEL="facebook/opt-125m" \
  -e TUNE_DATABASE_URL="postgresql://user:pass@postgres.example.com:5432/auto_tune_vllm" \
  auto-tune-vllm:v0.27.1
```

If you don't set this, treat the run as disposable: only pull results out
(`kubectl cp` the SQLite file and log directory) *before* the run finishes,
not after - or better, mount a PersistentVolume at the study/log paths
instead of relying on `kubectl cp` timing at all.

### `TUNE_WORKLOAD`: named traffic shapes

Optimal vLLM parameters differ by traffic shape - a config tuned on short
prompts won't necessarily be optimal for long-context RAG traffic. Rather
than picking `prompt_tokens`/`output_tokens` by hand, set `TUNE_WORKLOAD` to
one of:

| `TUNE_WORKLOAD` | `prompt_tokens` | `output_tokens` | `samples` | Represents |
|---|---|---|---|---|
| `short` | `100` | `100` | `128` | Short-form: classification, function calling |
| `chat` | `200` | `250` | `128` | Conversational turns |
| `long` | `TUNE_MAX_MODEL_LEN - 2000` | `2000` | `5` | Long-context: RAG, summarization - requires `TUNE_MAX_MODEL_LEN` so the prompt actually fills the model's context window |

`long`'s `samples` default is deliberately tiny: each request there is far
more expensive (long prefill + long decode) than `short`/`chat`'s, so 128
requests at that length would take drastically longer for comparatively
little extra statistical stability.

Explicit `TUNE_PROMPT_TOKENS`/`TUNE_OUTPUT_TOKENS`/`TUNE_SAMPLES` override
the profile's corresponding value if set. The study name prefix also picks
up the workload (`auto_tune_chat_...`) so studies for different shapes
don't collide.

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
end-to-end through the `vllmbench` provider - including a full 30-trial
env-var-driven optimization run (`TUNE_WORKLOAD=chat`, `TUNE_RATE=10`,
searching `gpu_memory_utilization`/`max_num_batched_tokens`/`max_num_seqs`
against `facebook/opt-125m`) that found a real improvement over the vLLM-
defaults baseline:

```
Best Value      5811.31 tok/s   (baseline: 5597.19 tok/s, +3.8%)
Best Trial      #23 of 30
gpu_memory_utilization  0.912
max_num_batched_tokens  5120
max_num_seqs            24
```

Along the way this surfaced and fixed three real bugs (see commit history):
`_validate_environment()` unconditionally requiring `guidellm` regardless of
the configured provider, an indentation bug in `patch_transformers.py` that
crashed transformers' import for every model, and `optuna-integration`'s
`BoTorchSampler` failing against the image's intentionally-old pinned
`botorch` (hence `tpe` as the default sampler above). It also surfaced the
`kubectl exec`/`cp`-vs-ephemeral-filesystem problem that motivated
`TUNE_DATABASE_URL`.
