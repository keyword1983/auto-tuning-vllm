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
