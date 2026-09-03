#!/usr/bin/env python3
"""Render a study config YAML from a small set of TUNE_* environment
variables, for the container's env-var-driven quick start (see
docker/entrypoint.sh and docs/container.md).

Only TUNE_MODEL is required. Everything else falls back to a sensible
default, and per-parameter env vars are only added to the search space if
the user actually set them - unset parameters use vLLM's own defaults, same
as leaving them out of a hand-written config file.
"""

import os
import sys

import yaml


def env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _num(token):
    token = token.strip()
    return float(token) if "." in token else int(token)


def parse_range(spec):
    """"0.85-0.95" -> {min, max}; "2048-32768:2048" -> {min, max, step}."""
    if ":" in spec:
        bounds, step = spec.split(":", 1)
    else:
        bounds, step = spec, None
    lo, hi = bounds.split("-", 1)
    result = {"min": _num(lo), "max": _num(hi)}
    if step is not None:
        result["step"] = _num(step)
    return result


def parse_list(spec):
    """"auto,fp8" -> ["auto", "fp8"]; casts numeric-looking entries."""
    values = []
    for token in spec.split(","):
        token = token.strip()
        try:
            values.append(_num(token))
        except ValueError:
            values.append(token)
    return values


# Range-style tunables: TUNE_<NAME>="min-max" or "min-max:step"
RANGE_PARAMS = {
    "TUNE_GPU_MEMORY_UTILIZATION": "gpu_memory_utilization",
    "TUNE_MAX_NUM_BATCHED_TOKENS": "max_num_batched_tokens",
    "TUNE_MAX_NUM_SEQS": "max_num_seqs",
    "TUNE_CUDA_GRAPH_SIZES": "cuda_graph_sizes",
    "TUNE_MAX_SEQ_LEN_TO_CAPTURE": "max_seq_len_to_capture",
}

# List-style (categorical) tunables: TUNE_<NAME>="opt1,opt2"
LIST_PARAMS = {
    "TUNE_KV_CACHE_DTYPE": "kv_cache_dtype",
    "TUNE_BLOCK_SIZE": "block_size",
}

DEFAULT_PARAMETERS = {
    "gpu_memory_utilization": {"enabled": True, "min": 0.85, "max": 0.95, "step": 0.01},
    "max_num_batched_tokens": {"enabled": True, "min": 2048, "max": 32768, "step": 2048},
    "kv_cache_dtype": {"enabled": True, "options": ["auto", "fp8"]},
}


def build_config():
    model = env("TUNE_MODEL")
    if not model:
        print("TUNE_MODEL is required (HF repo id or local model path)", file=sys.stderr)
        sys.exit(1)

    config = {
        "study": {},
        "optimization": {
            "preset": env("TUNE_PRESET", "high_throughput"),
            "sampler": env("TUNE_SAMPLER", "botorch"),
            "n_trials": int(env("TUNE_N_TRIALS", 50)),
            "max_concurrent_trials": int(env("TUNE_MAX_CONCURRENT_TRIALS", 1)),
        },
        "benchmark": {
            "benchmark_type": "vllmbench",
            "model": model,
            "dataset": None,
            "prompt_tokens": int(env("TUNE_PROMPT_TOKENS", 1000)),
            "output_tokens": int(env("TUNE_OUTPUT_TOKENS", 1000)),
            "rate": int(env("TUNE_RATE", 30)),
            "samples": int(env("TUNE_SAMPLES", 100)),
        },
        "logging": {
            "file_path": env("TUNE_LOG_PATH", "/tmp/auto-tune-vllm-local-run/logs"),
            "log_level": env("TUNE_LOG_LEVEL", "INFO"),
        },
        "parameters": {},
    }

    study_name = env("TUNE_STUDY_NAME")
    if study_name:
        config["study"]["name"] = study_name
    else:
        config["study"]["prefix"] = env("TUNE_STUDY_PREFIX", "auto_tune")

    for env_name, param_name in RANGE_PARAMS.items():
        value = env(env_name)
        if value:
            config["parameters"][param_name] = {"enabled": True, **parse_range(value)}

    for env_name, param_name in LIST_PARAMS.items():
        value = env(env_name)
        if value:
            config["parameters"][param_name] = {"enabled": True, "options": parse_list(value)}

    if not config["parameters"]:
        config["parameters"] = DEFAULT_PARAMETERS

    static_parameters = {}
    tensor_parallel_size = env("TUNE_TENSOR_PARALLEL_SIZE")
    if tensor_parallel_size:
        static_parameters["tensor_parallel_size"] = int(tensor_parallel_size)
    max_model_len = env("TUNE_MAX_MODEL_LEN")
    if max_model_len:
        static_parameters["max_model_len"] = int(max_model_len)
    if static_parameters:
        config["static_parameters"] = static_parameters

    return config


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/generated_study_config.yaml"
    rendered = yaml.dump(build_config(), sort_keys=False)
    with open(output_path, "w") as f:
        f.write(rendered)
    print(f"Generated study config at {output_path}:\n{rendered}", file=sys.stderr)
