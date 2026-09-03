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

# Named workload shapes for TUNE_WORKLOAD, so a study can be pointed at a
# traffic shape by name instead of hand-picking token counts. "long" has no
# fixed prompt_tokens - it's derived from TUNE_MAX_MODEL_LEN so the prompt
# actually fills the model's context window instead of an arbitrary number.
WORKLOAD_PROFILES = {
    "short": {"prompt_tokens": 100, "output_tokens": 100},
    "chat": {"prompt_tokens": 200, "output_tokens": 250},
    "long": {"output_tokens": 2000},  # prompt_tokens = max_model_len - output_tokens
}


def resolve_prompt_output_tokens():
    """TUNE_PROMPT_TOKENS/TUNE_OUTPUT_TOKENS win if set; otherwise derive
    from TUNE_WORKLOAD; otherwise fall back to the 1000/1000 default."""
    workload = env("TUNE_WORKLOAD")
    prompt_tokens = env("TUNE_PROMPT_TOKENS")
    output_tokens = env("TUNE_OUTPUT_TOKENS")

    if workload:
        if workload not in WORKLOAD_PROFILES:
            print(
                f"Unknown TUNE_WORKLOAD '{workload}'. Valid options: "
                f"{', '.join(WORKLOAD_PROFILES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        profile = WORKLOAD_PROFILES[workload]

        if output_tokens is None:
            output_tokens = profile.get("output_tokens", 1000)

        if prompt_tokens is None:
            if "prompt_tokens" in profile:
                prompt_tokens = profile["prompt_tokens"]
            else:
                max_model_len = env("TUNE_MAX_MODEL_LEN")
                if not max_model_len:
                    print(
                        f"TUNE_WORKLOAD={workload} derives prompt_tokens from "
                        f"TUNE_MAX_MODEL_LEN - {output_tokens} (to fill the "
                        f"model's context window) - set TUNE_MAX_MODEL_LEN, "
                        f"or set TUNE_PROMPT_TOKENS explicitly instead.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                prompt_tokens = int(max_model_len) - int(output_tokens)
                if prompt_tokens <= 0:
                    print(
                        f"TUNE_MAX_MODEL_LEN ({max_model_len}) is too small for "
                        f"the '{workload}' workload's output_tokens ({output_tokens}).",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    return (
        int(prompt_tokens) if prompt_tokens is not None else 1000,
        int(output_tokens) if output_tokens is not None else 1000,
    )


def build_config():
    model = env("TUNE_MODEL")
    if not model:
        print("TUNE_MODEL is required (HF repo id or local model path)", file=sys.stderr)
        sys.exit(1)

    prompt_tokens, output_tokens = resolve_prompt_output_tokens()

    config = {
        "study": {},
        "optimization": {
            "preset": env("TUNE_PRESET", "high_throughput"),
            # "tpe" has no extra dependency and is the documented
            # general-purpose default; "botorch" needs a botorch build new
            # enough for optuna-integration's BoTorchSampler, which isn't
            # guaranteed given this image intentionally keeps torch pinned
            # to whatever vllm/vllm-openai shipped with.
            "sampler": env("TUNE_SAMPLER", "tpe"),
            "n_trials": int(env("TUNE_N_TRIALS", 50)),
            "max_concurrent_trials": int(env("TUNE_MAX_CONCURRENT_TRIALS", 1)),
        },
        "benchmark": {
            "benchmark_type": "vllmbench",
            "model": model,
            "dataset": None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            # Defaults to 1 (--max-concurrency) so a first, unconfigured
            # run stays cheap - baseline trials run at this concurrency
            # before any optimization trial does, so a high default here
            # means every run starts with a heavy load test whether the
            # caller wanted one or not. Raise it once you actually want a
            # throughput/concurrency stress test.
            "rate": int(env("TUNE_RATE", 1)),
            "samples": int(env("TUNE_SAMPLES", 100)),
            # Open-loop arrival rate (requests/sec), or "inf" (vLLM's own
            # default) to submit everything at time 0. "rate" above still
            # caps concurrent in-flight requests on top of this - raise it
            # if you set a real TUNE_REQUEST_RATE, or it'll clip the
            # arrival pattern back into a closed-loop test.
            "request_rate": env("TUNE_REQUEST_RATE", "inf"),
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
        workload = env("TUNE_WORKLOAD")
        default_prefix = f"auto_tune_{workload}" if workload else "auto_tune"
        config["study"]["prefix"] = env("TUNE_STUDY_PREFIX", default_prefix)

    # Without this, trial results and logs only exist inside the
    # container's ephemeral filesystem - gone the moment the pod finishes
    # and its process exits (kubectl exec/cp only work against a running
    # container, so there's no grabbing them afterward). Point both the
    # Optuna study and the trial logger at the same Postgres instance so
    # results survive the pod regardless of how/when it exits.
    database_url = env("TUNE_DATABASE_URL")
    if database_url:
        config["study"]["database_url"] = database_url
        config["logging"]["database_url"] = database_url

    # n_startup_trials must be < n_trials (the sampler needs at least one
    # non-random trial) - only set it if the caller asked for it, so small
    # smoke-test runs (n_trials < the library default of 10) don't fail
    # validation just because we always emitted a value here.
    n_startup_trials = env("TUNE_N_STARTUP_TRIALS")
    if n_startup_trials:
        config["optimization"]["n_startup_trials"] = int(n_startup_trials)

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
