# Single-node/single-GPU image for auto-tune-vllm.
#
# Built on top of the official vllm/vllm-openai image so CUDA/torch/vLLM
# are already compiled and version-matched - we only add the optimization
# stack (Ray/Optuna) on top. Do NOT let pip re-resolve vllm/torch here:
# auto_tune_vllm is installed with --no-deps for that reason.
#
# auto_tune_vllm launches the vLLM server itself via
# `python3 -m vllm.entrypoints.openai.api_server` (see
# auto_tune_vllm/execution/trial_controller.py), not through this image's
# default entrypoint, so overriding ENTRYPOINT/CMD below is safe.
#
# ENTRYPOINT is docker/entrypoint.sh, which has two modes: with no
# arguments and TUNE_MODEL set, it renders a study config from TUNE_*
# environment variables (docker/generate_config.py) and runs it - the
# convenient path for `docker run`/Kubernetes users who don't want to hand
# -write a study config YAML. Any explicit arguments passthrough straight
# to the `auto-tune-vllm` CLI, unchanged (see docs/container.md).
ARG VLLM_IMAGE_TAG=v0.27.1
FROM docker.io/vllm/vllm-openai:${VLLM_IMAGE_TAG}

WORKDIR /opt/auto-tune-vllm

# Non-vLLM dependencies for the optimization/orchestration stack.
# vllm itself is NOT listed here - it must come from the base image.
RUN pip install --no-cache-dir \
    "ray[default]>=2.0.0" \
    "optuna>=3.0.0" \
    "optuna-integration[botorch]>=4.0.0" \
    "gpytorch>=1.1" \
    "pydantic>=2.0.0" \
    "typer>=0.9.0" \
    "rich>=13.0.0" \
    "pyyaml>=6.0" \
    "requests>=2.28.0" \
    "psycopg2-binary>=2.9.0"

COPY . /opt/auto-tune-vllm

# --no-deps: pyproject.toml declares "vllm>=0.11.0" which pip would
# otherwise try to re-resolve against, risking a torch/vllm version swap.
RUN pip install --no-cache-dir --no-deps -e /opt/auto-tune-vllm

# One-time workaround for a transformers bug hit by some model configs
# (see patch_transformers.py for details). Safe to no-op if already patched.
RUN python3 patch_transformers.py

RUN chmod +x docker/entrypoint.sh

ENV VLLM_NO_USAGE_STATS=1 \
    VLLM_DO_NOT_TRACK=1

ENTRYPOINT ["/opt/auto-tune-vllm/docker/entrypoint.sh"]
CMD []
