#!/bin/bash
# Container entrypoint with two modes:
#
# 1. No arguments + TUNE_MODEL set: env-var-driven quick start - render a
#    study config from TUNE_* environment variables (see
#    generate_config.py / docs/container.md) and run it.
# 2. Any arguments given (or TUNE_MODEL unset): passthrough to the normal
#    `auto-tune-vllm` CLI, unchanged from running the image directly - lets
#    `docker run <image> optimize --config /path/to/study.yaml ...` keep
#    working exactly as documented.
set -euo pipefail

if [ "$#" -ne 0 ]; then
    exec auto-tune-vllm "$@"
elif [ -z "${TUNE_MODEL:-}" ]; then
    exec auto-tune-vllm --help
else
    CONFIG_PATH="${TUNE_CONFIG_OUTPUT:-/tmp/generated_study_config.yaml}"
    python3 /opt/auto-tune-vllm/docker/generate_config.py "$CONFIG_PATH"

    auto-tune-vllm validate --config "$CONFIG_PATH"

    CREATE_DB_FLAG=()
    if [ -n "${TUNE_DATABASE_URL:-}" ]; then
        # Idempotent: checks whether the database already exists before
        # creating it, so this is safe to always pass once a database URL
        # is configured.
        CREATE_DB_FLAG=(--create-db)
    fi

    exec auto-tune-vllm optimize \
        --config "$CONFIG_PATH" \
        --backend "${TUNE_BACKEND:-ray}" \
        --start-ray-head \
        --python-executable "${TUNE_PYTHON_EXECUTABLE:-/usr/bin/python3}" \
        --max-concurrent-trials "${TUNE_MAX_CONCURRENT_TRIALS:-1}" \
        "${CREATE_DB_FLAG[@]}"
fi
