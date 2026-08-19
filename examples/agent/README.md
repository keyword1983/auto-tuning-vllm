# Agentic tuner examples

See [docs/agentic_tuning.md](../docs/agentic_tuning.md) for the full guide.

`experiment-pod.yaml` is the template cloned for each tuning attempt. The
agent appends extra vLLM CLI args to the container `args` list, waits for
readiness, port-forwards, benchmarks, then deletes the pod.

Edit before use:

- container image
- `--model` path (must match what is on the PVC)
- `claimName` for model storage
- GPU request/limit
