# Auto-Tune Serving

[![PyPI version](https://badge.fury.io/py/auto-tune-serving.svg)](https://badge.fury.io/py/auto-tune-serving)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A distributed hyperparameter optimization framework for LLM serving engines (**vLLM**, **SGLang**, **llama.cpp**), built with Ray, Optuna, and AFSBox orchestration.

> *Note: Formerly `auto-tune-vllm`. The CLI command `auto-tune-vllm` is retained as a backwards-compatible alias for `auto-tune-serving`.*

## Features

- 🚀 **Distributed Optimization**: Scale across multiple GPUs and nodes using Ray
- 🎯 **Multi-Engine Support**: Tune vLLM, SGLang, and llama.cpp with unified abstractions
- ☸️ **AFSBox & Kubernetes Native**: Direct CRD lifecycle management (`ModelTuning`, `ModelServing`, `Benchmark`)
- 📊 **Rich Benchmarking**: Built-in vLLM benchmark, GuideLLM, and AIPerf integration
- 🗄️ **Centralized Storage**: PostgreSQL for trials, metrics, and logs
- ⚙️ **Dual Execution Modes**: Local study YAML configuration or autonomous AFSBox CR-driven mode (`--tuning-name`)
- 📈 **Multi-Objective**: Pareto optimization for throughput vs TTFT / latency trade-offs
- 🔧 **Extensible**: Plugin system for custom benchmarks and serving engines

## Quick Start (5 minutes)
For a detailed starter guide, see the [Quick Start Guide](docs/quick_start.md) and [AFSBox Integration Guide](docs/afsbox_integration.md).

### Installation

```bash
git clone https://github.com/ocisd4/auto-tune-serving.git
cd auto-tune-serving
pip install -e .
```

### Basic Usage

#### CLI Interface
```bash
# Run optimization study with auto-tune-serving (or auto-tune-vllm)
auto-tune-serving optimize --config config.yaml --max-concurrent-trials 2

# Autonomous AFSBox CR-driven optimization mode
auto-tune-serving optimize --backend afsbox --tuning-name <tuning-cr-name> --tuning-namespace afsbox-tokenfactory

# Stream live logs  
auto-tune-serving logs --study-id 42 --trial-number 15

# Resume interrupted study
auto-tune-serving resume --study-name study_35884
```

## Documentation

- [Ray Cluster Setup](docs/ray_cluster_setup.md) - **Important for distributed optimization**
- [Configuration Reference](docs/configuration.md)
- [Container Image](docs/container.md) - Building and running as a Docker image

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- PostgreSQL database

All ML dependencies (vLLM, Ray, GuideLLM, BoTorch) are included automatically.

## Known Issues

### Ray Cluster Concurrency Validation

**Issue**: The `--max-concurrent-trials` parameter is not validated against available Ray cluster resources.

**Details**: When using Ray backend, the system doesn't check if the requested concurrency level is feasible given the cluster's GPU/CPU resources. For example, setting `--max-concurrent-trials 10` on a cluster with only 4 GPUs will not warn the user that only 4 trials can actually run concurrently.

**Reason**: There is not a clear answer if all the trials would use the exact same number of GPUs. For example, we might have different parallelism related tunings for different trials which might result in different number of GPUs being required for the trial.

**Current Behavior**: 
- Excess trials are queued by Ray until resources become available
- No warning or guidance is provided to users
- May lead to confusion about why trials aren't running as expected

**Workaround**: 
- Use `auto-tune-serving check-env --ray-cluster` to inspect available resources
- Set concurrency based on available GPUs (typically 1 GPU per trial)
- Monitor Ray dashboard at `http://<head-node>:8265` for resource utilization

**Example**:
```bash
# Check cluster resources first
auto-tune-serving check-env --ray-cluster

# Set realistic concurrency (e.g., if you have 4 GPUs)
auto-tune-serving optimize --config study.yaml --max-concurrent-trials 4
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.