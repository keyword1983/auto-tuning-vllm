"""Engine adapter for llama.cpp."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .base import EngineAdapter


class LlamaCppEngineAdapter(EngineAdapter):
    """Adapter for llama.cpp serving engine."""

    name: str = "llamacpp"
    default_port: int = 8080
    openai_compatible: bool = True

    def get_default_parameter_space(self) -> Dict[str, Any]:
        """Return default search space for llama.cpp."""
        return {
            "batch_size": {
                "enabled": True,
                "options": [2, 4, 8, 16],
            },
            "threads": {
                "enabled": True,
                "options": [4, 8, 16],
            },
            "ubatch_size": {
                "enabled": False,
                "options": [128, 256, 512],
            },
        }

    def extract_baseline_parameters(self, serving_template: Dict[str, Any]) -> Dict[str, Any]:
        """Extract baseline parameters from serving template."""
        baseline: Dict[str, Any] = {}
        if "batchSize" in serving_template:
            try:
                baseline["batch_size"] = int(serving_template["batchSize"])
            except (ValueError, TypeError):
                pass
        if "contextLength" in serving_template:
            try:
                baseline["context_length"] = int(serving_template["contextLength"])
            except (ValueError, TypeError):
                pass

        # Check extraCommand for llama.cpp specific flags
        for cmd in serving_template.get("extraCommand", []):
            if cmd.startswith("--threads=") or cmd.startswith("-t="):
                try:
                    baseline["threads"] = int(cmd.split("=")[-1].strip())
                except ValueError:
                    pass
            elif cmd.startswith("--ubatch-size=") or cmd.startswith("-ub="):
                try:
                    baseline["ubatch_size"] = int(cmd.split("=")[-1].strip())
                except ValueError:
                    pass
            elif cmd.startswith("--n-gpu-layers=") or cmd.startswith("-ngl="):
                try:
                    baseline["n_gpu_layers"] = int(cmd.split("=")[-1].strip())
                except ValueError:
                    pass

        return baseline

    def map_to_serving_patch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map trial parameters to AFSBox ModelServing patch for llama.cpp."""
        spec_patch: Dict[str, Any] = {}
        extra_args: List[str] = []

        for k, v in params.items():
            k_lower = k.lower().replace("-", "_")

            # Top-level AFSBox CR fields: controller translates batchSize to --parallel
            if k_lower in ("batchsize", "batch_size", "parallel"):
                spec_patch["batchSize"] = str(v)
            # Controller translates contextLength to --ctx-size
            elif k_lower in ("contextlength", "context_length", "ctx_size"):
                spec_patch["contextLength"] = str(v)
            elif k_lower in ("replicas", "replica_count"):
                spec_patch["replicas"] = int(v)

            # llama.cpp CLI extra arguments
            elif k_lower in ("threads", "thread_count"):
                extra_args.append(f"--threads={v}")
            elif k_lower in ("ubatch_size", "micro_batch"):
                extra_args.append(f"--ubatch-size={v}")
            elif k_lower in ("n_gpu_layers", "ngl"):
                extra_args.append(f"--n-gpu-layers={v}")
            elif k_lower in ("flash_attn", "flash_attention"):
                if v:
                    extra_args.append("--flash-attn")
            elif isinstance(v, bool):
                flag = k.replace("_", "-")
                if v:
                    extra_args.append(f"--{flag}")
            elif k.startswith("values.") or k.startswith("params."):
                extra_args.append(f"--{k}={v}")
            else:
                extra_args.append(f"--{k.replace('_', '-')}={v}")

        if extra_args:
            spec_patch["extraCommand"] = extra_args

        return spec_patch

    def build_server_command(
        self,
        model: str,
        port: int,
        host: str = "0.0.0.0",
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build llama.cpp llama-server launch command."""
        cmd = [
            "llama-server",
            "-m",
            model,
            "--port",
            str(port),
            "--host",
            host,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd
