# AFSBox 多推論引擎調優支援架構設計 (vLLM, SGLang, llama.cpp)

## 1. 背景與核心目標

AFSBox 作為雲原生 AI 推論服務平台，原生支援多種主流開源推論引擎，包括：
- **vLLM**：主流 高吞吐量 LLM 批次推論引擎，支援 PagedAttention、Chunked Prefill、V1/V2 Engine。
- **SGLang**：極致延遲與複雜結構化輸出（RadixAttention、Multi-Node DeepEP、Chunked Prefill）推論引擎。
- **llama.cpp**：輕量化、跨平台（CPU / GPU 混合切分、GGUF 量化）低資源推論引擎。

為了讓 `auto-tuning-vllm`（後續可演進為 `auto-tune-serving`）具備全方位的推論調優能力，本文件深入分析 AFSBox Controller 既有的引擎抽象機制，並提出通用的調優架構設計。

---

## 2. AFSBox 既有架構分析：跨引擎參數抽象層

經查證 AFSBox Controller 的 CUE 模板定義（`charts/afsbox-controller/files/modelserving-definitions/engineparams.yaml` 及 `parallelism.yaml`），AFSBox 在架構上已經將常用推論參數高度抽象為 `ModelServing.spec` 頂層欄位。

### 2.1 頂層抽象欄位與引擎 CLI 旗標對照表

當 Tuner 更新 `ModelServing` 的頂層欄位時，AFSBox Controller 會根據 `spec.engine.type` 自動翻譯為各引擎對應的啟動旗標：

| 通用抽象欄位 (`spec.*`) | vLLM 旗標 | SGLang 旗標 | llama.cpp 旗標 |
| :--- | :--- | :--- | :--- |
| `batchSize` | `--max-num-seqs` | `--max-running-requests` | `--parallel` (`-np`) |
| `contextLength` | `--max-model-len` | `--context-length` | `--ctx-size` (`-c`) |
| `gpuMemoryUtilization` | `--gpu-memory-utilization` | `--mem-fraction-static` | *(N/A，CPU/RAM 切分)* |
| `prefillSettings.maxBatchTokens` | `--max-num-batched-tokens` | `--max-prefill-tokens` | *(N/A)* |
| `prefillSettings.prefillChunkTokens`| *(N/A)* | `--chunked-prefill-size` | *(N/A)* |
| `kvCacheDtype` | `--kv-cache-dtype` | `--kv-cache-dtype` | *(N/A)* |
| `parallelism.tp` | `--tensor-parallel-size` | `--tp-size` | *(N/A)* |
| `parallelism.pp` | `--pipeline-parallel-size` | `--pp-size` | *(N/A)* |
| `parallelism.dp` | `--data-parallel-size` | `--dp-size` | *(N/A)* |

> **結論**：Tuner 後端（`AFSBoxK8sBackend._map_parameters_to_serving_patch`）直接 patch 到這些通用抽象欄位，因此在通用超參數（如 Batch Size、Context Length、GPU Memory Ratio）上**已具備完全跨引擎相容性**。

### 2.2 引擎端口與網路規範

| 引擎類型 (`spec.engine.type`) | 預設服務端口 (`servicePort`) | 典型容器映像檔範例 |
| :--- | :--- | :--- |
| `vllm` | `8000` | `docker.io/vllm/vllm-openai:v0.27.1` |
| `sglang` | `30000` 或 `8000` | `docker.io/lmsysorg/sglang:latest` |
| `llamacpp` | `8080` | `ghcr.io/ggerganov/llama.cpp:server` |

---

## 3. 評測協議相容性 (OpenAI-Compatible API)

AIPerf Benchmark 與 AFSBox 評測工作負載（`Benchmark` CR）統一透過標準 OpenAI HTTP 協議進行負載發送與效能測量：
- **API 端點**：`http://<service>.<namespace>.svc.cluster.local:<port>/v1/chat/completions`
- **相容性現狀**：
  - **vLLM**：原生支援 OpenAI API（SSE 串流、`usage` 統計）。
  - **SGLang**：原生支援 OpenAI 相容模式（透過 `sglang.launch_server` 提供）。
  - **llama.cpp**：llama-server 原生提供完全相同的 `/v1/chat/completions` 及 `/v1/models` 端點。
- **結論**：評測端（AIPerf）**無需修改任何程式碼**，即可直接對三種引擎量測相同的 TTFT（首字延遲）、ITL（字間延遲）、TPS（輸出吞吐量）等指標。

---

## 4. 多引擎調優支援方案

### 4.1 階段一：配置驅動模式（零代碼修改，立即支援）

使用者只需在 Study 配置檔的 `afsbox.serving_template` 指定相應引擎即可：

#### 範例 A：SGLang 調優配置
```yaml
study:
  prefix: "sglang_tuning"
backend: "afsbox"
afsbox:
  namespace: "afsbox-tokenfactory"
  serving_name: "sglang-optuna-exp"
  cleanup_serving: true
  serving_template:
    image: "docker.io/lmsysorg/sglang:latest"
    engine:
      type: sglang
      servicePort: 30000
    modelType: llm
    servedModelName: qwen-7b
    command:
      - "python3"
      - "-m"
      - "sglang.launch_server"
      - "--model-path=/models/afsbox/qwen/qwen-7b/latest"
      - "--served-model-name=qwen-7b"
      - "--port=30000"
    model:
      valueFrom:
        kind: ClusterModelRepository
        name: qwen-7b
    batchSize: "32"             # AFSBox 自動轉為 --max-running-requests=32
    gpuMemoryUtilization: "0.8" # AFSBox 自動轉為 --mem-fraction-static=0.8
benchmark:
  benchmark_type: "aiperf"
  model: "qwen-7b"
parameters:
  batch_size:
    enabled: true
    options: [16, 32, 64]
  gpu_memory_utilization:
    enabled: true
    min: 0.7
    max: 0.9
    step: 0.05
```

#### 範例 B：llama.cpp 調優配置
```yaml
study:
  prefix: "llamacpp_tuning"
backend: "afsbox"
afsbox:
  namespace: "afsbox-tokenfactory"
  serving_name: "llamacpp-optuna-exp"
  cleanup_serving: true
  serving_template:
    image: "ghcr.io/ggerganov/llama.cpp:server"
    engine:
      type: llamacpp
      servicePort: 8080
    modelType: llm
    servedModelName: llama3-8b-gguf
    command:
      - "/llama-server"
      - "-m"
      - "/models/afsbox/llama3/8b-q4_k_m.gguf"
      - "--alias=llama3-8b-gguf"
      - "--port=8080"
    model:
      valueFrom:
        kind: ClusterModelRepository
        name: llama3-8b-gguf
    batchSize: "8"              # AFSBox 自動轉為 --parallel 8
    contextLength: "4096"       # AFSBox 自動轉為 --ctx-size 4096
benchmark:
  benchmark_type: "aiperf"
  model: "llama3-8b-gguf"
parameters:
  batch_size:
    enabled: true
    options: [4, 8, 16]
```

---

### 4.2 階段二：引擎專屬方言適配器（Engine Dialect Adapter）

針對各引擎獨佔的進階性能參數，在 `auto-tune-vllm` 中引入 `EngineAdapter` 抽象介面：

#### 引擎獨佔參數表：
1. **SGLang 獨佔**：
   - `chunked_prefill_size`: 分塊預填充大小 (e.g. 512, 1024, 2048)
   - `schedule_policy`: 請求調度策略 (`lpm` 最長前綴匹配 / `fcfs` 先來先服務)
   - `attention_backend`: 注意力後端 (`flashinfer`, `triton`)
2. **llama.cpp 獨佔**：
   - `threads` (`-t`): CPU 推論執行緒數量
   - `n_gpu_layers` (`-ngl`): 卸載至 GPU 的模型層數
   - `ubatch_size` (`-ub`): 物理微批次 (micro-batch) 大小
3. **vLLM 獨佔**：
   - `block_size`: KV Cache 區塊大小 (`16`, `32`)
   - `swap_space`: CPU 記憶體交換空間 (GB)
   - `enforce_eager`: 停用 CUDA Graph (`true` / `false`)

#### 架構實作方式：
```python
class EngineAdapter(ABC):
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """驗證參數是否符合該引擎規範"""
        pass

    @abstractmethod
    def map_to_serving_patch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """將搜尋空間參數映射至 ModelServing CR 的 spec 欄位或 spec.extraCommand"""
        pass
```
未收錄在 AFSBox CRD 頂層的專屬參數，適配器會自動轉成 CLI 旗標塞入 `spec.extraCommand` 注入 Pod，無須更動 K8s CRD 即可支援任意引擎參數！

---

### 4.3 階段三：專案命名與通用演進路徑

1. **模組重命名與結構通用化**：
   - 將專案逐步演進為 `auto-tune-serving` 或保留向後相容別名。
   - 將 `auto_tune_vllm/schemas/vllm_defaults/` 擴展為：
     - `schemas/vllm_defaults/`
     - `schemas/sglang_defaults/`
     - `schemas/llamacpp_defaults/`
2. **自動推導最佳預設**：
   - 若使用者設定 `engine.type: sglang`，自動推薦適合 SGLang 的超參數搜索空間（如 `schedule_policy`, `chunked_prefill_size`）。
   - 若設定 `engine.type: llamacpp`，自動推薦 CPU/GPU 混合與線程數搜索空間。
