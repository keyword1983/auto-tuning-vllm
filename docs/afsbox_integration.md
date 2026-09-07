# AFSBox × auto-tuning-vllm (Optuna) 超參數自動調校整合設計與規劃

## 一、架構背景與設計理念

### 1.1 背景與問題
AFSBox 現有的推論調校功能（`ModelTuning`）採用靜態網格搜尋（Grid Search / Zip / OneAtATime）：
- **維度爆炸**：若調校 4 個維度各 4 個數值，需執行 $4^4 = 256$ 次測試。每次測試含 Pod 啟動、權重加載、預熱與 AIPerf 壓測約需 5~10 分鐘，整體耗時過長。
- **缺乏反饋（Blind Search）**：無法依據前期測試結果進行自適應學習；若某參數區間引發 OOM，靜態搜尋仍會重複嘗試，造成數小時的算力浪費。
- **難以權衡多目標**：Throughput（每秒吞吐）與 TTFT（首字延遲）通常互斥，使用者缺乏直觀的 Pareto 前沿決策依據。

### 1.2 整合目標
引進 [`auto-tuning-vllm`](https://github.com/openshift-psap/auto-tuning-vllm) 的 Optuna 核心，實現：
1. **自適應學習**：使用 TPE（貝氏最佳化）與 NSGA-II（多目標基因演算法），以 15~30 次 Trial 快速收斂至最佳參數區域。
2. **多目標 Pareto 前沿**：同時最佳化 Throughput（最大化）與 TTFT P95（最小化），產出帕雷托前沿解集。
3. **無效參數剪枝（Pruning）**：提前攔截不合規或已失敗的參數組合，零浪費 GPU 算力。
4. **保留 AFSBox 雲原生與多節點優勢**：完整複用 AFSBox 的 LeaderWorkerSet (LWS)、RDMA 互聯、S3 直讀串流與 NVIDIA AIPerf 壓測體系。

---

## 二、專案定位、職責邊界與架構設計（Architecture & Project Scope）

### 2.1 專案邊界釐清：本專案（Autonomous Runner）vs 另專案（optuna-advisor 微服務）

在 AFSBox 智慧調優體系中，存在兩種不同的技術實現路線。**本專案 (`auto-tuning-vllm`) 採用的是「路線 A：端到端自主調校 Runner」**：

```mermaid
graph TB
    subgraph RouteA["【路線 A：本專案 auto-tune-vllm】端到端自主 Runner（Autonomous Driver）"]
        direction TB
        JobA["Tuner Runner Job<br/>(auto-tune-vllm:runner)"]
        OptunaA["內建 Optuna 最佳化引擎<br/>(TPE / NSGA-II)"]
        BackendA["內建 AFSBoxK8sBackend<br/>(直接驅動 K8s 調校迴圈)"]
        K8sA["K8s APIServer<br/>(ModelServing / Benchmark / Status)"]
        
        JobA --- OptunaA
        JobA --- BackendA
        BackendA -->|"主動操作 CRD / 收集指標 / 回寫狀態"| K8sA
        NoteA["特性：大腦與驅動者合一，獨立隨需啟動，完工自毀，無需外部常駐服務"]
    end

    subgraph RouteB["【路線 B：另一個專案 optuna-advisor】微服務 Advisor 模式"]
        direction TB
        AdvisorB["optuna-advisor 微服務<br/>(FastAPI 常駐 Service:8000)"]
        CtrlB["afsbox-controller<br/>(feat/optuna-advisor-client)"]
        K8sB["K8s APIServer<br/>(ModelServing / Benchmark)"]

        CtrlB -->|"HTTP POST /suggest (索取參數)"| AdvisorB
        CtrlB -->|"HTTP POST /feedback (回傳分數)"| AdvisorB
        CtrlB -->|"由 Controller 自己驅動"| K8sB
        NoteB["特性：Advisor 僅為純數學計算服務，所有 K8s 調校迴圈由 Controller 主導"]
    end
```

| 比較維度 | 路線 A：本專案 (`auto-tuning-vllm`) | 路線 B：另一個專案 (`optuna-advisor`) |
| :--- | :--- | :--- |
| **角色定位** | **端到端自主調校執行器（Autonomous Tuner Runner）** | **純 HPO 顧問微服務（HTTP REST Advisor Service）** |
| **所在儲存庫** | `ai-workspace/auto-tuning-vllm` | `afsbox/optuna-advisor` |
| **調校迴圈主導者** | **Runner 自己主導**（讀 CR、改 Serving、建 Benchmark、寫 Status） | **`afsbox-controller` 主導**（Controller 逐輪呼叫 Advisor API） |
| **K8s API 依賴** | Runner 具備 K8s Client，直接與 APIServer 互動 | Advisor 完全不接觸 K8s，純吃 JSON / 吐 JSON |
| **部署型態** | 隨需啟動的 **`batchv1.Job` / 容器**（跑完自動釋放資源） | 常駐叢集的 **HTTP 微服務**（ClusterIP:8000） |

---

### 2.2 本專案系統層級與組件架構圖

在目前分支中，`auto-tune-vllm` 是一個全自動的調優驅動者：

```mermaid
graph TB
    subgraph TunerRunner["auto-tune-vllm : Runner 容器內部組件"]
        CLI["CLI 進入點 (main.py)<br/>--backend afsbox --tuning-name <cr>"]
        StudyCtrl["StudyController<br/>(Optuna 搜尋迴圈排程器)"]
        Synthesizer["CR 配置合成器 (config.py)<br/>將 ModelTuning 轉為 Optuna 設定檔"]
        Backend["AFSBoxK8sBackend (afsbox.py)<br/>實作 ExecutionBackend 介面"]
        K8sClient["K8s Python Client<br/>CustomObjectsApi"]
    end

    subgraph K8sAPI["Kubernetes APIServer (宣告式 CRD)"]
        CR_Tuning["ModelTuning CR<br/>(調校規格、目標、候選狀態與 Pareto)"]
        CR_Serving["ModelServing CR<br/>&lt;tuning&gt;-exp (實驗推論服務)"]
        CR_Bench["Benchmark CR<br/>&lt;tuning&gt;-cN (單輪壓測任務)"]
        CR_Report["BenchmarkReport CR<br/>&lt;tuning&gt;-cN (效能評測指標)"]
    end

    subgraph AFSBoxInfra["AFSBox 底層基礎設施 (Operator & Workloads)"]
        AFS_Ctrl["afsbox-controller<br/>(負責 ModelServing 與 Benchmark 的實體調度)"]
        ServingPod["vLLM 推論容器 Pod<br/>(掛載實體 GPU 與模型權重)"]
        AIPerfPod["AIPerf 壓測容器 Pod<br/>(產生高並發合成負載)"]
    end

    CLI --> Synthesizer
    Synthesizer -->|"1. 讀取調校規格"| CR_Tuning
    CLI --> StudyCtrl
    StudyCtrl --> Backend
    Backend --> K8sClient

    K8sClient -->|"2. 建立/Patch 參數 (generation+1)"| CR_Serving
    K8sClient -->|"3. 建立壓測任務"| CR_Bench
    K8sClient -->|"4. 輪詢壓測數據"| CR_Report
    K8sClient -->|"5. 即時回寫 candidates (variables/metrics)"| CR_Tuning

    AFS_Ctrl -->|"Watch & 部署"| CR_Serving
    AFS_Ctrl -->|"調度 GPU 起 Pod"| ServingPod
    AFS_Ctrl -->|"Watch & 執行壓測"| CR_Bench
    AFS_Ctrl -->|"啟動壓測 Job"| AIPerfPod
    AIPerfPod -->|"HTTP /v1/chat/completions"| ServingPod
    AIPerfPod -->|"輸出指標成果"| CR_Report
```

---

## 三、與 AFSBox 互動溝通與完整生命週期時序圖（Communication & Lifecycle Sequence）

下圖反映目前分支中 Runner 與 K8s APIServer、Controller 及 Workloads 的**真實端到端互動過程**：

```mermaid
sequenceDiagram
    autonumber
    participant Portal as 使用者 / Portal
    participant Runner as auto-tune-vllm Runner
    participant K8s as Kubernetes APIServer
    participant Ctrl as afsbox-controller
    participant Serving as vLLM Serving Pod
    participant Bench as AIPerf 壓測 Pod

    Note over Portal, K8s: 階段 1：任務建立與配置合成
    Portal->>K8s: 建立 ModelTuning CR (指定模型、搜尋空間、最佳化目標)
    Runner->>K8s: GET ModelTuning CR (讀取 --tuning-name)
    K8s-->>Runner: 回傳 CR Spec
    Runner->>Runner: 自動合成 Optuna 配置（動態調整 n_startup_trials、Baseline 參數）

    Note over Runner, Serving: 階段 2：實驗 Serving 建立與 Ready 判定
    Runner->>K8s: POST ModelServing (<name>-exp)
    Ctrl->>K8s: 偵測到新 ModelServing 建立
    Ctrl->>Serving: 渲染 Helm Release、掛載 GPU、啟動 vLLM 容器
    loop 每 5 秒輪詢 Serving 狀態
        Runner->>K8s: GET ModelServing (<name>-exp)
        K8s-->>Runner: status.phase=Progressing
    end
    Serving-->>Ctrl: 權重載入完畢，Health Check 200 OK
    Ctrl->>K8s: 更新 status.phase=Ready, observedGeneration=1
    Runner->>K8s: GET ModelServing -> 狀態為 Ready！進入調優迴圈

    Note over Runner, Bench: 階段 3：逐輪參數調教與壓測評估 (Trial 0..N-1)
    loop 每一輪 Trial
        Runner->>Runner: 演算法產生候選參數 (batchSize, memUtil)
        Runner->>K8s: PATCH ModelServing (<name>-exp) [generation+1]
        Ctrl->>Serving: 重啟 / 滾動更新推論容器
        loop 防競態等候 (Generation-Aware Sync)
            Runner->>K8s: GET ModelServing
            Note over Runner: 等候 observedGeneration >= targetGeneration 且 phase == Ready
        end
        Runner->>K8s: POST Benchmark CR (<name>-c{N})
        Runner->>K8s: PATCH ModelTuning.status (標記該候選為 Testing 並寫入 variables)
        Ctrl->>Bench: 啟動 AIPerf Job 容器
        Bench->>Serving: 內網直連發送並發流量 (HTTP /v1/chat/completions)
        Bench-->>Ctrl: 壓測完成，輸出 Throughput 與 TTFT Summary
        Ctrl->>K8s: 建立同名 BenchmarkReport CR (<name>-c{N})
        Runner->>K8s: GET BenchmarkReport (<name>-c{N})
        K8s-->>Runner: 回傳完整評測數據 (Throughput, TTFT)
        Runner->>Runner: 吸收效能指標，更新 Optuna 最佳化狀態
        Runner->>K8s: PATCH ModelTuning.status (標記 Completed 並寫入 metrics 與 variables)
    end

    Note over Runner, Ctrl: 階段 4：最優解彙整與資源清理
    Runner->>Runner: 運算 Pareto Frontier 與 Best Candidate
    Runner->>K8s: PATCH ModelTuning.status (寫入 bestCandidate 與 paretoFrontier)
    Runner->>K8s: DELETE ModelServing (<name>-exp)
    Ctrl->>Serving: 刪除實驗 Serving Pod，徹底釋放實體 GPU 算力！
    Portal->>K8s: 讀取 ModelTuning.status 呈現帕雷托前沿散佈圖
```

---

## 四、Optuna 隨需啟動、時序與防停等生命週期（Execution Lifecycle & Safety Guardrails）

### 4.1 隨需啟動機制（On-Demand Job vs. Daemon）
Optuna 並非作為常駐的背景 Daemon 運行，而是採用**「雲原生隨需即開（Run-to-Completion）」**模式：
1. 當使用者在 Portal 送出調校請求，`afsbox-platform` 建立 `ModelTuning` CR。
2. `afsbox-controller` 在 `reconcilePending` 識別 `isOptuna(tuning) == true`。
3. Controller 建立底層載體 `<tuning>-exp` ModelServing，並調用 `ensureTunerJob` 建立 K8s 原生 `batchv1.Job`（名稱為 `<tuning>-tuner`）。
4. 容器映像檔為純 CPU 的輕量鏡像（`auto-tune-vllm:runner`，約 400MB），不佔用任何實體 GPU。

### 4.2 啟動後執行時序（9 步全生命週期）
1. **組態合成 (Synthesis)**：Tuner Pod 啟動後，連線 In-Cluster K8s API 讀取父級 `ModelTuning` CR 的 `spec.optimization` 與 `spec.testSuite`，在記憶體中自動合成 Optuna 組態。
2. **智慧採樣 (Ask)**：Sampler（NSGA-II 或 TPE）計算後驗機率分佈，採樣出一組超參數（如 `tp=2, gpu_mem=0.9`）。
3. **底層注入 (Patch)**：`AFSBoxK8sBackend.submit_trial()` 對 `<tuning>-exp` 進行 Spec Patch。
4. **等待就緒 (Wait Ready)**：等待 ModelServing 的 LeaderWorkerSet 完成滾動更新，確認 `Phase == Ready` 且 `observedGeneration >= generation`。
5. **發起壓測 (Trigger Benchmark)**：建立子 `Benchmark` CR（名稱為 `<tuning>-trial-N`），觸發 AIPerf 引擎。
6. **收集指標與回饋 (Tell & Sync)**：讀取壓測產出的 TTFT、Throughput、ITL，呼叫 `Optuna.tell()` 更新後驗模型，同時寫回 `status.candidates` 供前端即時顯示進度。
7. **迴圈迭代 (Loop)**：重複步驟 2~6 直到跑滿 `n_trials`（預設 20 輪）。
8. **前沿寫回 (Writeback)**：Tuner 結束前呼叫 `sync_final_results_to_tuning`，將 `bestCandidate` 與 `paretoFrontier` 寫入 `ModelTuning.status`。
9. **收斂退出 (Exit 0)**：Tuner Job 正常退出，Controller 接手進行 GPU 資源釋放。

### 4.3 四層防停等與死鎖護欄體系（Anti-Hanging Guardrails）
| 層級 | 防護對象 | 潛在停等風險 | 系統防護機制 |
| :--- | :--- | :--- | :--- |
| **L1** | **ModelServing 啟動** | 某組參數顯存超額引發 OOM，或 Pod CrashLoopBackOff | `deploy_timeout_seconds`（預設 600s）。若 Serving 轉為 `Failed` 或超時未 Ready，立即中斷該 Trial 判定為失敗，Optuna 施加極大懲罰後直接進下一輪，**絕不卡住**。 |
| **L2** | **AIPerf 壓測執行** | 網路中斷或請求掛起導致壓測 Pod 卡死 | AFSBox `Benchmark.spec.suite` 內建 `timeoutSeconds`（預設 300s），且 Benchmark Controller 配置有原生 `ActiveDeadlineSeconds`，超時自動標記 `Failed`，Tuner 後端立刻結案該輪。 |
| **L3** | **Tuner Job 自身** | 演算法內部異常或連線死鎖導致 Job 不退出 | `batchv1.JobSpec` 配置了 `ActiveDeadlineSeconds: 7200`（2 小時硬上限）。超過 2 小時 K8s 排程器**強制終止 Job**。 |
| **L4** | **GPU 並發衝突** | 多個 Trial 同時搶同一張卡導致互撞 | 啟動命令固定傳入 `--max-concurrent-trials 1`，強制單一序向執行，徹底杜絕並發資源競爭。 |

### 4.4 完成通知與狀態雙保險機制
1. **APIServer 即時推播（Watch Event）**：
   * Controller 註冊了 `Owns(&batchv1.Job{})`。
   * Tuner Job 一旦完成（`JobComplete` 條件為 True），APIServer 會微秒級推播事件喚醒 Controller 進入 `reconcileRunning`。
2. **5 秒備援輪詢（Poll Fallback）**：
   * Controller 設有 `RequeueAfter: 5s`，即使遇到極端網路抖動漏掉事件，每 5 秒也會主動檢查一次 Job 狀態。
3. **前端 Portal 即時渲染**：
   * Portal 透過 TanStack Query 每 3 秒輪詢後端，一旦 Controller 將 Phase 標記為 `Completed`，UI 即刻切換為完成狀態並繪製帕雷托圖表。

### 4.5 資源分級清理與垃圾回收（Resource Deletion & GC）
1. **實驗 ModelServing（佔用昂貴 GPU）**：
   * **立刻刪除**。Controller 在 `reconcileFinalizing` 階段第一時間執行 `r.Delete(ctx, exp)`，立即歸還 GPU、LWS 容器與顯存。
2. **Tuner Runner Job（純 CPU Pod）**：
   * 配置了 `TTLSecondsAfterFinished: 86400`（24 小時後自動回收）。
   * 完工後 Pod 處於 `Completed` 狀態不佔 CPU/GPU，保留 24 小時便於工程師使用 `kubectl logs job/<tuning>-tuner` 查閱搜尋日誌。
3. **子 Benchmark CRs（壓測報告與歷史數據）**：
   * **調校完當下不刪除**，因為調校報告需要調閱每輪 Trial 的詳細數據。
   * 每個子 Benchmark 均打上 `OwnerReference` 指向父級 `ModelTuning`。**當使用者在 Portal 點擊刪除該任務時，K8s 垃圾回收器會自動級聯清空所有相關的子 Benchmark，零孤兒物件殘留**。

### 4.6 測試流量鑑權與 API Token 機制
* **為什麼需要 API Token？**：
  * **不是給 Optuna 用的**（Optuna 跑在 K8s 內部透過 ServiceAccount 通訊）。
  * **是給 AIPerf 壓測引擎用的**。在 AFSBox 規範（PR `!210`）中，所有流經內部 `AgentGateway` 的模型請求均採取 **Deny-by-Default** 零信任政策，未帶 Token 會回傳 `401 Unauthorized`。
* **Token 來源**：使用者只要在 AFSBox Portal 左側選單的 **`Workspace ──> API Keys`** 複製既有的模型存取金鑰即可。
* **安全保護**：Platform 收到後會存入 K8s Secret（`<tuning>-apikey`），CR 僅記載 `SecretKeyRef`，絕不外洩明文。該 Secret 同樣綁定 OwnerReference，隨任務刪除自動級聯銷毀。

---

## 五、Benchmark 壓測參數配置與映射機制（Benchmark Alignment）

在 `backend: afsbox` 架構下，壓力測試由 AFSBox 叢集內的 `Benchmark` CR（NVIDIA AIPerf 引擎）負責。系統支援雙模式獲取 Benchmark 壓測組態：

### 5.1 雙模式運作
1. **模式 A：AFSBox CR / Portal UI 驅動（推薦在生產與平台環境）**：
   * 使用者在 Portal 介面選擇模型與壓測模板，組態自動存入 `ModelTuning.spec.testSuite`。
   * `auto-tuning-vllm` 的 Tuner Runner 啟動後，`AFSBoxK8sBackend` 會**優先讀取並鎖定該 `testSuite`**，每輪 Optuna Trial 直接套用此測試規格。
2. **模式 B：YAML / CLI 驅動（供研發實驗與離線自動化）**：
   * 開發者撰寫 Study YAML 檔中的 `benchmark:` 區塊，`AFSBoxK8sBackend._build_benchmark_suite()` 會自動將其轉換為標準 AFSBox `Benchmark.spec.suite`。

### 5.2 參數精準映射表
| `auto-tuning-vllm` 欄位 | AFSBox `Benchmark.spec.suite[0].params` (AIPerf) | 說明與作用 |
| :--- | :--- | :--- |
| `samples` | `requestCount` | 總發送請求數 (例如 200) |
| `rate` / `concurrency` | `concurrency` | 同時在途併發路數 (例如 16) |
| `request_rate` | `requestRate` (`type: "request-rate"`) | 開迴路到達速率（設為 `"inf"` 則為閉迴路固定併發） |
| `prompt_tokens` | `isl: { mean: 1024, stddev: 0 }` | 輸入序列長度常態分佈 |
| `output_tokens` | `osl: { mean: 512, stddev: 0 }` | 輸出序列長度常態分佈 |
| `dataset: "sharegpt"` | `dataset: "sharegpt"` (`type: "dataset-replay"`) | 資料集重放 (支援 `sharegpt`, `sonnet`) |
| `max_seconds` | `timeoutSeconds` | 單次壓測最大執行超時時間 |
| *(自動注入)* | `streaming: true` | **固定開啟**（SSE 串流是量測 TTFT/ITL 的必要前提） |
| *(自動注入)* | `ignoreEOS: true` | **固定開啟**（強制測滿指定 OSL，避免模型提早結束導致吞吐被虛假高估） |

### 5.3 Study YAML 範例 (`examples/study_config_afsbox.yaml`)
```yaml
study:
  name: "llama3_afsbox_tuning"

backend: "afsbox"
afsbox:
  namespace: "default"
  tuning_name: "llama3-tuning-01"
  deploy_timeout_seconds: 600

optimization:
  approach: "multi_objective"
  objectives:
    - metric: "output_tokens_per_second"
      direction: "maximize"
    - metric: "time_to_first_token_ms"
      direction: "minimize"
  sampler: "nsga2"
  n_trials: 20
  max_concurrent_trials: 1

benchmark:
  benchmark_type: "aiperf"
  model: "meta-llama/Meta-Llama-3-8B-Instruct"
  samples: 200
  rate: 16
  request_rate: "inf"
  prompt_tokens: 1024
  output_tokens: 512
  dataset: "sharegpt"
  max_seconds: 300

parameters:
  tensor_parallel_size:
    enabled: true
    options: [1, 2]
  max_num_batched_tokens:
    enabled: true
    options: [2048, 4096, 8192]
  gpu_memory_utilization:
    enabled: true
    min: 0.80
    max: 0.95
    step: 0.05
```

---

## 六、超參數搜尋空間全景（Hyperparameter Search Space）

系統絕不僅限於基礎參數，而是支援 **vLLM / SGLang 的全方位效能旋鈕** 以及 **任意自訂 CLI 旗標**。

### 6.1 核心參數分類一覽表
| 領域分類 | 參數名稱 (Variable) | 調校模式 | 常見取值範例 | 效能影響與調優目標 |
| :--- | :--- | :---: | :--- | :--- |
| **吞吐量與批次排程** | `max_num_batched_tokens` | `Values` | `[2048, 4096, 8192, 16384]` | 單步最大 Token 數，極度影響 Throughput。 |
| | `max_num_seqs` (`batchSize`) | `Range/Values` | `min: 64, max: 512, step: 64` | 同時併發處理序列上限。 |
| | `enable_chunked_prefill` | `Values` | `[true, false]` | 分塊預填充，有效平抑高併發時的 TTFT 延遲尖峰。 |
| | `max_model_len` (`contextLength`)| `Values` | `[4096, 8192, 32768]` | 上下文長度限制，縮小可釋放顯存給更大 Batch。 |
| **顯存與 KV 快取** | `gpu_memory_utilization` | `Range` | `min: 0.80, max: 0.96, step: 0.02` | GPU 顯存佔用上限，過高恐 OOM，過低吞吐差。 |
| | `kv_cache_dtype` | `Values` | `["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]` | KV 快取量化，改為 fp8 可顯存減半、併發容量翻倍。 |
| | `block_size` | `Values` | `[16, 32]` | PagedAttention 快取分塊大小。 |
| | `swap_space` | `Range` | `min: 4, max: 16, step: 4` | CPU 換頁記憶體（GiB），防止偶發負載爆量崩潰。 |
| **分散式並行策略** | `parallelism.tp` (`tensor_parallel_size`) | `Values` | `[1, 2, 4, 8]` | 張量平行度，跨卡拆分，跨卡通信影響延遲。 |
| | `parallelism.pp` (`pipeline_parallel_size`) | `Values` | `[1, 2, 4]` | 流水線平行度，超大模型跨節點切層。 |
| | `parallelism.ep` | `Values` | `[1, 2, 4]` | 專家平行度（針對 DeepSeek / Qwen 等 MoE 模型）。 |
| **核心加速與編譯** | `enable_cuda_graphs` | `Values` | `[true, false]` | 是否啟用 CUDA Graphs，大幅降低 CPU 調度開銷。 |
| | `attention_backend` | `Values` | `["FLASH_ATTN", "XFORMERS", "TORCH_SDPA"]` | 底層注意力運算核心選擇。 |
| | `speculative_model` / `num_speculative_tokens` | `Values` | `[1, 3, 5]` | 投機解碼（Speculative Decoding）草稿步數。 |
| **量化加載** | `quantization` | `Values` | `["awq", "gptq", "fp8", "bitsandbytes"]` | 權重量化方式。 |
| **任意自訂旗標** | `extraArgs` / 自訂變數 | `Values` | `--flag=val` | 任何 vLLM CLI 新旗標均可直接透傳注入。 |

---

## 七、Portal 前端互動設計與參數填寫規格（Portal UI & UX Design）

在 `afsbox-portal` 中，Optuna 自適應調校直接嵌入在推論部署精靈的「部署前調校（Tuning Mode）」中：

### 7.1 面板四層結構
1. **搜尋引擎與最佳化模式 (Optimization Engine)**：
   - 提供「Optuna 自適應最佳化 (推薦)」與「傳統網格搜尋 (Static Grid)」切換。
   - **採樣演算法 (Sampler)**：下拉選單（`NSGA-II (多目標)`、`TPE (貝氏最佳化)`、`Random`）。
   - **試驗輪數 (n_trials)**：自訂測試輪數（預設 20 次，可設 10~100）。
   - **多目標權衡 (Objectives)**：Switch 開關勾選「吞吐量最大化 (Tokens/s)」與「首字延遲最小化 (TTFT P95)」。
2. **超參數搜尋維度 (Search Space Dimensions)**：
   - 點擊「+ 新增維度」，可挑選引擎內建題目、自訂變數或指令變體。
   - 支援連續範圍（Range：min, max, step）或離散清單（Values：選項清單）。
3. **壓力測試規格 (TestSuite Conditions)**：
   - 設定 AIPerf 壓測型態、併發路數、總請求數、ISL/OSL 與超時時間。
4. **驗證與送出 (Confirm & Launch)**：
   - 填寫調校顯示名稱與內部 Gateway 鑑權用 API Key，點擊「開始 Optuna 智慧調校」。

### 7.2 欄位屬性規格一覽
| 欄位區塊 | 欄位名稱 | 必填/選填 | 預設值 | 說明 |
| :--- | :--- | :---: | :---: | :--- |
| **演算法組態** | 搜尋引擎 (`engine`) | **必選** | `optuna` | `optuna` 或 `static`。 |
| | 採樣演算法 (`sampler`) | 選填 | `nsga2` | 推薦多目標使用 `nsga2`，單目標推薦 `tpe`。 |
| | 試驗輪數 (`n_trials`) | 選填 | `20` | 建議 15~30 輪即可快速收斂。 |
| | 最佳化目標 (`objectives`) | **必選至少一項** | 雙開 (Throughput + TTFT) | 決定 Optuna 損失函數與收斂方向。 |
| **搜尋空間** | 調校維度 (`dimensions`) | **必填至少一維** | 系統建議 4 參數 | 可自由增刪為任意 vLLM 參數。 |
| **壓測標準** | 測試項目 (`testSuite`) | **必填至少一項** | `concurrency` (c=8, n=100) | 決定每輪 Trial 的客觀測試標準。 |
| **執行安全** | 存取金鑰 (`apiKey`) | **必填** | 無 (使用者手動輸入) | 供子 Benchmark 經內部 agentgateway 通訊鑑權。 |
| | 任務名稱 (`displayName`) | 選填 | 自動產生 | UGC 顯示名（支援中文）。 |

---

## 八、Optuna 與 Static 模式架構對比、相容性與硬體邊界（Architecture Comparison & Constraints）

### 8.1 為什麼 Static 寫在 Controller 內，而 Optuna 採用獨立 Tuner Job？
* **Static 模式**：
  * 本質是**無狀態的固定迴圈（Stateless For-Loop）**。
  * Platform 先透過 `sweep_combine.go` 的笛卡爾積算好候選清單（`c01, c02...`），Controller 只需要按陣列順序跑，不涉及動態決策，寫在 Go 裡面最簡潔。
* **Optuna 模式**：
  * 本質是**有狀態的機器學習統計運算（Stateful ML Loop）**。
  * 依賴複雜數學（NumPy, SciPy, TPE, NSGA-II），Go 語言缺乏成熟庫。
  * **抗故障高可用**：將其做成獨立的 Tuner Job Pod，即使 Controller 在 2 小時調校過程中重啟或升級，**跑在背景的 Tuner 運算完全不受干擾**。Controller 恢復後只需查看 Job 狀態即可接軌。
  * **跑完即焚 (Run-to-Completion)**：不需要常駐維護一個 Python RPC Server，調校結束 Pod 自動銷毀，不浪費叢集常態資源。

### 8.2 原本 Static 未支援 TP/顯存的歷史原因
* **調查結論**：原本未支援**不是因為技術死穴，而是剛好卡在架構重構的交接期**。
* **Git 紀錄證實**：在 2026-09-01（Commit `880112ffe`）的重構中，團隊將 `ModelServingSpec` 改為具名欄位，當時工程師先填補了最常用的 3 個欄位，並留有註解：`// tuning 變數要改指 spec 的欄位路徑...那是獨立工項，尚未做。`
* **解鎖現況**：本次重構已全面補齊 `helpers.go`，使底層 `setVariable` 同時解鎖了 TP、顯存、KV 快取與自訂參數。

### 8.3 調校進階參數的 4 大客觀物理限制
使用者在設定調校維度時，平台雖已解鎖，但仍需注意以下客觀限制：
1. **實體 GPU 卡數容量限制**：
   * 若叢集單節點只有 2 張卡，但調校維度包含 `TP=4`，Pod 會因為節點資源不足卡在 `Pending` 狀態（超時後系統會記錄 `DeployTimedOut` 並跳過）。
2. **GPU 架構代數限制（硬體不支援）**：
   * 原生 FP8 KV 快取（`kv_cache_dtype: fp8`）僅支援 NVIDIA Ada Lovelace（L4/4090）與 Hopper（H100/H200）以上顯卡。跑在舊卡（如 A100）上 vLLM 會直接拋錯中斷該 Trial。
3. **模型張量幾何整除限制**：
   * `TP` 必須能夠整除該模型的注意力頭數（Attention Heads）。例如 32 頭的模型若設定 `TP=3` 或 `TP=6`，vLLM 會在啟動階段報錯。
4. **Static 盲踩 OOM 浪費 vs. Optuna 智慧避坑**：
   * 在 Static 模式中，若某個參數（如顯存 0.98）會引發 OOM，靜態網格依然會盲目地為每一組組合重複加載模型測試 10 分鐘，浪費數百分鐘。
   * Optuna 模式在第 1 輪發現 OOM 失敗後，貝氏演算法會給予嚴重懲罰，**後續輪次會自動主動繞開該崩潰區域**。

### 8.4 Webhook 防雙源漂移與 GPUClaim 自動聯鎖機制
* AFSBox 的 ModelServing Webhook 具備 `validateParallelismGPUCountConsistency` 規則，強制要求 `gpuClaim.requests.count` 必須與 `TP * PP * DP / nodes` 推導值一致。
* 我們在 `helpers.go` 實作了 `syncGPUClaimCount(spec)`：**當調校修改 TP 或 PP 時，系統會自動重新推導並同步更新 `GPUClaim.Requests.Count`**，徹底杜絕了被 Webhook 當場拒絕的衝突問題。

---

## 九、部署架構、所屬 Helm Chart 與發布流程（Deployment & Helm Architecture）

### 9.1 所屬 Helm Chart 分工矩陣
| 部件名稱 | 原始碼目錄 | 所屬 Helm Chart | 部署型態 | 職責與角色 |
| :--- | :--- | :--- | :--- | :--- |
| **控制面 Operator** | `afsbox-controller` | **`charts/afsbox-controller`** | `Deployment` (常駐) | 提供 `ModelTuning` CRD、派發 Tuner Job、調校結束後刪除實驗 Serving 釋放 GPU。 |
| **後端 API (BFF)** | `afsbox-platform` | **`charts/afsbox-platform`** | `Deployment` (常駐) | 接收 Portal 請求、寫入 CR、提供調校報告 API。 |
| **前端 UI (Portal)** | `afsbox-portal` | **`charts/afsbox-portal`** | `Deployment` (Nginx) | 部署精靈中的 Optuna 表單與 Pareto 散佈圖報表。 |
| **演算法大腦 (Optuna)** | `auto-tuning-vllm` | **隨需映像檔**<br>*(不建獨立 Chart)* | **`batchv1.Job`**<br>*(跑完退出)* | 純 CPU 輕量容器（~400MB），執行 Optuna 採樣迴圈，完工後自動結束。 |

### 9.2 Tuner 映像檔三級覆寫機制
```
Level 1 (CR 級自訂): ModelTuning.spec.optimization.tunerImage
         │ (若未設定，往下一級 fallback)
         ▼
Level 2 (Helm 級環境變數): afsbox-controller 的 OPTUNA_TUNER_IMAGE
         │ (若未設定，往下一級 fallback)
         ▼
Level 3 (系統預設): keyword1983/auto-tune-vllm:runner
```
在離線或企業私有 Harbor 環境中，只需在 `charts/afsbox-controller` 的 `values.yaml` 中配置：
```yaml
controllerManager:
  extraEnv:
    - name: OPTUNA_TUNER_IMAGE
      value: "harbor.internal.corp/afsbox/auto-tune-vllm:runner"
```

### 9.3 完整部署發布指南（4 步驟）
1. **建置演算法 Runner 映像檔**：
   ```bash
   cd /mnt/d/work/ai-workspace/auto-tuning-vllm
   docker build -f docker/Dockerfile.runner -t <registry>/auto-tune-vllm:runner .
   docker push <registry>/auto-tune-vllm:runner
   ```
2. **升級 `afsbox-controller` Chart**：
   ```bash
   helm upgrade -i afsbox-controller ./charts/afsbox-controller -n afsbox-system
   ```
3. **升級 `afsbox-platform` Chart**：
   ```bash
   helm upgrade -i afsbox-platform ./charts/afsbox-platform -n afsbox-system
   ```
4. **升級 `afsbox-portal` Chart**：
   ```bash
   helm upgrade -i afsbox-portal ./charts/afsbox-portal -n afsbox-system
   ```

---

## 十、各專案修改範圍與工作拆解（Commit 追蹤清單）

| 專案 | 本地路徑 | 最新 Commit | 核心修改內容 |
| :--- | :--- | :--- | :--- |
| **1. auto-tuning-vllm** | `/mnt/d/work/ai-workspace/auto-tuning-vllm` | `8a43028` | 1. 實作 `AFSBoxK8sBackend` 串接 K8s API<br>2. 實作 `_build_benchmark_suite()` 自動映射 Benchmark 參數<br>3. 支援新版 ModelServing 規格（`batchSize`, `gpuMemoryUtilization`, `prefillSettings.maxBatchTokens`）<br>4. 實作 `observedGeneration` 等候防競爭機制<br>5. 串接獨立 `BenchmarkReport` 擷取 Throughput / TTFT 指標 |
| **2. afsbox-controller** | `/mnt/d/work/afsbox/afsbox-controller` | `4471460` | 1. `modeltuning_types.go`: 擴充 `OptimizationSpec`、`ObjectiveSpec`、`ParetoFrontier`<br>2. 更新 CRD OpenAPI YAML schemas<br>3. `helpers.go`: 擴充 `setVariable` 支援 TP/PP/KV 快取，並實作 `syncGPUClaimCount` 避免 Webhook 雙源漂移<br>4. `modeltuning_controller.go`: 實作 `ensureTunerJob`，配置 `ActiveDeadlineSeconds: 7200`、`TTLSecondsAfterFinished: 86400` 與 `OPTUNA_TUNER_IMAGE` 環境變數覆寫機制 |
| **3. afsbox-platform** | `/mnt/d/work/afsbox/afsbox-platform` | `50cdd79e` | 1. `models/template.go`: `ApplyTemplateRequest` 新增 `OptimizationRequest`<br>2. `services/template.go`: `ApplyTemplate` 透傳 `OptimizationSpec`<br>3. `models/tuning.go` & `services/tuning_report.go`: 調校報告投影新增 `OptimizationView`、`BestCandidate` 與 `ParetoFrontier` |
| **4. afsbox-portal** | `/mnt/d/work/afsbox/afsbox-portal` | `5e05399a` | 1. `api-types.ts`: 定義前後端 Optuna 型別契約<br>2. `TuningConfigSection.tsx`: 部署精靈新增 Optuna 卡片、Sampler、輪數與目標切換<br>3. `ModelTuningReport.tsx`: 呈現 Optuna 摘要卡片，標註「🏆 最佳解」與「✨ 帕雷托前沿」徽章 |

---

## 十一、實機真實叢集整合驗證（NVIDIA GB10 實測記錄）

在真實 AFSBox 叢集（節點 `172.20.36.21`，配備 NVIDIA GB10 128GB Unified Memory GPU）上，使用 `facebook/opt-125m` 進行了 20 輪 TPE 貝氏最佳化自動調校，完整驗證了全鏈路的雲原生整合：

### 11.1 核心修復與實測總結
- **試驗總覽**：20 輪測試全數順利完成（`Trial 0 ~ 19`），耗時約 40 分鐘，0 失敗率。
- **最佳吞吐量冠軍**：**Trial 12**（`batch_size = 8, gpu_memory_utilization = 0.70`），Decode 聚合吞吐量達到 **`1,290.97 tok/s`**，相比 Baseline（1,189.99 tok/s）提升了 **`+8.5%`**！
- **最低首字延遲**：**Trial 16**（`batch_size = 8, gpu_memory_utilization = 0.65`），TTFT (p50) 達到極致的 **`11.73 ms`**。

1. **動態參數映射（Dynamic Parameter Mapping）**：
   - 實作在 `_map_parameters_to_serving_patch`，自動將 Optuna 推薦之下劃線命名轉為 K8s CRD CamelCase 欄位（如 `batchSize`, `gpuMemoryUtilization`），未知旗標自動包裝進 `extraCommand`。
2. **防競爭 Rollout 同步（Generation-Aware Sync）**：
   - 解決 Client Patch 後 Controller 尚未反應的競態條件，確保 `status.observedGeneration >= metadata.generation` 且 `status.phase == Ready` 後才發起壓測。
3. **獨立 BenchmarkReport 指標萃取**：
   - 自動透過 `Benchmark.status.reportRef` 追蹤取得同名 `BenchmarkReport`，解析 `spec.items[0].metrics` 萃取出 Throughput、TTFT 等即時指標供 Optuna 計算目標分數。
4. **Base 模型 Chat Template 注入**：
   - 為 OPT-125M 等純 Base 模型追加 `--chat-template=/vllm-workspace/examples/template_chatml.jinja`，解決 AIPerf `/v1/chat/completions` 缺少對話樣板回傳 HTTP 400 的問題。
5. **內網 Service 端點直連**：
   - Benchmark CR 指定 `target.endpoint.url = http://<service>.<ns>.svc.cluster.local:8000/v1`，實現純淨內網直連壓測。

---

## 十二、多推論引擎調優支援設計（vLLM, SGLang, llama.cpp）

關於 AFSBox 跨引擎抽象架構分析、各引擎通用欄位映射表、AIPerf 評測協議相容性與多引擎調優完整設計方案，請參閱獨立設計文檔：
- [AFSBox 多推論引擎調優支援架構設計 (vLLM, SGLang, llama.cpp)](file:///mnt/d/work/ai-workspace/auto-tuning-vllm/docs/multi_engine_support_design.md)

---

## 十三、兩種執行模式與實戰操作指南（Execution Modes & Operational Guide）

`auto-tune-vllm` 提供兩種核心執行模式，分別針對 **「本機研發與自訂調優」** 與 **「雲原生 Portal / CR 整合調優」**：

### 13.1 模式比較與運行拓撲

```mermaid
graph LR
    subgraph Mode1["模式一：CLI 模式（獨立 Config 驅動）"]
        direction TB
        LocalConfig["study_config.yaml<br/>(本機指定演算法/目標/參數)"] --> Runner1["auto-tune-vllm<br/>CLI Runner 容器"]
        Runner1 -->|"直連 Patch / 壓測"| Serving1["ModelServing<br/>(optuna-tune-exp)"]
        Runner1 -->|"輸出日誌與指標"| DB1["SQLite<br/>(study.db)"]
    end

    subgraph Mode2["模式二：CR 驅動模式（Portal / 雲原生整合）"]
        direction TB
        Portal["AFSBox Portal UI<br/>(精靈建立)"] -->|"Apply CR"| CR["ModelTuning CR<br/>(opt125m-cr-test)"]
        CR -->|"委派 Job / 讀取規格"| Runner2["auto-tune-vllm<br/>Runner Pod"]
        Runner2 -->|"自動拉起與刪除"| Serving2["ModelServing<br/>(opt125m-cr-test-exp)"]
        Runner2 -->|"即時回寫 variables/metrics"| CR
    end
```

| 比較維度 | 模式一：CLI 模式（獨立 Config 驅動） | 模式二：CR 驅動模式（ModelTuning CR 整合） |
| :--- | :--- | :--- |
| **核心旗標** | `--config <yaml-path>` | `--tuning-name <cr-name> [--namespace <ns>]` |
| **參數來源** | 本機 YAML 檔案（手動指定搜尋空間、目標與壓測） | Kubernetes `ModelTuning` CR（由 Runner 自動讀取並合成） |
| **搜尋空間與目標** | 完全自由自訂（支援高級 Optuna 設定、多層階梯） | 自動從 `spec.optimization`、`servingTemplate` 與 `testSuite` 萃取 |
| **適用情境** | 演算法研發、本機/單機測試、離線批次調優、精細微調 | AFSBox Portal UI 一鍵發起、Controller 委派 Job、端到端自動化 |
| **狀態回寫 (Status Sync)** | 僅記錄於本機 SQLite 資料庫（`study.db`）與終端機輸出 | 即時同步至 `ModelTuning.status`（含 `variables`, `metrics`, `paretoFrontier`） |
| **Serving 生命週期** | 可指向既有 Serving 或自動建立 | 自動依 CR 規格建立 `<tuning_name>-exp`，完工後自動刪除釋放 GPU |

---

### 13.2 模式一：CLI 模式（獨立 Config 驅動）

適用於研發階段直接測試不同演算法或特定模型，不依賴 AFSBox Portal 或 `ModelTuning` CR。

#### 1. 配置範例 (`study_config.yaml`)
```yaml
study:
  name: opt125m_cli_tpe
backend: afsbox
afsbox:
  namespace: afsbox-tokenfactory
  serving_name: opt125m-cli-exp
  serving_template:
    image: "docker.io/vllm/vllm-openai:v0.27.1"
    servedModelName: opt-125m
    model:
      valueFrom:
        kind: ClusterModelRepository
        name: facebook-opt-125m
    gpuClaim:
      className: nvidia-gb10
      requests:
        memoryMB: 16384
  cleanup_serving: true

optimization:
  approach: multi_objective
  sampler: tpe
  n_trials: 10
  objectives:
    - metric: output_tokens_per_second
      direction: maximize
    - metric: time_to_first_token_ms
      direction: minimize

benchmark:
  benchmark_type: aiperf
  model: opt-125m
  samples: 10
  rate: 4

parameters:
  batch_size:
    enabled: true
    options: [8, 16, 32, 64]
  gpu_memory_utilization:
    enabled: true
    min: 0.65
    max: 0.85
    step: 0.05
```

#### 2. 本機 Python 執行
```bash
auto-tune-vllm optimize \
  --backend afsbox \
  --config ./study_config.yaml \
  --verbose
```

#### 3. Docker 容器執行
```bash
sudo docker run --rm --net=host \
  --entrypoint auto-tune-vllm \
  -v /etc/rancher/k3s/k3s.yaml:/root/.kube/config:ro \
  -v $(pwd)/study_config.yaml:/app/config.yaml:ro \
  -v /tmp/optuna_studies:/app/optuna_studies \
  keyword1983/auto-tune-vllm:runner \
  optimize \
    --backend afsbox \
    --config /app/config.yaml \
    --verbose
```

---

### 13.3 模式二：CR 驅動模式（ModelTuning CR 整合）

適用於由 AFSBox Portal 觸發或透過 Kubernetes 原生宣告式管理的調校任務。

#### 1. ModelTuning CR 定義範例 (`opt125m-cr.yaml`)
```yaml
apiVersion: afsbox.asus.com/v1beta1
kind: ModelTuning
metadata:
  name: opt125m-cr-test
  namespace: afsbox-tokenfactory
spec:
  displayName: "OPT-125M Auto Tuning"
  servingTemplate:
    image: "docker.io/vllm/vllm-openai:v0.27.1"
    engine:
      servicePort: 8000
      type: vllm
    servedModelName: opt-125m
    model:
      valueFrom:
        kind: ClusterModelRepository
        name: facebook-opt-125m
    gpuClaim:
      className: nvidia-gb10
      requests:
        memoryMB: 16384
    batchSize: "32"
    gpuMemoryUtilization: "0.8"
  optimization:
    engine: "optuna"
    approach: "multi_objective"
    sampler: "tpe"
    nTrials: 10
    objectives:
      - metric: "output_tokens_per_second"
        direction: "maximize"
      - metric: "time_to_first_token_ms"
        direction: "minimize"
        percentile: "p50"
  testSuite:
    - name: "perf-eval"
      type: "concurrency"
      timeoutSeconds: 60
      params:
        concurrency: 4
        requestCount: 10
        streaming: true
        ignoreEOS: true
        isl: { mean: 64 }
        osl: { mean: 32 }
```

#### 2. 手動/測試執行（Docker 容器 + 原始碼熱掛載）
在叢集節點上快速驗證 Runner 行為：
```bash
sudo docker run --rm --net=host \
  --entrypoint auto-tune-vllm \
  -v /etc/rancher/k3s/k3s.yaml:/root/.kube/config:ro \
  -v /tmp/auto_tune_vllm:/app/auto_tune_vllm:ro \
  -v /tmp/optuna_studies:/app/optuna_studies \
  keyword1983/auto-tune-vllm:runner \
  optimize \
    --backend afsbox \
    --tuning-name opt125m-cr-test \
    --namespace afsbox-tokenfactory \
    --verbose
```

#### 3. 雲原生 K8s Job 部署（AFSBox Controller 自動派發）
在正式環境中，`afsbox-controller` 偵測到含 `spec.optimization` 的 CR 時，會自動建立以下 `batchv1.Job`：
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: opt125m-cr-test-tuner
  namespace: afsbox-tokenfactory
  labels:
    afsbox.asus.com/model-tuning: opt125m-cr-test
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 7200
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: afsbox-modeltuning-runner
      containers:
        - name: tuner
          image: keyword1983/auto-tune-vllm:runner
          command:
            - auto-tune-vllm
            - optimize
            - --backend
            - afsbox
            - --tuning-name
            - opt125m-cr-test
            - --namespace
            - afsbox-tokenfactory
            - --max-concurrent-trials
            - "1"
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: "2"
              memory: 2Gi
```

#### 4. CR Status 即時回寫規格
調校過程中與完成後，Runner 會自動回寫 CR 的 `status`，包含每組候選的參數、壓測結果與帕雷托前沿：
```yaml
status:
  phase: Completed
  bestCandidate: trial_0
  paretoFrontier:
    - trial_0
  candidates:
    - name: baseline_concurrency_4
      phase: Completed
      benchmarkRef: opt125m-cr-test-c0
      variables:
        batch_size: 32
        gpu_memory_utilization: 0.8
      metrics:
        output_tokens_per_second: "1151.3705"
        time_to_first_token_ms: "12.4317"
    - name: trial_1
      phase: Completed
      benchmarkRef: opt125m-cr-test-c1
      variables:
        batch_size: 32
        gpu_memory_utilization: 0.75
      metrics:
        output_tokens_per_second: "1120.2769"
        time_to_first_token_ms: "15.9512"
```



