# Docstring Analyzer

從 Git commit 自動定位變更函數，並生成 docstring patch。

## 腳本總覽

| 腳本 | 功能 |
|------|------|
| `extract_seeds.py` | 從 commit diff 提取變更種子（函數/類） |
| `call_graph.py` | 調用圖構建（ast / tree-sitter） |
| `impact_graph.py` | **Typed BFS + 權重衰減**（核心演算法） |
| `process_impacted_set.py` | 基於調用圖計算 k-hop 受影響集合（等權重版） |
| `process_impacted_set_typed.py` | **Typed BFS 版本**（帶分數、路徑） |
| `agent.py` | 模板版 agent（生成 TODO docstring） |
| `agent_llm.py` | **LLM 版 agent**（調用 LLM 生成 docstring） |
| `agent_batch.py` | 批量執行多 commits |
| `agent_neg_control.py` | 負面控制實驗 |
| `docstring_style.py` | 自動檢測 docstring 風格 |

### LLM 模組（獨立目錄）

```
analyzer/llm/
├── __init__.py
└── llm_client.py    # LLM API 封裝（OpenAI SDK）
```

---

## 快速開始

### 1. 單 Commit 執行（模板版）

```powershell
python .\analyzer\unidiff_extract\agent.py --repo .\black --commit abc123
```

### 2. 單 Commit 執行（LLM 版）

```powershell
python .\analyzer\unidiff_extract\agent_llm.py --repo .\fastapi --commit 1c4fc96c
```

### 3. 批量執行

```powershell
python .\analyzer\unidiff_extract\agent_batch.py --repo .\black --n 200
```

---

## 參數說明

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `--repo` | 必填 | Git 倉庫路徑 |
| `--commit` | 必填 | Commit SHA |
| `--parser` | `ast` | 解析器：`ast` / `treesitter` / `auto` |
| `--seed-scope` | `code_only` | 種子範圍：`code_only`（排除測試）/ `all` |
| `--max-hops` | `2` | 調用圖傳播跳數 |
| `--limit-targets` | `50` | 最多處理的目標數 |
| `--style` | `auto` | docstring 風格（僅 LLM 版） |
| `--no-llm` | - | 禁用 LLM，使用模板（僅 LLM 版） |

---

## LLM 配置

在 `d:\locbench\config.yaml` 中配置：

```yaml
llm_chat:
  api_base_url: "http://your-server:8000/v1"
  api_key: "your-key"
  model_name: "qwen3:235b"
```

---

## 輸出目錄

```
analyzer/runs/<repo>/<agent_type>/<commit>_<timestamp>/
├── impacted_seeds_<commit>.json    # 變更種子
├── impacted_set_<commit>.json      # 受影響集合
├── docstring_patch_<commit>.diff   # patch 檔案
├── verifier_report_<commit>.json   # 驗證報告
└── history.json                    # 執行歷史
```

---

## 批量統計結果

| Repo | Commits | Agent 成功率 | Verifier 通過率 | 平均 Targets |
|------|---------|-------------|----------------|--------------|
| python-unidiff | 143 | 99.3% | 100% | 1.02 |
| black | 200 | 98.5% | 99.5% | 1.10 |
| fastapi | 200 | 100% | 100% | 1.33 |

---

## Typed BFS 影響傳播（核心演算法）

### 公式

```
score(v) = max_{p: seed→v} ∏_{e∈p} w(type(e)) · γ^|p|
```

- `w(type(e))`: 邊類型權重
- `γ`: hop 衰減係數（默認 0.85）
- 使用 priority queue（類似 Dijkstra，但乘法/取 max）

### 使用 Typed BFS 版本

```powershell
# 基本用法
python .\analyzer\unidiff_extract\process_impacted_set_typed.py --repo .\black --seeds <seeds.json>

# 自定義參數
python .\analyzer\unidiff_extract\process_impacted_set_typed.py \
    --repo .\black \
    --seeds <seeds.json> \
    --gamma 0.85 \
    --threshold 0.01 \
    --max-hops 4
```

### 消融實驗

```powershell
# 等權重 baseline（所有邊 w=0.7）
python .\analyzer\unidiff_extract\process_impacted_set_typed.py --repo .\black --seeds <seeds.json> --ablation uniform

# 無 hop 衰減（γ=1.0）
python .\analyzer\unidiff_extract\process_impacted_set_typed.py --repo .\black --seeds <seeds.json> --ablation no_decay

# 高衰減（γ=0.5）
python .\analyzer\unidiff_extract\process_impacted_set_typed.py --repo .\black --seeds <seeds.json> --ablation high_decay

# 只考慮 CALLS 邊
python .\analyzer\unidiff_extract\process_impacted_set_typed.py --repo .\black --seeds <seeds.json> --ablation calls_only
```

### 輸出格式

```json
{
  "impacted": [
    {
      "qualname": "module.Class.method",
      "score": 0.7225,
      "hop": 2,
      "reason": "propagated",
      "source": "module.other_func",
      "best_path": ["seed_func", "intermediate", "module.Class.method"],
      "edge_types_on_path": ["CALLS", "CALLED_BY"],
      "is_internal": true,
      "is_test": false
    }
  ],
  "stats": {
    "by_hop": {"0": 5, "1": 12, "2": 8},
    "score_distribution": {"min": 0.01, "max": 1.0, "mean": 0.45}
  }
}
```

### 邊類型權重（可配置）

| 邊類型 | 默認權重 | 說明 |
|--------|----------|------|
| `EXPOSES_CLI` | 0.95 | CLI 參數暴露（最高） |
| `READS_CONFIG` | 0.90 | 讀取配置 |
| `USES_ENV` | 0.90 | 環境變數 |
| `DOCS` | 0.85 | 文檔關係 |
| `DATA_DEP` | 0.80 | 資料依賴 |
| `CTRL_DEP` | 0.75 | 控制依賴 |
| `CALLS` | 0.70 | 函數調用 |
| `CALLED_BY` | 0.60 | 被調用 |
| `DEFINES` | 0.50 | 定義關係 |

---

## 底層工具

### 提取 Seeds

```powershell
python .\analyzer\unidiff_extract\extract_seeds.py --repo .\black --commit abc123
```

### 計算 Impacted Set（等權重版）

```powershell
python .\analyzer\unidiff_extract\process_impacted_set.py --repo .\black --seeds <seeds.json> --max-hops 2
```

### 計算 Impacted Set（Typed BFS 版）

```powershell
python .\analyzer\unidiff_extract\process_impacted_set_typed.py --repo .\black --seeds <seeds.json>
```

---

## 依賴

```bash
pip install openai pyyaml
# 可選
pip install tree_sitter tree_sitter_python
```
