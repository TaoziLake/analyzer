# Analyzer 測試指南

本指南將帶你從零開始測試 Analyzer 項目，包括如何找到測試用的 commit、運行完整工作流，以及查看結果。

## 快速參考

### 最常用的命令

```powershell
# 1. 找到 commit
cd d:\locbench\rich
git log --oneline -10

# 2. 運行完整工作流（模板模式，最快）
cd d:\locbench
python -m analyzer.main run --repo .\rich --commit <commit_sha> --mode template
python -m analyzer.main run --repo .\rich --commit 73ee8232 --mode template
# 3. 運行完整工作流（LLM 模式，需要配置）
python -m analyzer.main run --repo .\rich --commit 73ee8232 --mode llm

# 4. 批量測試
python -m analyzer.main batch --repo .\rich --n 50
```

### 實際示例（rich 項目）

```powershell
# 完整流程示例
cd d:\locbench\rich
git log --oneline -5
# 選擇：73ee8232

cd d:\locbench
python -m analyzer.main run --repo .\rich --commit 73ee8232 --mode template
```

---

## 目錄

1. [環境準備](#環境準備)
2. [找到測試用的 Commit](#找到測試用的-commit)
3. [快速開始](#快速開始)
4. [完整工作流](#完整工作流)
5. [單步執行](#單步執行)
6. [批量測試](#批量測試)
7. [查看結果](#查看結果)
8. [常見問題](#常見問題)

---

## 環境準備

### 1. 確保項目依賴已安裝

```powershell
# 進入項目根目錄
cd d:\locbench

# 安裝依賴（如果還沒安裝）
pip install openai pyyaml
# 可選：tree-sitter（用於更準確的解析）
pip install tree_sitter tree_sitter_python
```

### 2. 配置 LLM（可選，僅 LLM 模式需要）

編輯 `d:\locbench\config.yaml`：

```yaml
llm_chat:
  api_base_url: "http://your-server:8000/v1"
  api_key: "your-key"
  model_name: "qwen3:235b"
```

如果沒有配置 LLM，可以使用模板模式（`--mode template`），不需要 LLM。

### 3. 準備測試倉庫

本項目已包含以下測試倉庫：
- `rich` - Rich 終端格式化庫
- `black` - Python 代碼格式化工具
- `fastapi` - 現代 Web 框架
- `sqlalchemy` - SQL 工具包

如果沒有這些倉庫，可以克隆：

```powershell
# 克隆 rich 倉庫（示例）
cd d:\locbench
git clone https://github.com/Textualize/rich.git
```

### 4. 驗證安裝

```powershell
# 檢查 Python 版本（需要 3.8+）
python --version

# 檢查依賴
python -c "import openai, yaml; print('Dependencies OK')"

# 檢查 analyzer 模組是否可以導入
python -c "import analyzer; print('Analyzer module OK')"

# 檢查 main.py 是否可用
python -m analyzer.main --help
```

如果出現錯誤，請檢查：
- Python 版本是否 >= 3.8
- 依賴是否已安裝：`pip install openai pyyaml`
- 當前目錄是否在 `d:\locbench`

---

## 找到測試用的 Commit

### 方法 1：查看最近的 commits（推薦）

```powershell
# 進入倉庫目錄
cd d:\locbench\rich

# 查看最近的 20 個 commits
git log --oneline -20

# 輸出示例：
# 14713ac Fix issue with console markup
# 5cdb4b6 Add new feature for table rendering
# a1b2c3d Update documentation
# ...
```

選擇一個有代碼變更的 commit（避免純文檔或配置變更）。

### 方法 2：查找包含特定關鍵字的 commits

```powershell
# 查找包含 "def" 的 commits（通常表示函數變更）
git log --oneline --grep="def" -10

# 查找修改特定文件的 commits
git log --oneline -- "src/rich/console.py" -10
```

### 方法 3：查看特定文件的變更歷史

```powershell
# 查看某個文件的變更歷史
git log --oneline --follow -- "src/rich/console.py" -10

# 查看包含 Python 文件變更的 commits
git log --oneline -- "*.py" -10
```

### 方法 4：使用 git show 預覽 commit 內容

```powershell
# 查看 commit 的詳細變更
git show 14713ac --stat

# 只查看變更的文件列表
git show 14713ac --name-only

# 查看具體的 diff
git show 14713ac
```

**建議**：選擇一個修改了 1-5 個 Python 文件的 commit，這樣測試結果更容易理解。

**實際示例**（rich 項目）：
```powershell
cd d:\locbench\rich
git log --oneline -10
# 選擇：73ee8232 (fix fonts) - 這個 commit 修改了字體相關代碼
# 或：36fe3f7c (docstring) - 這個 commit 可能包含 docstring 變更
```

---

## 快速開始

### 模板模式（不需要 LLM）

```powershell
# 從項目根目錄執行
cd d:\locbench

# 運行完整工作流（模板模式）
python -m analyzer.main run --repo .\rich --commit 14713ac --mode template
```

### LLM 模式（需要配置 LLM）

```powershell
# 運行完整工作流（LLM 模式）
python -m analyzer.main run --repo .\rich --commit 14713ac --mode llm
```

### 參數說明

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--repo` | Git 倉庫路徑（相對於當前目錄） | 必填 |
| `--commit` | Commit SHA（完整或短 SHA） | 必填 |
| `--mode` | Agent 模式：`template` 或 `llm` | `template` |
| `--max-hops` | 影響傳播的最大跳數 | `2` |
| `--parser` | 解析器：`ast` / `treesitter` / `auto` | `ast` |
| `--seed-scope` | 種子範圍：`code_only` / `all` | `code_only` |
| `--limit-targets` | 最多處理的目標函數數 | `50` |
| `--style` | Docstring 風格（僅 LLM）：`auto` / `google` / `numpy` / `sphinx` | `auto` |

---

## 完整工作流

### 步驟 1：找到測試 commit

```powershell
cd d:\locbench\rich
git log --oneline -10
# 選擇一個 commit，例如：14713ac
```

### 步驟 2：運行完整工作流

```powershell
cd d:\locbench

# 模板模式（快速，不需要 LLM）
python -m analyzer.main run --repo .\rich --commit 14713ac --mode template

# 或 LLM 模式（需要 LLM 配置）
python -m analyzer.main run --repo .\rich --commit 14713ac --mode llm --style auto
```

### 步驟 3：查看輸出

工作流會自動執行以下步驟：

1. **提取種子**（Extract Seeds）
   - 從 git diff 提取變更的函數/類
   - 輸出：`analyzer/runs/rich/agent/<commit>_<timestamp>/impacted_seeds_<commit>.json`

2. **構建調用圖**（Build Call Graph）
   - 掃描整個倉庫構建函數調用關係
   - 輸出：調用圖數據（內存中）

3. **傳播影響**（Propagate Impacts）
   - 從種子節點進行 k-hop 傳播
   - 輸出：`impacted_set_<commit>.json`

4. **選擇目標**（Select Targets）
   - 過濾出需要生成 docstring 的函數
   - 條件：hop <= 1, internal, 由 code_change seeds 推導

5. **生成 Docstring**（Generate Docstrings）
   - 模板模式：生成 TODO 模板
   - LLM 模式：調用 LLM 生成（失敗時 fallback 到模板）

6. **驗證**（Verify）
   - AST 級別驗證 docstring 格式
   - 檢查參數是否在 docstring 中提及
   - 輸出：`verifier_report_<commit>.json`

### 輸出文件結構

```
analyzer/runs/rich/agent/<commit>_<timestamp>/
├── impacted_seeds_<commit>.json      # 變更種子
├── impacted_set_<commit>.json       # 受影響函數集合
├── docstring_patch_<commit>.diff    # 生成的 patch 文件
├── verifier_report_<commit>.json     # 驗證報告
└── history.json                      # 執行歷史和統計
```

---

## 單步執行

如果你想逐步執行，可以使用單步命令：

### 步驟 1：提取種子

```powershell
python -m analyzer.main extract-seeds --repo .\rich --commit 14713ac

# 輸出：analyzer/runs/rich/impacted_seeds/impacted_seeds_14713ac_<timestamp>.json
```

### 步驟 2：傳播影響

```powershell
# 使用簡單 BFS（等權重）
python -m analyzer.main propagate --repo .\rich --seeds analyzer\runs\rich\impacted_seeds\impacted_seeds_14713ac_*.json --mode simple

# 或使用 Typed BFS（帶權重衰減）
python -m analyzer.main propagate --repo .\rich --seeds analyzer\runs\rich\impacted_seeds\impacted_seeds_14713ac_*.json --mode typed --max-hops 4
```

### 步驟 3：運行 Agent（生成 Docstring）

```powershell
# 模板模式
python -m analyzer.main run --repo .\rich --commit 14713ac --mode template

# LLM 模式
python -m analyzer.main run --repo .\rich --commit 14713ac --mode llm
```

---

## 批量測試

### 批量處理多個 commits

```powershell
# 處理最近的 50 個 commits
python -m analyzer.main batch --repo .\rich --n 50

# 從文件讀取 commits（每行一個 SHA）
# 先創建 commits.txt：
# 14713ac
# 5cdb4b6
# a1b2c3d
python -m analyzer.main batch --repo .\rich --commits-file commits.txt
```

### 批量測試輸出

批量測試會生成：

```
analyzer/runs/rich/agent_batch/<timestamp>/
├── summary.csv          # CSV 格式的統計摘要
├── summary.jsonl        # JSONL 格式的詳細記錄
└── aggregate.json       # 聚合統計
```

查看結果：

```powershell
# 查看 CSV（可用 Excel 打開）
notepad analyzer\runs\rich\agent_batch\<timestamp>\summary.csv

# 查看聚合統計
type analyzer\runs\rich\agent_batch\<timestamp>\aggregate.json
```

---

## 查看結果

### 1. 查看生成的 Patch

```powershell
# 查看 diff 文件
notepad analyzer\runs\rich\agent\<commit>_<timestamp>\docstring_patch_<commit>.diff

# 或在 Git 中應用 patch
cd d:\locbench\rich
git apply --check analyzer\..\..\runs\rich\agent\<commit>_<timestamp>\docstring_patch_<commit>.diff
```

### 2. 查看驗證報告

```powershell
# 查看 JSON 報告
type analyzer\runs\rich\agent\<commit>_<timestamp>\verifier_report_<commit>.json

# 或使用 Python 格式化查看
python -m json.tool analyzer\runs\rich\agent\<commit>_<timestamp>\verifier_report_<commit>.json
```

### 3. 查看執行歷史

```powershell
# 查看完整的執行歷史
type analyzer\runs\rich\agent\<commit>_<timestamp>\history.json

# 格式化查看
python -m json.tool analyzer\runs\rich\agent\<commit>_<timestamp>\history.json
```

### 4. 查看受影響函數列表

```powershell
# 查看 impacted set
python -m json.tool analyzer\runs\rich\agent\<commit>_<timestamp>\impacted_set_<commit>.json | findstr "qualname"
```

---

## 完整測試示例

### 示例：測試 rich 項目的一個 commit

```powershell
# 1. 進入 rich 目錄，找到一個 commit
cd d:\locbench\rich
git log --oneline -10
# 輸出示例：
# 1d402e0c fix dates
# f2a1c3b8 Merge pull request #3944
# 2e5a5dad changelog
# 73ee8232 fix fonts
# 36fe3f7c docstring
# 選擇一個有代碼變更的 commit，例如：73ee8232

# 2. 預覽 commit 變更（可選）
git show 73ee8232 --stat
# 查看修改了哪些文件

# 3. 返回項目根目錄，運行測試
cd d:\locbench
python -m analyzer.main run --repo .\rich --commit 73ee8232 --mode template

# 4. 查看輸出
# 終端會顯示：
# === agent outputs ===
# patch: analyzer/runs/rich/agent/73ee8232_20260128_123456/docstring_patch_73ee8232.diff
# verifier: analyzer/runs/rich/agent/73ee8232_20260128_123456/verifier_report_73ee8232.json
# history: analyzer/runs/rich/agent/73ee8232_20260128_123456/history.json
# num_targets: 5, num_files_changed: 2, verifier_ok: True

# 5. 查看生成的 patch
notepad analyzer\runs\rich\agent\73ee8232_20260128_123456\docstring_patch_73ee8232.diff
```

### 示例：使用 LLM 模式

```powershell
# 1. 確保 config.yaml 已配置 LLM（已配置在 d:\locbench\config.yaml）
# 2. 運行 LLM 模式
python -m analyzer.main run --repo .\rich --commit 73ee8232 --mode llm --style auto

# 3. 查看結果（會包含 LLM 統計）
# 輸出會顯示：
# llm_calls: 5, success: 4, fallback: 1
type analyzer\runs\rich\agent_llm\73ee8232_<timestamp>\history.json
```

---

## 常見問題

### Q1: 找不到 commit？

**A**: 確保：
- 倉庫路徑正確（使用相對路徑 `.\rich` 或絕對路徑）
- commit SHA 正確（可以使用短 SHA，如 `14713ac`）
- 倉庫是 git 倉庫（有 `.git` 目錄）

```powershell
# 檢查倉庫
cd d:\locbench\rich
git status
git log --oneline -5
```

### Q2: LLM 調用失敗？

**A**: 
- 檢查 `config.yaml` 配置是否正確
- 檢查網絡連接
- 使用 `--no-llm` 標誌強制使用模板模式

```powershell
python -m analyzer.main run --repo .\rich --commit 14713ac --mode llm --no-llm
```

### Q3: 輸出目錄不存在？

**A**: 程序會自動創建目錄。如果失敗，檢查：
- 寫入權限
- 磁盤空間
- 路徑長度（Windows 路徑限制）

### Q4: 如何查看詳細的執行日誌？

**A**: 查看 `history.json` 文件，包含：
- 每個步驟的執行時間
- 統計計數器（subprocess_calls, git_show_calls, ast_parse_calls）
- 摘要信息（num_targets, num_files_changed, verifier_ok）

### Q5: 如何測試 Typed BFS？

**A**: 使用 `propagate` 命令的 `typed` 模式：

```powershell
python -m analyzer.main propagate --repo .\rich --seeds <seeds.json> --mode typed --max-hops 4
```

### Q6: 如何只測試特定文件？

**A**: 目前不支持文件過濾，但可以：
1. 選擇只修改特定文件的 commit
2. 在結果中過濾 `impacted_set` JSON 文件

---

## 進階用法

### 自定義參數

```powershell
# 增加傳播跳數
python -m analyzer.main run --repo .\rich --commit 14713ac --mode template --max-hops 3

# 使用 tree-sitter 解析器（更準確但更慢）
python -m analyzer.main run --repo .\rich --commit 14713ac --mode template --parser treesitter

# 包含測試文件的變更
python -m analyzer.main run --repo .\rich --commit 14713ac --mode template --seed-scope all

# 限制目標數量
python -m analyzer.main run --repo .\rich --commit 14713ac --mode template --limit-targets 10
```

### 指定輸出目錄

```powershell
# 自定義輸出父目錄
python -m analyzer.main run --repo .\rich --commit 14713ac --mode template --run-parent .\my_output
```

### 批量測試自定義參數

```powershell
# 批量測試，自定義參數
python -m analyzer.main batch --repo .\rich --n 100 --max-hops 3 --parser ast --limit-targets 30
```

---

## 輸出文件說明

### `impacted_seeds_<commit>.json`

包含從 git diff 提取的變更種子：

```json
{
  "commit": "14713ac",
  "parent": "5cdb4b6",
  "repo": "d:\\locbench\\rich",
  "num_seeds": 3,
  "seeds": [
    {
      "path": "src/rich/console.py",
      "qualname": "rich.console.Console.print",
      "kind": "function",
      "seed_type": "code_change"
    }
  ]
}
```

### `impacted_set_<commit>.json`

包含受影響的函數集合：

```json
{
  "num_impacted": 15,
  "impacted": [
    {
      "qualname": "rich.console.Console.print",
      "hop": 0,
      "is_internal": true,
      "is_test": false
    }
  ]
}
```

### `docstring_patch_<commit>.diff`

標準 unified diff 格式的 patch 文件，可以直接應用：

```diff
--- a/src/rich/console.py
+++ b/src/rich/console.py
@@ -100,6 +100,12 @@
     def print(self, *objects):
+        """TODO: docstring
+
+        Args:
+            objects:
+        """
         ...
```

### `verifier_report_<commit>.json`

驗證報告：

```json
{
  "ok": true,
  "num_targets_checked": 5,
  "results": [
    {
      "qualname": "rich.console.Console.print",
      "ok": true,
      "errors": []
    }
  ]
}
```

### `history.json`

完整的執行歷史：

```json
{
  "repo": "d:\\locbench\\rich",
  "commit": "14713ac",
  "duration_ms": 1234,
  "summary": {
    "num_targets": 5,
    "num_changed_targets": 3,
    "num_files_changed": 2,
    "verifier_ok": true
  },
  "steps": [...]
}
```

---

## 聯繫與支持

如有問題，請查看：
- 項目架構設計：`analyzer/项目架构设计.md`
- 開發進度：`analyzer/开发进度与计划.md`
- 舊版 README：`analyzer/unidiff_extract/README.md`

---

**最後更新**：2026-01-28
