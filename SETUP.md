# 環境配置（Mac）

## 1. Python

建議使用 **Python 3.10、3.11 或 3.12**（專案在 3.13 上也可運行，但可選依賴 `tree_sitter_languages` 的 wheel 可能不全）。

```bash
# 檢查版本
python3 --version
```

若未安裝或想用 pyenv 管理多版本：

```bash
# Homebrew 安裝
brew install python@3.12

# 或 pyenv
brew install pyenv
pyenv install 3.12.0
pyenv local 3.12.0
```

## 2. 虛擬環境（推薦）

在專案根目錄（或 `analyzer` 同級目錄）執行：

```bash
cd /Users/syhe/2026
python3 -m venv .venv
source .venv/bin/activate
```

## 3. 安裝依賴

```bash
pip install --upgrade pip
pip install -r analyzer/requirements.txt
```

若 **只用 AST 解析器**（不用 tree-sitter），可只裝必需依賴：

```bash
pip install openai PyYAML
```

## 4. LLM 配置（若要用 LLM 版 agent）

在專案根目錄或 `analyzer` 上層建立 `config.yaml`，或設置環境變數 `LOCBENCH_CONFIG` 指向該文件：

```yaml
llm_chat:
  api_base_url: "http://your-server:8000/v1"
  api_key: "your-key"
  model_name: "qwen3:235b"
```

## 5. 驗證

```bash
# 在 2026 目錄下，確保已 activate .venv
cd /Users/syhe/2026
python -c "
from analyzer.llm import load_llm_config
print('LLM config OK')
" 2>/dev/null || python -c "
import yaml, openai
print('openai & PyYAML OK')
"
```

單跑腳本（例如模板版 agent，不調 LLM）：

```bash
python analyzer/unidiff_extract/agent.py --repo /path/to/your/repo --commit <sha>
```

---

**常見問題**

- **ModuleNotFoundError: No module named 'analyzer'**  
  在專案根目錄（含 `analyzer` 資料夾的那一層）執行腳本，或設置 `PYTHONPATH=/Users/syhe/2026`。
- **tree_sitter / tree_sitter_languages 安裝失敗**  
  不影響使用，腳本會回退到 `--parser ast`；若要 tree-sitter，可只裝 `tree-sitter` 和 `tree-sitter-python`。
