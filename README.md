# Analyzer — 增量文档维护工具

针对每次代码变动，自动分析所有受影响的函数，使用 LLM 更新其 docstring，并生成变更总结。

---

## 目录

1. [工作流总览](#工作流总览)
2. [架构与模块](#架构与模块)
3. [数据 I/O 说明](#数据-io-说明)
4. [快速开始](#快速开始)
5. [命令参考](#命令参考)
6. [输出文件说明](#输出文件说明)
7. [环境配置](#环境配置)
8. [常见问题](#常见问题)

---

## 工作流总览

```
Git Commit (代码变动)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: 提取种子 (seed_extractor)                                │
│   输入: repo path + commit SHA                                   │
│   处理: 解析 git diff → 定位变更行 → AST 映射到函数/类定义        │
│   输出: impacted_seeds_<commit>.json                             │
├──────────────────────────────────────────────────────────────────┤
│ Step 2: 构建调用图 + 影响传播 (call_graph + impact_propagator)    │
│   输入: repo path + seeds                                        │
│   处理: 扫描全仓库构建调用图 → 从 seed 节点 BFS 传播             │
│   输出: impacted_set_<commit>.json                               │
├──────────────────────────────────────────────────────────────────┤
│ Step 3: 检测 Docstring 风格 (style_detector)     [仅 LLM 模式]   │
│   输入: repo path                                                │
│   处理: 采样文件 → 提取现有 docstring → 正则分类风格              │
│   输出: 风格标识 (google / numpy / sphinx)                       │
├──────────────────────────────────────────────────────────────────┤
│ Step 4: 选择目标函数                                              │
│   输入: impacted_set                                             │
│   过滤: hop ≤ 1, is_internal, 非测试文件, function/async 类型     │
│   输出: targets 列表 (≤ limit-targets 个)                        │
├──────────────────────────────────────────────────────────────────┤
│ Step 5: 生成 Docstring                                           │
│   模板模式: 生成 TODO 占位 docstring                              │
│   LLM 模式: 调用 LLM 生成 → 失败则 fallback 到模板               │
│   输出: docstring_patch_<commit>.diff                            │
├──────────────────────────────────────────────────────────────────┤
│ Step 6: AST 验证                                                 │
│   输入: patch + 原始源码                                          │
│   处理: AST 解析检查 + docstring 存在性 + 参数覆盖率检查          │
│   输出: verifier_report_<commit>.json                            │
├──────────────────────────────────────────────────────────────────┤
│ Step 7: 生成变更总结 README                          [计划中]     │
│   输入: seeds + impacted_set + patch                             │
│   处理: 调用 LLM 汇总本次所有变更                                 │
│   输出: change_summary_<commit>.md                               │
└──────────────────────────────────────────────────────────────────┘
```

### 两种运行模式

| 模式 | 命令参数 | 特点 | 是否需要 LLM |
|------|---------|------|-------------|
| **Template** | `--mode template` | 生成 TODO 占位 docstring，速度快 | 否 |
| **LLM** | `--mode llm` | 调用 LLM 生成高质量 docstring，失败自动 fallback | 是 |

---

## 架构与模块

```
analyzer/
├── main.py                    # CLI 入口，命令分发
│
├── core/                      # 核心分析引擎（纯静态分析，不涉及 LLM）
│   ├── seed_extractor.py      #   Step 1: git diff → 变更种子
│   ├── call_graph.py          #   Step 2: 构建函数调用图
│   ├── impact_propagator.py   #   Step 2: BFS 影响传播
│   └── style_detector.py      #   Step 3: docstring 风格检测
│
├── agents/                    # Agent 编排层（串联 core + llm 完成完整流程）
│   ├── llm_agent.py           #   LLM 模式 agent（Steps 1-6 全流程）
│   ├── template_agent.py      #   模板模式 agent（Steps 1-6 全流程）
│   └── base_agent.py          #   基类（部分实现）
│
├── llm/                       # LLM 调用层
│   ├── llm_client.py          #   OpenAI SDK 封装，重试/限流/统计
│   ├── prompt_loader.py       #   Prompt 模板加载器（从文件读取 + 渲染）
│   └── prompts/               #   所有 Prompt 模板（独立文件，不写在源码中）
│       ├── docstring_system.txt   # docstring 生成 - system prompt
│       ├── docstring_user.txt     # docstring 生成 - user prompt
│       ├── readme_system.txt      # 变更总结 - system prompt [预留]
│       └── readme_user.txt        # 变更总结 - user prompt [预留]
│
├── experiments/               # 实验/批量测试
│   ├── batch_runner.py        #   批量处理多个 commit
│   └── negative_control.py    #   负对照实验
│
└── runs/                      # 输出目录（自动生成）
    └── <repo>/
        ├── agent/             #   模板模式输出
        ├── agent_llm/         #   LLM 模式输出
        └── agent_batch/       #   批量测试输出
```

### 模块间调用关系

```
main.py
  ├─ run --mode template ──→ agents/template_agent.py
  │                            ├── core/seed_extractor.py
  │                            ├── core/call_graph.py
  │                            └── core/impact_propagator.py
  │
  ├─ run --mode llm ──→ agents/llm_agent.py
  │                       ├── core/seed_extractor.py
  │                       ├── core/call_graph.py
  │                       ├── core/impact_propagator.py
  │                       ├── core/style_detector.py
  │                       └── llm/llm_client.py
  │                             └── llm/prompt_loader.py → llm/prompts/*.txt
  │
  ├─ extract-seeds ──→ core/seed_extractor.py
  ├─ propagate ──→ core/call_graph.py + core/impact_propagator.py
  └─ batch ──→ experiments/batch_runner.py → agents/template_agent.py (循环)
```

---

## 数据 I/O 说明

### 输入

| 输入项 | 来源 | 说明 |
|--------|------|------|
| `--repo` | 本地 Git 仓库路径 | 如 `d:\locbench\rich` |
| `--commit` | Git commit SHA | 短 SHA（如 `36fe3f7c`）或完整 SHA |
| `config.yaml` | 项目根目录 `d:\locbench\config.yaml` | LLM API 配置（仅 LLM 模式需要） |

### 各步骤数据流

```
[输入] repo + commit
         │
         ▼
Step 1 ──→ impacted_seeds_<commit>.json
         │   字段: commit, parent, seeds[{path, qualname, kind, seed_type, line, span}]
         │
         ▼
Step 2 ──→ impacted_set_<commit>.json
         │   字段: commit, num_impacted, impacted[{qualname, hop, reason, source,
         │          callers, callees, is_internal, is_test, is_external}]
         │
         ▼
Step 3 ──→ style: "google" | "numpy" | "sphinx"  (内存，记录在 history)
         │
         ▼
Step 4 ──→ targets: [{qualname, rel_path, lineno}]  (内存)
         │   过滤条件: hop≤1, is_internal, 非测试, function/async 类型
         │
         ▼
Step 5 ──→ docstring_patch_<commit>.diff
         │   格式: unified diff，可直接 git apply
         │
         ▼
Step 6 ──→ verifier_report_<commit>.json
         │   字段: ok, num_targets_checked, results[{qualname, ok, errors}],
         │          per_target[{action, changed, used_llm, llm_latency_ms, tokens}]
         │
         ▼
汇总   ──→ history.json
             字段: repo, commit, steps[], counters, llm_stats, summary, outputs
```

### 输出目录结构

单次运行（以 commit `36fe3f7c` 为例）：

```
analyzer/runs/rich/agent_llm/36fe3f7c_20260209_144152/
├── impacted_seeds_36fe3f7c.json    # 变更种子（哪些函数被直接修改）
├── impacted_set_36fe3f7c.json      # 受影响函数集（传播后的完整列表）
├── docstring_patch_36fe3f7c.diff   # Docstring 变更补丁
├── verifier_report_36fe3f7c.json   # AST 验证报告
└── history.json                     # 执行历史与统计
```

批量运行：

```
analyzer/runs/rich/agent_batch/<timestamp>/
├── <commit>_<timestamp>/            # 每个 commit 的子目录（同上述结构）
├── summary.csv                      # CSV 格式统计摘要
├── summary.jsonl                    # JSONL 格式详细记录
└── aggregate.json                   # 聚合统计
```

---

## 快速开始

### 1. 获取一个测试 commit

```powershell
cd d:\locbench\rich
git log --oneline -5
# 选择一个有代码变更的 commit SHA
```

### 2. 运行单次分析

```powershell
cd d:\locbench

# 模板模式（快速，不需要 LLM）
python -m analyzer.main run --repo .\rich --commit 36fe3f7c --mode template

# LLM 模式（需要 config.yaml 配置）
python -m analyzer.main run --repo .\rich --commit 36fe3f7c --mode llm
```

### 3. 查看输出

运行完成后终端会打印所有输出文件路径，例如：

```
=== agent_llm outputs ===
patch:    ...\docstring_patch_36fe3f7c.diff
verifier: ...\verifier_report_36fe3f7c.json
history:  ...\history.json
seeds:    ...\impacted_seeds_36fe3f7c.json
impacted: ...\impacted_set_36fe3f7c.json
style: google
num_targets: 1, num_changed: 1, verifier_ok: True
llm_calls: 1, success: 1, fallback: 0
```

---

## 命令参考

### `run` — 运行完整工作流

```powershell
python -m analyzer.main run --repo <path> --commit <sha> [options]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--repo` | Git 仓库路径 | **必填** |
| `--commit` | Commit SHA | **必填** |
| `--mode` | `template` 或 `llm` | `template` |
| `--max-hops` | 影响传播最大跳数 | `2` |
| `--parser` | 解析器: `ast` / `treesitter` / `auto` | `treesitter` |
| `--seed-scope` | 种子范围: `code_only` / `all` | `code_only` |
| `--limit-targets` | 最多处理的目标函数数 | `50` |
| `--style` | Docstring 风格（仅 LLM）: `auto` / `google` / `numpy` / `sphinx` | `auto` |
| `--no-llm` | 禁用 LLM，强制使用模板（仅 LLM 模式） | - |
| `--run-parent` | 自定义输出父目录 | - |

### `extract-seeds` — 仅提取种子

```powershell
python -m analyzer.main extract-seeds --repo <path> --commit <sha> [--out <path>]
```

### `propagate` — 构建调用图 + 影响传播

```powershell
python -m analyzer.main propagate --repo <path> --seeds <json> [--mode simple|typed] [--max-hops N]
```

### `batch` — 批量处理

```powershell
python -m analyzer.main batch --repo <path> [--n 200] [--commits-file <file>]
```

---

## 输出文件说明

### `impacted_seeds_<commit>.json` — 变更种子

从 git diff 解析出的直接变更函数：

```json
{
  "commit": "36fe3f7c",
  "parent": "9a99acc9...",
  "num_seeds": 1,
  "seeds": [
    {
      "path": "rich/progress.py",
      "qualname": "rich.progress.Progress.reset",
      "kind": "function",
      "seed_type": "code_change",
      "line": 1496,
      "span": [1478, 1515]
    }
  ]
}
```

### `impacted_set_<commit>.json` — 受影响函数集

BFS 传播后的完整影响范围：

```json
{
  "num_impacted": 1,
  "max_hops": 2,
  "impacted": [
    {
      "qualname": "rich.progress.Progress.reset",
      "hop": 0,
      "reason": "direct_change",
      "callers": [],
      "callees": ["task._reset"],
      "is_internal": true,
      "is_test": false,
      "is_external": false
    }
  ]
}
```

### `docstring_patch_<commit>.diff` — Docstring 变更补丁

标准 unified diff 格式，可直接 `git apply`：

```diff
--- a/rich/progress.py
+++ b/rich/progress.py
@@ -1486,16 +1486,21 @@
     ) -> None:
-        """Reset a task so completed is 0 and the clock is reset.
+        """Reset a task so completed steps are set to zero and the elapsed time is reset.
         ...
```

### `verifier_report_<commit>.json` — AST 验证报告

```json
{
  "ok": true,
  "num_targets_checked": 1,
  "results": [
    { "qualname": "rich.progress.Progress.reset", "ok": true, "errors": [] }
  ],
  "per_target": [
    {
      "qualname": "rich.progress.Progress.reset",
      "action": "updated",
      "changed": true,
      "used_llm": true,
      "llm_latency_ms": 15442,
      "llm_prompt_tokens": 13372,
      "llm_completion_tokens": 608
    }
  ]
}
```

### `history.json` — 执行历史

完整的运行记录，包含每步输出路径、计数器、LLM 统计和最终摘要：

```json
{
  "repo": "d:\\locbench\\rich",
  "commit": "36fe3f7c",
  "duration_ms": 17370,
  "counters": {
    "subprocess_calls": 1,
    "git_show_calls": 1,
    "ast_parse_calls": 3,
    "llm_calls": 1,
    "llm_fallbacks": 0
  },
  "llm_stats": {
    "total_calls": 1,
    "success_calls": 1,
    "total_prompt_tokens": 13372,
    "total_completion_tokens": 608,
    "total_latency_ms": 15442
  },
  "summary": {
    "num_targets": 1,
    "num_changed_targets": 1,
    "verifier_ok": true,
    "style": "google"
  }
}
```

---

## 环境配置

### 依赖安装

```powershell
pip install openai pyyaml
# 可选（更准确的解析）
pip install tree_sitter tree_sitter_python
```

### LLM 配置（仅 LLM 模式需要）

编辑 `d:\locbench\config.yaml`：

```yaml
llm_chat:
  api_base_url: "http://your-server:8000/v1"
  api_key: "your-key"
  model_name: "qwen3:235b"
```

配置文件查找顺序：
1. 环境变量 `LOCBENCH_CONFIG` 指定的路径
2. `d:\locbench\config.yaml`（项目根目录）
3. 当前工作目录的 `config.yaml`

### 可用的测试仓库

| 仓库 | 路径 | 说明 |
|------|------|------|
| rich | `d:\locbench\rich` | Rich 终端格式化库 |
| black | `d:\locbench\black` | Python 代码格式化工具 |
| fastapi | `d:\locbench\fastapi` | 现代 Web 框架 |
| sqlalchemy | `d:\locbench\sqlalchemy` | SQL 工具包 |

---

## 常见问题

### Q: LLM 调用失败怎么办？

检查 `config.yaml` 配置和网络连接。也可用 `--no-llm` 强制走模板模式：

```powershell
python -m analyzer.main run --repo .\rich --commit <sha> --mode llm --no-llm
```

### Q: 如何只执行分析（不生成 docstring）？

分步执行前两步：

```powershell
# 提取种子
python -m analyzer.main extract-seeds --repo .\rich --commit <sha>

# 影响传播（确认 seeds JSON 路径后执行）
python -m analyzer.main propagate --repo .\rich --seeds <seeds.json路径>
```

### Q: Prompt 在哪里修改？

所有 prompt 模板在 `analyzer/llm/prompts/` 目录下，以 `.txt` 文件存放，修改后无需改动源码。

### Q: 如何测试 Typed BFS（带权重衰减的传播）？

```powershell
python -m analyzer.main propagate --repo .\rich --seeds <seeds.json> --mode typed --max-hops 4
```

---

**最后更新**：2026-02-09
