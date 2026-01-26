## unidiff_extract（分析代码，和待分析项目隔离）

这个目录下的脚本用于对 **任意待分析的 Git 仓库** 提取：
- commit 的 **seeds**（变更定位到函数/类/模块）
- 基于调用图的 **k-hop 传播**（impacted set）

默认情况下，所有输出都会写到：
`d:\locbench\analyzer\runs\<repo-name>\impacted_seeds\`

### 1) 从 commit 提取 seeds

在 `d:\locbench`（或任意目录）运行：

```powershell
python .\analyzer\unidiff_extract\extract_seeds.py --repo D:\locbench\python-unidiff --commit 7eb7da9
```

### 2) 基于 seeds 构建调用图并传播（k-hop）→ impacted set

```powershell
python .\analyzer\unidiff_extract\process_impacted_set.py --repo D:\locbench\python-unidiff --seeds .\analyzer\runs\python-unidiff\impacted_seeds\impacted_seeds_7eb7da9_*.json --max-hops 2 --parser ast
```

说明：
- `--seeds` 支持 glob（Windows/PowerShell 不展开 `*` 也没关系）
- 输出会写回 seeds 同目录，文件名为 `impacted_set_<commit>_<timestamp>.json`
- `--parser`：
  - `ast`：最稳定、无额外依赖
  - `auto`：本机装了 tree-sitter 时会自动用，否则回退到 ast

### （可选）安装 tree-sitter

```powershell
pip install tree_sitter tree_sitter_languages
```

