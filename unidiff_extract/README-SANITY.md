## 跑一批 commits 做统计 sanity check（精简版）

目标：对一批 commit 批量生成 `seeds` + `impacted_set`，并汇总核心统计到 CSV，快速判断流程是否“看起来合理”。

### 0) 参数

```powershell
$REPO = "D:\locbench\python-unidiff"           # 待分析项目（必须是 git repo）
$OUT  = "D:\locbench\analyzer\runs\python-unidiff\sanity"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
```

### 1) 准备 commit 列表（示例：取最近 30 个）

```powershell
git -C $REPO log --oneline -n 30 | % { ($_ -split ' ')[0] } | Set-Content -Encoding UTF8 "$OUT\commits.txt"
```

（也可以手写 `commits.txt`，每行一个 SHA）

### 2) 批量跑 seeds + impacted_set（固定输出文件名，便于汇总）

```powershell
$commits = Get-Content "$OUT\commits.txt"
foreach ($c in $commits) {
  python .\analyzer\unidiff_extract\extract_seeds.py --repo $REPO --commit $c --out "$OUT\impacted_seeds_$c.json"
  python .\analyzer\unidiff_extract\process_impacted_set.py --repo $REPO --seeds "$OUT\impacted_seeds_$c.json" --max-hops 2 --parser ast --out "$OUT\impacted_set_$c.json"
}
```

### 3) 汇总统计到 CSV

```powershell
$rows = foreach ($c in $commits) {
  $j = Get-Content "$OUT\impacted_set_$c.json" | ConvertFrom-Json
  $g = $j.impacted | Group-Object hop -NoElement
  $hop0 = ($g | ? Name -eq "0").Count
  $hop1 = ($g | ? Name -eq "1").Count
  $hop2 = ($g | ? Name -eq "2").Count
  [pscustomobject]@{
    commit = $c
    num_seeds = $j.num_seeds
    num_impacted = $j.num_impacted
    total_functions = $j.call_graph_stats.total_functions
    total_call_relations = $j.call_graph_stats.total_call_relations
    hop0 = $hop0; hop1 = $hop1; hop2 = $hop2
  }
}
$rows | Export-Csv "$OUT\summary.csv" -NoTypeInformation -Encoding UTF8
```

### 4) 快速 sanity check 看什么（只看这几项）

- **`num_seeds`**：大多数 commit 应该 >0（除非只改非 `.py` 文件）
- **`num_impacted`**：不应长期大量为 0（否则大概率调用图太稀疏/解析失败）
- **`total_call_relations`**：应稳定且非极小值（过小通常说明 call 抽取没生效）
- **`hop0/hop1/hop2`**：通常 hop>0 会占一定比例（不然传播没意义）

