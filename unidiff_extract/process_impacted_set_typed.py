#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_impacted_set_typed.py

使用 Typed BFS + 權重衰減的影響傳播

與 process_impacted_set.py 的區別：
- 使用 impact_graph.py 的 Typed BFS 演算法
- 輸出包含 score、best_path、edge_types_on_path
- 支持消融實驗配置
"""

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

from call_graph import build_call_graph
from impact_graph import (
    PropagationConfig,
    propagate_impacts_typed,
    compute_impact_stats,
    impacted_to_json_list,
    create_ablation_config,
    DEFAULT_GAMMA,
    DEFAULT_THRESHOLD,
)


def resolve_seeds_path(seeds_arg: str) -> str:
    """
    解析 seeds 路徑（支持 glob pattern 和目錄）
    """
    if os.path.isfile(seeds_arg):
        return seeds_arg

    if any(ch in seeds_arg for ch in ("*", "?", "[")):
        matches = glob.glob(seeds_arg)
        if not matches:
            raise FileNotFoundError(f"seeds pattern 未匹配到文件: {seeds_arg}")
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        chosen = matches[0]
        if len(matches) > 1:
            print(f"提示：seeds pattern 匹配到 {len(matches)} 個文件，默認取最新的：{chosen}")
        return chosen

    if os.path.isdir(seeds_arg):
        pattern = os.path.join(seeds_arg, "impacted_seeds_*.json")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"目錄下未找到 seeds 文件（{pattern}）")
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        chosen = matches[0]
        print(f"提示：seeds 指向目錄，默認取最新的：{chosen}")
        return chosen

    raise FileNotFoundError(f"seeds 參數不是有效文件/目錄: {seeds_arg}")


def _is_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    base = os.path.basename(p)
    return ("/tests/" in p) or p.startswith("tests/") or base.startswith("test_")


def _classify_qualname(repo_path: str, qualname: str, ext: str = ".py") -> Tuple[bool, bool, bool]:
    """
    分類 qualname:
      - internal: 可以映射到 repo 內的非測試模組
      - test: 映射到測試模組
      - external: 無法映射到 repo 源碼
    """
    if not qualname:
        return False, False, True

    parts = [p for p in qualname.split(".") if p]
    if not parts:
        return False, False, True

    for i in range(len(parts), 0, -1):
        module_qual = ".".join(parts[:i])
        rel_mod = module_qual.replace(".", os.sep)

        cand1 = os.path.join(repo_path, f"{rel_mod}{ext}")
        cand2 = os.path.join(repo_path, rel_mod, f"__init__{ext}")

        if os.path.isfile(cand1):
            is_test = _is_test_path(os.path.relpath(cand1, repo_path))
            is_internal = not is_test
            return is_internal, is_test, False
        if os.path.isfile(cand2):
            is_test = _is_test_path(os.path.relpath(cand2, repo_path))
            is_internal = not is_test
            return is_internal, is_test, False

    return False, False, True


def main():
    ap = argparse.ArgumentParser(description="Typed BFS 影響傳播（帶權重衰減）")
    ap.add_argument("--repo", required=True, help="Git 倉庫根目錄")
    ap.add_argument("--seeds", required=True, help="Seeds JSON 路徑（支持 glob）")
    ap.add_argument("--ext", default=".py", help="目標文件擴展名（默認 .py）")
    ap.add_argument(
        "--parser",
        choices=["auto", "ast", "treesitter"],
        default="auto",
        help="調用圖解析器（默認 auto）",
    )
    ap.add_argument(
        "--seed-scope",
        choices=["all", "code_only"],
        default="all",
        help="傳播起點：all | code_only（僅 code_change seeds）",
    )
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, 
                    help=f"Hop 衰減係數（默認 {DEFAULT_GAMMA}）")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"分數閾值（默認 {DEFAULT_THRESHOLD}）")
    ap.add_argument("--max-hops", type=int, default=4, help="最大跳數（默認 4）")
    ap.add_argument(
        "--ablation",
        choices=["full", "uniform", "no_decay", "high_decay", "calls_only"],
        default="full",
        help="消融實驗類型（默認 full）",
    )
    ap.add_argument("--out", default=None, help="輸出 JSON 路徑")
    args = ap.parse_args()

    repo_path = os.path.abspath(args.repo)
    seeds_path = resolve_seeds_path(args.seeds)
    
    print(f"讀取 seeds: {seeds_path}")
    with open(seeds_path, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)

    seeds = seeds_data["seeds"]
    print(f"找到 {len(seeds)} 個 seeds")

    # 過濾 seeds
    seeds_used = seeds
    if args.seed_scope == "code_only":
        seeds_used = [s for s in seeds if s.get("seed_type") == "code_change"]
        print(f"seed_scope=code_only: 使用 {len(seeds_used)} / {len(seeds)} 個 seeds")

    # 構建調用圖
    print(f"構建調用圖: {repo_path}")
    call_graph = build_call_graph(repo_path, args.ext, parser_mode=args.parser)
    print(f"調用圖包含 {len(call_graph.all_functions)} 個函數")

    # 創建配置
    if args.ablation != "full":
        config = create_ablation_config(args.ablation)
        config.max_hops = args.max_hops  # 允許覆蓋 max_hops
        print(f"使用消融配置: {args.ablation}")
    else:
        config = PropagationConfig(
            gamma=args.gamma,
            threshold=args.threshold,
            max_hops=args.max_hops,
        )

    # 傳播
    print(f"從 {len(seeds_used)} 個 seeds 開始傳播...")
    impacted = propagate_impacts_typed(seeds_used, call_graph, config)

    # 分類節點
    for qn, node in impacted.items():
        is_internal, is_test, is_external = _classify_qualname(repo_path, qn, ext=args.ext)
        node.is_internal = is_internal
        node.is_test = is_test
        node.is_external = is_external

    # 統計
    stats = compute_impact_stats(impacted)
    
    # 輸出
    output = {
        "commit": seeds_data.get("commit", ""),
        "parent": seeds_data.get("parent", ""),
        "repo": seeds_data.get("repo", repo_path),
        "num_seeds": len(seeds),
        "num_seeds_used": len(seeds_used),
        "seed_scope": args.seed_scope,
        "num_impacted": len(impacted),
        "config": config.to_dict(),
        "ablation": args.ablation,
        "call_graph_stats": {
            "total_functions": len(call_graph.all_functions),
            "total_call_relations": sum(len(callees) for callees in call_graph.functions.values()),
        },
        "stats": stats,
        "impacted": impacted_to_json_list(impacted),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 輸出路徑
    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        seeds_dir = os.path.dirname(seeds_path)
        seeds_basename = os.path.basename(seeds_path)
        out_basename = seeds_basename.replace("impacted_seeds_", "impacted_typed_")
        out_path = os.path.join(seeds_dir, out_basename)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n輸出: {out_path}")
    print(f"受影響節點: {len(impacted)}")
    print(f"按跳數分佈: {stats['by_hop']}")
    print(f"分數分佈: min={stats['score_distribution']['min']:.4f}, "
          f"max={stats['score_distribution']['max']:.4f}, "
          f"mean={stats['score_distribution']['mean']:.4f}")


if __name__ == "__main__":
    main()
