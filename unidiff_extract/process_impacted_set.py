#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
第二步：从 seeds 构建调用图并传播影响范围（k-hop）→ 输出 impacted set JSON
"""

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

from call_graph import build_call_graph, propagate_impacts


def resolve_seeds_path(seeds_arg: str) -> str:
    """
    Windows/PowerShell 下通配符不一定会被 shell 展开，因此这里支持：
      - 直接传文件路径
      - 传 glob pattern（例如 impacted_seeds_7eb7da9_*.json）
      - 传目录（默认取目录里最新的 impacted_seeds_*.json）

    如果 pattern 匹配多个文件，默认选择“最新修改”的那个。
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
            print(f"提示：seeds pattern 匹配到 {len(matches)} 个文件，默认取最新的：{chosen}")
        return chosen

    if os.path.isdir(seeds_arg):
        pattern = os.path.join(seeds_arg, "impacted_seeds_*.json")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"目录下未找到 seeds 文件（{pattern}）")
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        chosen = matches[0]
        print(f"提示：seeds 指向目录，默认取最新的：{chosen}")
        return chosen

    raise FileNotFoundError(f"seeds 参数不是有效文件/目录，也无法作为 pattern 展开: {seeds_arg}")


def _is_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    base = os.path.basename(p)
    return ("/tests/" in p) or p.startswith("tests/") or base.startswith("test_")


def _classify_qualname(repo_path: str, qualname: str, ext: str = ".py") -> Tuple[bool, bool, bool]:
    """
    Heuristic classification:
      - internal: qualname can be mapped back to a repo module file (*.py or package/__init__.py) and not test
      - test:     mapped module file is under tests/ or test_*.py
      - external: cannot map back to repo source file
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
    ap = argparse.ArgumentParser(description="从 seeds 构建调用图并传播影响范围（k-hop）")
    ap.add_argument("--repo", required=True, help="Git 仓库根目录路径（待分析项目）")
    ap.add_argument("--seeds", required=True, help="extract_seeds.py 输出的 seeds JSON（支持 glob）")
    ap.add_argument("--ext", default=".py", help="目标文件扩展名 (默认 .py)")
    ap.add_argument(
        "--parser",
        choices=["auto", "ast", "treesitter"],
        default="auto",
        help="调用图解析器：auto|ast|treesitter (默认 auto)",
    )
    ap.add_argument("--max-hops", type=int, default=2, help="最大传播跳数 (默认 2)")
    ap.add_argument(
        "--seed-scope",
        choices=["all", "code_only"],
        default="all",
        help="传播起点 seeds 范围：all（默认）| code_only（仅 seed_type=code_change）",
    )
    ap.add_argument("--out", default=None, help="输出 JSON 文件路径（默认写回 seeds 同目录）")
    args = ap.parse_args()

    repo_path = os.path.abspath(args.repo)
    seeds_path = resolve_seeds_path(args.seeds)
    print(f"读取 seeds 文件: {seeds_path}")
    with open(seeds_path, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)

    seeds = seeds_data["seeds"]
    print(f"找到 {len(seeds)} 个 seeds")

    # Optional: only propagate from code_change seeds (avoid test-only diffusion)
    seeds_used = seeds
    if args.seed_scope == "code_only":
        seeds_used = [s for s in seeds if s.get("seed_type") == "code_change"]
        print(f"seed_scope=code_only: 使用 {len(seeds_used)} / {len(seeds)} 个 seeds 作为起点")

    print(f"构建调用图 for {repo_path}...")
    call_graph = build_call_graph(repo_path, args.ext, parser_mode=args.parser)
    print(f"调用图包含 {len(call_graph.all_functions)} 个函数")

    print(f"从 {len(seeds_used)} 个 seeds 开始传播影响 (最大 {args.max_hops} 跳)...")
    impacted = propagate_impacts(seeds_used, call_graph, args.max_hops)

    # Tag impacted nodes
    for item in impacted.values():
        qn = item.get("qualname", "")
        is_internal, is_test, is_external = _classify_qualname(repo_path, qn, ext=args.ext)
        item["is_internal"] = is_internal
        item["is_test"] = is_test
        item["is_external"] = is_external

    output = {
        "commit": seeds_data.get("commit", ""),
        "parent": seeds_data.get("parent", ""),
        "repo": seeds_data.get("repo", repo_path),
        "num_seeds": len(seeds),
        "num_seeds_used": len(seeds_used),
        "seed_scope": args.seed_scope,
        "num_impacted": len(impacted),
        "max_hops": args.max_hops,
        "call_graph_stats": {
            "total_functions": len(call_graph.all_functions),
            "total_call_relations": sum(len(callees) for callees in call_graph.functions.values()),
        },
        "impacted": list(impacted.values()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        seeds_dir = os.path.dirname(seeds_path)
        seeds_basename = os.path.basename(seeds_path)
        out_basename = seeds_basename.replace("impacted_seeds_", "impacted_set_")
        out_path = os.path.join(seeds_dir, out_basename)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"输出写入: {out_path}")
    print(f"总共 {len(impacted)} 个受影响的函数")

    hop_counts: Dict[int, int] = {}
    for item in impacted.values():
        hop = int(item["hop"])
        hop_counts[hop] = hop_counts.get(hop, 0) + 1

    print("按跳数统计:")
    for hop in sorted(hop_counts.keys()):
        print(f"  {hop} 跳: {hop_counts[hop]} 个函数")


if __name__ == "__main__":
    main()

