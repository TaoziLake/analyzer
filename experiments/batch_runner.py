#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量跑 commits，做 agent-level sanity check。

输出：
  - summary.csv
  - summary.jsonl
  - aggregate.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
        return 0, out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return int(e.returncode or 1), e.output.decode("utf-8", errors="replace")


def _repo_name(repo_path: str) -> str:
    return os.path.basename(os.path.abspath(repo_path).rstrip("\\/")) or "repo"


def _runs_root() -> str:
    # <locbench>/analyzer/experiments/batch_runner.py -> <locbench>/analyzer
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_commits_from_file(path: str) -> List[str]:
    commits: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            commits.append(ln.split()[0])
    return commits


def _read_commits_from_git(repo: str, n: int) -> List[str]:
    code, out = _run(["git", "-C", repo, "log", "--oneline", "-n", str(n)])
    if code != 0:
        raise RuntimeError(out)
    commits = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        commits.append(ln.split()[0])
    return commits


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_impacted_labels(impacted_path: str) -> Dict[str, int]:
    j = _load_json(impacted_path)
    impacted = j.get("impacted", [])
    internal = sum(1 for x in impacted if x.get("is_internal"))
    test = sum(1 for x in impacted if x.get("is_test"))
    external = sum(1 for x in impacted if x.get("is_external"))
    total = len(impacted)
    return {
        "impacted_total": total,
        "impacted_internal": internal,
        "impacted_test": test,
        "impacted_external": external,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commits-file", default=None, help="每行一个 commit sha；如果不提供则用 git log")
    ap.add_argument("--n", type=int, default=200, help="从 git log 取多少个 commit（默认 200）")
    ap.add_argument("--parser", choices=["auto", "ast", "treesitter"], default="ast")
    ap.add_argument("--seed-scope", choices=["all", "code_only"], default="code_only")
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--limit-targets", type=int, default=50)
    ap.add_argument(
        "--out",
        default=None,
        help="输出目录（默认 analyzer/runs/<repo>/agent_batch/<tag>/）",
    )
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        out_dir = os.path.join(_runs_root(), "runs", _repo_name(repo), "agent_batch", _now_tag())
    os.makedirs(out_dir, exist_ok=True)

    if args.commits_file:
        commits = _read_commits_from_file(args.commits_file)
    else:
        commits = _read_commits_from_git(repo, args.n)

    commits = commits[: max(args.n, 0)]

    rows: List[Dict] = []
    jsonl_path = os.path.join(out_dir, "summary.jsonl")
    csv_path = os.path.join(out_dir, "summary.csv")
    agg_path = os.path.join(out_dir, "aggregate.json")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for idx, c in enumerate(commits, 1):
            t0 = time.monotonic()
            import sys
            code, out = _run(
                [
                    sys.executable,
                    "-m", "analyzer.agents.template_agent",
                    "--repo",
                    repo,
                    "--commit",
                    c,
                    "--parser",
                    args.parser,
                    "--seed-scope",
                    args.seed_scope,
                    "--max-hops",
                    str(args.max_hops),
                    "--limit-targets",
                    str(args.limit_targets),
                    "--run-parent",
                    out_dir,
                ]
            )
            dt_ms = int((time.monotonic() - t0) * 1000)

            # Parse history path from stdout
            history_path = None
            for ln in out.splitlines():
                if ln.strip().startswith("history:"):
                    history_path = ln.split("history:", 1)[1].strip()
                    break

            row: Dict = {
                "commit": c,
                "ok": code == 0,
                "runner_ms": dt_ms,
                "agent_exit_code": code,
            }

            if history_path and os.path.isfile(history_path):
                hist = _load_json(history_path)
                outputs = hist.get("outputs", {})
                patch_path = outputs.get("patch")
                report_path = outputs.get("verifier_report")
                impacted_path = outputs.get("impacted")

                row["duration_ms"] = hist.get("duration_ms")
                row["subprocess_calls"] = (hist.get("counters") or {}).get("subprocess_calls")
                row["git_show_calls"] = (hist.get("counters") or {}).get("git_show_calls")
                row["ast_parse_calls"] = (hist.get("counters") or {}).get("ast_parse_calls")

                summ = hist.get("summary") or {}
                row["verifier_ok"] = summ.get("verifier_ok")
                row["num_targets"] = summ.get("num_targets")
                row["num_changed_targets"] = summ.get("num_changed_targets")
                row["num_files_changed"] = summ.get("num_files_changed")
                row["patch_added_lines"] = summ.get("patch_added_lines")
                row["patch_removed_lines"] = summ.get("patch_removed_lines")
                row["patch_total_lines"] = summ.get("patch_total_lines")

                if report_path and os.path.isfile(report_path):
                    rep = _load_json(report_path)
                    row["verifier_checked"] = rep.get("num_targets_checked")

                if impacted_path and os.path.isfile(impacted_path):
                    lab = _count_impacted_labels(impacted_path)
                    row.update(lab)
                    total = lab["impacted_total"]
                    row["impacted_internal_ratio"] = (lab["impacted_internal"] / total) if total else 0.0

                row["patch_path"] = patch_path
                row["history_path"] = history_path
            else:
                row["history_path"] = history_path
                row["stderr"] = out[-2000:]

            rows.append(row)
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate
    total = len(rows)
    ok = sum(1 for r in rows if r.get("ok"))
    verifier_ok = sum(1 for r in rows if r.get("verifier_ok") is True)
    verifier_total = sum(1 for r in rows if r.get("verifier_ok") is not None)

    def _avg(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else 0.0

    agg = {
        "repo": repo,
        "num_commits": total,
        "agent_success_rate": (ok / total) if total else 0.0,
        "verifier_ok_rate": (verifier_ok / verifier_total) if verifier_total else 0.0,
        "avg_num_targets": _avg("num_targets"),
        "avg_patch_total_lines": _avg("patch_total_lines"),
        "avg_duration_ms": _avg("duration_ms"),
        "avg_runner_ms": _avg("runner_ms"),
        "avg_subprocess_calls": _avg("subprocess_calls"),
        "avg_git_show_calls": _avg("git_show_calls"),
        "avg_ast_parse_calls": _avg("ast_parse_calls"),
        "avg_impacted_total": _avg("impacted_total"),
        "avg_impacted_internal": _avg("impacted_internal"),
        "avg_impacted_internal_ratio": _avg("impacted_internal_ratio"),
        "seed_scope": args.seed_scope,
        "parser": args.parser,
        "max_hops": args.max_hops,
        "limit_targets": args.limit_targets,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": out_dir,
    }
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    print("Wrote:")
    print(" -", csv_path)
    print(" -", jsonl_path)
    print(" -", agg_path)


if __name__ == "__main__":
    main()

