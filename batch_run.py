#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
batch_run.py  —  批量调用 analyzer.main run 的独立脚本

用法:
  # 从 git log 取最近 50 个 commit，用 llm 模式跑
  python -m analyzer.batch_run --repo ./black --n 50 --mode llm

  # 从文件读 commit 列表，用 template 模式跑
  python -m analyzer.batch_run --repo ./black --commits-file commits.txt --mode template

  # 同时跑两种模式做对比
  python -m analyzer.batch_run --repo ./black --n 20 --mode both

输出:
  <out>/summary.csv
  <out>/summary.jsonl
  <out>/aggregate.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 600) -> Tuple[int, str]:
    """Run a command and return (exit_code, combined_output)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return -1, f"[TIMEOUT] command exceeded {timeout}s"
    except Exception as e:
        return -2, f"[ERROR] {e}"


def _repo_name(repo_path: str) -> str:
    return os.path.basename(os.path.abspath(repo_path).rstrip("\\/")) or "repo"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _analyzer_root() -> str:
    """Return the analyzer package root directory (where this file lives)."""
    return os.path.dirname(os.path.abspath(__file__))


def _read_commits_from_file(path: str) -> List[str]:
    """Read commit SHAs from a text file (one per line, # comments ok)."""
    commits: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            commits.append(ln.split()[0])
    return commits


def _read_commits_from_git(repo: str, n: int) -> List[str]:
    """Get latest N commit SHAs from git log."""
    code, out = _run_cmd(["git", "-C", repo, "log", "--oneline", "-n", str(n)])
    if code != 0:
        raise RuntimeError(f"git log failed: {out}")
    commits = []
    for ln in out.splitlines():
        ln = ln.strip()
        if ln:
            commits.append(ln.split()[0])
    return commits


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_history_path(stdout: str) -> Optional[str]:
    """从 agent 的 stdout 中提取 history: 路径"""
    for ln in stdout.splitlines():
        if ln.strip().startswith("history:"):
            return ln.split("history:", 1)[1].strip()
    return None


def _count_impacted_labels(impacted_path: str) -> Dict[str, int]:
    j = _load_json(impacted_path)
    items = j.get("impacted", [])
    return {
        "impacted_total": len(items),
        "impacted_internal": sum(1 for x in items if x.get("is_internal")),
        "impacted_test": sum(1 for x in items if x.get("is_test")),
        "impacted_external": sum(1 for x in items if x.get("is_external")),
    }


# ---------------------------------------------------------------------------
# single commit run
# ---------------------------------------------------------------------------

def run_one_commit(
    repo: str,
    commit: str,
    mode: str,
    out_parent: str,
    extra_args: List[str],
    timeout: int,
) -> Dict:
    """Run analyzer.main run for a single commit and return a result dict."""
    cmd = [
        sys.executable, "-m", "analyzer.main", "run",
        "--repo", repo,
        "--commit", commit,
        "--mode", mode,
        "--run-parent", out_parent,
        *extra_args,
    ]

    t0 = time.monotonic()
    code, out = _run_cmd(cmd, timeout=timeout)
    dt_ms = int((time.monotonic() - t0) * 1000)

    history_path = _parse_history_path(out)

    row: Dict = {
        "commit": commit,
        "mode": mode,
        "ok": code == 0,
        "runner_ms": dt_ms,
        "exit_code": code,
    }

    if history_path and os.path.isfile(history_path):
        hist = _load_json(history_path)
        outputs = hist.get("outputs", {})

        row["duration_ms"] = hist.get("duration_ms")
        row["run_dir"] = hist.get("run_dir")

        # counters (both modes have these)
        counters = hist.get("counters") or {}
        row["subprocess_calls"] = counters.get("subprocess_calls")
        row["git_show_calls"] = counters.get("git_show_calls")
        row["ast_parse_calls"] = counters.get("ast_parse_calls")

        # summary (both modes have these)
        summ = hist.get("summary") or {}
        row["verifier_ok"] = summ.get("verifier_ok")
        row["num_targets"] = summ.get("num_targets")
        row["num_changed_targets"] = summ.get("num_changed_targets")
        row["num_files_changed"] = summ.get("num_files_changed")
        row["patch_added_lines"] = summ.get("patch_added_lines")
        row["patch_removed_lines"] = summ.get("patch_removed_lines")
        row["patch_total_lines"] = summ.get("patch_total_lines")

        # llm-specific fields
        if mode == "llm":
            row["style"] = summ.get("style")
            row["llm_success_rate"] = summ.get("llm_success_rate")
            llm_stats = hist.get("llm_stats") or {}
            row["llm_total_calls"] = llm_stats.get("total_calls")
            row["llm_success_calls"] = llm_stats.get("success_calls")
            row["llm_total_prompt_tokens"] = llm_stats.get("total_prompt_tokens")
            row["llm_total_completion_tokens"] = llm_stats.get("total_completion_tokens")
            row["llm_total_latency_ms"] = llm_stats.get("total_latency_ms")
            row["llm_fallbacks"] = counters.get("llm_fallbacks")

        # verifier report
        report_path = outputs.get("verifier_report")
        if report_path and os.path.isfile(report_path):
            rep = _load_json(report_path)
            row["verifier_checked"] = rep.get("num_targets_checked")

        # impacted set labels
        impacted_path = outputs.get("impacted")
        if impacted_path and os.path.isfile(impacted_path):
            lab = _count_impacted_labels(impacted_path)
            row.update(lab)
            total = lab["impacted_total"]
            row["impacted_internal_ratio"] = (lab["impacted_internal"] / total) if total else 0.0

        row["patch_path"] = outputs.get("patch")
        row["readme_path"] = outputs.get("readme")
        row["readme_generated"] = summ.get("readme_generated", False)
        row["history_path"] = history_path
    else:
        row["history_path"] = history_path
        row["stderr"] = out[-2000:]

    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="批量调用 analyzer.main run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m analyzer.batch_run --repo ./black --n 50 --mode llm
  python -m analyzer.batch_run --repo ./black --commits-file commits.txt --mode template
  python -m analyzer.batch_run --repo ./black --n 20 --mode both
        """,
    )
    ap.add_argument("--repo", required=True, help="Git repository path")
    ap.add_argument("--commits-file", default=None, help="File with commit SHAs (one per line)")
    ap.add_argument("--n", type=int, default=200, help="Number of commits from git log (default 200)")
    ap.add_argument("--mode", choices=["template", "llm", "both"], default="llm",
                     help="Agent mode: template, llm, or both (default: llm)")
    ap.add_argument("--max-hops", type=int, default=2, help="Max propagation hops (default 2)")
    ap.add_argument("--parser", choices=["auto", "ast", "treesitter"], default="treesitter")
    ap.add_argument("--seed-scope", choices=["all", "code_only"], default="code_only")
    ap.add_argument("--limit-targets", type=int, default=50)
    ap.add_argument("--style", choices=["auto", "google", "numpy", "sphinx"], default="auto",
                     help="Docstring style for LLM mode (default: auto)")
    ap.add_argument("--no-llm", action="store_true", help="Disable LLM (LLM mode only, for testing)")
    ap.add_argument("--timeout", type=int, default=600, help="Per-commit timeout in seconds (default 600)")
    ap.add_argument("--out", default=None, help="Output directory (default: analyzer/runs/<repo>/batch_run/<tag>/)")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)

    # Determine output directory
    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        out_dir = os.path.join(_analyzer_root(), "runs", _repo_name(repo), "batch_run", _now_tag())
    os.makedirs(out_dir, exist_ok=True)

    # Collect commits
    if args.commits_file:
        commits = _read_commits_from_file(args.commits_file)
    else:
        commits = _read_commits_from_git(repo, args.n)
    commits = commits[:max(args.n, 0)]

    # Determine which modes to run
    modes: List[str] = []
    if args.mode == "both":
        modes = ["template", "llm"]
    else:
        modes = [args.mode]

    # Build extra args (shared across all runs)
    extra_args: List[str] = [
        "--max-hops", str(args.max_hops),
        "--parser", args.parser,
        "--seed-scope", args.seed_scope,
        "--limit-targets", str(args.limit_targets),
    ]

    print(f"=" * 60)
    print(f"batch_run.py")
    print(f"  repo:    {repo}")
    print(f"  commits: {len(commits)}")
    print(f"  modes:   {modes}")
    print(f"  out:     {out_dir}")
    print(f"  timeout: {args.timeout}s per commit")
    print(f"=" * 60)

    rows: List[Dict] = []
    total_runs = len(commits) * len(modes)
    run_idx = 0

    jsonl_path = os.path.join(out_dir, "summary.jsonl")
    csv_path = os.path.join(out_dir, "summary.csv")
    agg_path = os.path.join(out_dir, "aggregate.json")

    # Save config
    config = {
        "repo": repo,
        "n": args.n,
        "commits_file": args.commits_file,
        "modes": modes,
        "max_hops": args.max_hops,
        "parser": args.parser,
        "seed_scope": args.seed_scope,
        "limit_targets": args.limit_targets,
        "style": args.style,
        "no_llm": args.no_llm,
        "timeout": args.timeout,
        "num_commits": len(commits),
        "commits": commits,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for commit in commits:
            for mode in modes:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] commit={commit} mode={mode}")

                mode_extra = list(extra_args)
                if mode == "llm":
                    mode_extra.extend(["--style", args.style])
                    if args.no_llm:
                        mode_extra.append("--no-llm")

                row = run_one_commit(
                    repo=repo,
                    commit=commit,
                    mode=mode,
                    out_parent=out_dir,
                    extra_args=mode_extra,
                    timeout=args.timeout,
                )

                # Print quick status
                status = "OK" if row["ok"] else "FAIL"
                targets = row.get("num_targets", "?")
                changed = row.get("num_changed_targets", "?")
                v_ok = row.get("verifier_ok", "?")
                ms = row.get("duration_ms") or row.get("runner_ms", 0)
                print(f"  -> {status}  targets={targets} changed={changed} verifier_ok={v_ok}  {ms}ms")

                rows.append(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()

    # Write CSV
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", encoding="utf-8", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Aggregate
    def _avg(key: str, subset: List[Dict] = rows) -> float:
        vals = [r[key] for r in subset if isinstance(r.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else 0.0

    def _rate(key: str, value, subset: List[Dict] = rows) -> float:
        matched = sum(1 for r in subset if r.get(key) is value)
        total = sum(1 for r in subset if r.get(key) is not None)
        return (matched / total) if total else 0.0

    agg: Dict = {
        "repo": repo,
        "num_commits": len(commits),
        "modes": modes,
        "total_runs": len(rows),
    }

    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        prefix = f"{mode}_" if len(modes) > 1 else ""
        ok_count = sum(1 for r in subset if r.get("ok"))
        agg[f"{prefix}runs"] = len(subset)
        agg[f"{prefix}success_rate"] = (ok_count / len(subset)) if subset else 0.0
        agg[f"{prefix}verifier_ok_rate"] = _rate("verifier_ok", True, subset)
        agg[f"{prefix}avg_num_targets"] = _avg("num_targets", subset)
        agg[f"{prefix}avg_num_changed_targets"] = _avg("num_changed_targets", subset)
        agg[f"{prefix}avg_patch_total_lines"] = _avg("patch_total_lines", subset)
        agg[f"{prefix}avg_duration_ms"] = _avg("duration_ms", subset)
        agg[f"{prefix}avg_runner_ms"] = _avg("runner_ms", subset)
        agg[f"{prefix}avg_impacted_total"] = _avg("impacted_total", subset)
        agg[f"{prefix}avg_impacted_internal"] = _avg("impacted_internal", subset)
        agg[f"{prefix}avg_impacted_internal_ratio"] = _avg("impacted_internal_ratio", subset)

        if mode == "llm":
            agg[f"{prefix}avg_llm_success_rate"] = _avg("llm_success_rate", subset)
            agg[f"{prefix}avg_llm_total_latency_ms"] = _avg("llm_total_latency_ms", subset)
            agg[f"{prefix}avg_llm_total_prompt_tokens"] = _avg("llm_total_prompt_tokens", subset)
            agg[f"{prefix}avg_llm_total_completion_tokens"] = _avg("llm_total_completion_tokens", subset)

    agg["config"] = {
        "max_hops": args.max_hops,
        "parser": args.parser,
        "seed_scope": args.seed_scope,
        "limit_targets": args.limit_targets,
        "style": args.style,
        "timeout": args.timeout,
    }
    agg["generated_at"] = datetime.now().isoformat(timespec="seconds")
    agg["out_dir"] = out_dir

    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"DONE - {len(rows)} runs completed")
    print(f"  {csv_path}")
    print(f"  {jsonl_path}")
    print(f"  {agg_path}")
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        ok_count = sum(1 for r in subset if r.get("ok"))
        v_ok = sum(1 for r in subset if r.get("verifier_ok") is True)
        print(f"  [{mode}] {ok_count}/{len(subset)} success, {v_ok} verifier_ok")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
