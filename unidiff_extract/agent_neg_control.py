#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Negative control experiments for docstring verifier.

Settings:
  - Ours: baseline docstring template insertion
  - Ours + DropParam: drop one real param from docstring Args
  - Ours + AddFakeParam: add one fake param into docstring Args
  - Ours + RenameParam: rename one real param in docstring Args

Outputs:
  - negative_control.json (per-commit + aggregate)
  - negative_control_table.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

# Reuse the existing agent implementation helpers (selection/patch logic).
# This directory is not packaged; add it to sys.path for a local import.
sys.path.insert(0, os.path.dirname(__file__))
import agent as agent_mod  # type: ignore  # noqa: E402


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
        return 0, out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return int(e.returncode or 1), e.output.decode("utf-8", errors="replace")


def _repo_name(repo_path: str) -> str:
    return os.path.basename(os.path.abspath(repo_path).rstrip("\\/")) or "repo"


def _runs_root() -> str:
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


def _extract_args_from_docstring(doc: str) -> List[str]:
    """
    Parse Args section into a list of param names.
    Expected format:
      Args:
          a:
          b:
    """
    if not doc:
        return []
    lines = doc.splitlines()
    # Find "Args:" (strip-equal)
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "Args:":
            start = i + 1
            break
    if start is None:
        return []

    out: List[str] = []
    pat = re.compile(r"^\s*([*]{0,2}[A-Za-z_][A-Za-z0-9_]*):\s*$")
    for ln in lines[start:]:
        if not ln.strip():
            continue
        if not ln.startswith((" ", "\t")):
            break
        m = pat.match(ln)
        if not m:
            break
        out.append(m.group(1))
    return out


def _mut_drop_param(doc: str, expected: List[str]) -> str:
    if not doc or not expected:
        return doc
    drop = expected[0]
    lines = doc.splitlines()
    out = []
    in_args = False
    dropped = False
    for ln in lines:
        if ln.strip() == "Args:":
            in_args = True
            out.append(ln)
            continue
        if in_args and (not dropped) and re.match(rf"^\s*{re.escape(drop)}:\s*$", ln):
            dropped = True
            continue
        out.append(ln)
    return "\n".join(out)


def _mut_add_fake_param(doc: str) -> str:
    if not doc:
        return doc
    lines = doc.splitlines()
    out = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if (not inserted) and ln.strip() == "Args:":
            out.append("    fake_param:")
            inserted = True
    return "\n".join(out)


def _mut_rename_param(doc: str, expected: List[str]) -> str:
    if not doc or not expected:
        return doc
    src = expected[0]
    dst = f"{src}_renamed"
    lines = doc.splitlines()
    out = []
    in_args = False
    renamed = False
    for ln in lines:
        if ln.strip() == "Args:":
            in_args = True
            out.append(ln)
            continue
        if in_args and (not renamed) and re.match(rf"^\s*{re.escape(src)}:\s*$", ln):
            indent = re.match(r"^(\s*)", ln).group(1)  # type: ignore[union-attr]
            out.append(f"{indent}{dst}:")
            renamed = True
            continue
        out.append(ln)
    return "\n".join(out)


def _verifier_strict(doc: str, expected_params: List[str]) -> Tuple[bool, str]:
    """
    Strict verifier for experiments: args_in_doc == expected_params (set equality).
    """
    args_in_doc = _extract_args_from_docstring(doc)
    a = set(args_in_doc)
    e = set(expected_params)
    if a == e:
        return True, "ok"
    missing = sorted(e - a)
    extra = sorted(a - e)
    return False, f"missing={missing} extra={extra}"


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commits-file", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--parser", choices=["auto", "ast", "treesitter"], default="ast")
    ap.add_argument("--seed-scope", choices=["all", "code_only"], default="code_only")
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--limit-targets", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        out_dir = os.path.join(_runs_root(), "runs", _repo_name(repo), "neg_control", _now_tag())
    os.makedirs(out_dir, exist_ok=True)

    if args.commits_file:
        commits = _read_commits_from_file(args.commits_file)
    else:
        commits = _read_commits_from_git(repo, args.n)
    commits = commits[: max(args.n, 0)]

    settings = [
        ("Ours", "normal"),
        ("Ours + DropParam", "should fail a lot"),
        ("Ours + AddFakeParam", "should fail a lot"),
        ("Ours + RenameParam", "should fail a lot"),
    ]

    per_commit: List[Dict] = []
    totals = {name: {"ok": 0, "total": 0} for name, _ in settings}

    for c in commits:
        t0 = time.monotonic()
        run_dir = os.path.join(out_dir, c)
        os.makedirs(run_dir, exist_ok=True)
        seeds_path = os.path.join(run_dir, f"impacted_seeds_{c}.json")
        impacted_all_path = os.path.join(run_dir, f"impacted_set_{c}_all.json")
        impacted_code_path = os.path.join(run_dir, f"impacted_set_{c}_code_only.json")

        # 1) seeds
        code, out = _run(
            ["python", os.path.join(os.path.dirname(__file__), "extract_seeds.py"), "--repo", repo, "--commit", c, "--out", seeds_path]
        )
        if code != 0:
            per_commit.append({"commit": c, "ok": False, "error": "extract_seeds_failed", "detail": out[-2000:]})
            print(f"DONE {c}")
            continue

        # 2) impacted (all vs code_only) for convergence stats
        _run(
            [
                "python",
                os.path.join(os.path.dirname(__file__), "process_impacted_set.py"),
                "--repo",
                repo,
                "--seeds",
                seeds_path,
                "--max-hops",
                str(args.max_hops),
                "--parser",
                args.parser,
                "--seed-scope",
                "all",
                "--out",
                impacted_all_path,
            ]
        )
        _run(
            [
                "python",
                os.path.join(os.path.dirname(__file__), "process_impacted_set.py"),
                "--repo",
                repo,
                "--seeds",
                seeds_path,
                "--max-hops",
                str(args.max_hops),
                "--parser",
                args.parser,
                "--seed-scope",
                args.seed_scope,
                "--out",
                impacted_code_path,
            ]
        )

        seeds_data = _load_json(seeds_path)
        impacted_data = _load_json(impacted_code_path)

        code_change_seeds = {s["qualname"] for s in seeds_data["seeds"] if s.get("seed_type") == "code_change"}

        # 3) select targets and build baseline new sources with inserted docstrings (in-memory)
        impacted_items = impacted_data.get("impacted", [])
        targets: List[Dict] = []
        file_nodes_cache: Dict[str, Dict[str, object]] = {}
        file_source_cache: Dict[str, str] = {}

        for it in impacted_items:
            hop = int(it.get("hop", 999))
            if hop > 1:
                continue
            if not it.get("is_internal", False):
                continue
            qn = it.get("qualname", "")
            src = it.get("source")
            is_code_lineage = (hop == 0 and qn in code_change_seeds) or (hop == 1 and src in code_change_seeds)
            if not is_code_lineage:
                continue

            resolved = agent_mod._resolve_qualname_to_file(repo, qn, ext=".py")  # type: ignore[attr-defined]
            if not resolved:
                continue
            _, rel_path = resolved
            if agent_mod._is_test_relpath(rel_path):  # type: ignore[attr-defined]
                continue
            rel_path = agent_mod._posix(rel_path)  # type: ignore[attr-defined]

            if rel_path not in file_nodes_cache:
                base = agent_mod._git_show_text(repo, c, rel_path)  # type: ignore[attr-defined]
                file_source_cache[rel_path] = base
                module_qual = agent_mod._module_qual_from_relpath(rel_path, ext=".py")  # type: ignore[attr-defined]
                try:
                    file_nodes_cache[rel_path] = agent_mod._collect_target_nodes(base, module_qual)  # type: ignore[attr-defined]
                except SyntaxError:
                    file_nodes_cache[rel_path] = {}

            module_qual = agent_mod._module_qual_from_relpath(rel_path, ext=".py")  # type: ignore[attr-defined]
            node = file_nodes_cache[rel_path].get(qn)  # type: ignore[union-attr]
            if node is None or not isinstance(node, (agent_mod.ast.FunctionDef, agent_mod.ast.AsyncFunctionDef)):  # type: ignore[attr-defined]
                continue
            if agent_mod.ast.get_docstring(node) is not None:  # type: ignore[attr-defined]
                continue

            targets.append(
                {
                    "qualname": qn,
                    "rel_path": rel_path,
                    "params": agent_mod._func_param_names(node),  # type: ignore[attr-defined]
                }
            )

        targets = targets[: max(args.limit_targets, 0)]

        # If no targets, skip this commit in rate denominator (nothing to verify)
        if not targets:
            per_commit.append(
                {
                    "commit": c,
                    "num_targets": 0,
                    "skipped": True,
                    "duration_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            print(f"DONE {c}")
            continue

        # Build baseline in-memory sources and extract docstrings per target
        file_to_new: Dict[str, str] = {}
        changed_targets: List[Dict] = []
        docs: Dict[str, str] = {}  # qualname -> docstring

        # apply insertions per file
        for t in targets:
            rel_path = t["rel_path"]
            if rel_path not in file_to_new:
                file_to_new[rel_path] = file_source_cache[rel_path]
            module_qual = agent_mod._module_qual_from_relpath(rel_path, ext=".py")  # type: ignore[attr-defined]
            changed, new_src, _ = agent_mod._insert_docstring_into_source(file_to_new[rel_path], t["qualname"], module_qual)  # type: ignore[attr-defined]
            if changed:
                file_to_new[rel_path] = new_src
                changed_targets.append(t)

        # Extract docstrings from updated sources
        for rel_path, src_text in file_to_new.items():
            module_qual = agent_mod._module_qual_from_relpath(rel_path, ext=".py")  # type: ignore[attr-defined]
            nodes = agent_mod._collect_target_nodes(src_text, module_qual)  # type: ignore[attr-defined]
            for t in changed_targets:
                if t["rel_path"] != rel_path:
                    continue
                node = nodes.get(t["qualname"])
                if node is None:
                    continue
                doc = agent_mod.ast.get_docstring(node)  # type: ignore[attr-defined]
                if doc:
                    docs[t["qualname"]] = doc

        # Evaluate settings per commit: pass iff all changed targets pass
        setting_ok: Dict[str, bool] = {}
        setting_notes: Dict[str, List[str]] = {name: [] for name, _ in settings}

        for name, _note in settings:
            all_ok = True
            for t in changed_targets:
                qn = t["qualname"]
                expected = t["params"]
                base_doc = docs.get(qn, "")
                if name == "Ours":
                    doc = base_doc
                elif name == "Ours + DropParam":
                    doc = _mut_drop_param(base_doc, expected)
                elif name == "Ours + AddFakeParam":
                    doc = _mut_add_fake_param(base_doc)
                elif name == "Ours + RenameParam":
                    doc = _mut_rename_param(base_doc, expected)
                else:
                    doc = base_doc

                ok, detail = _verifier_strict(doc, expected)
                if not ok:
                    all_ok = False
                    setting_notes[name].append(f"{qn}:{detail}")
            setting_ok[name] = all_ok

        # update totals (only for commits with targets)
        for name, _ in settings:
            totals[name]["total"] += 1
            if setting_ok[name]:
                totals[name]["ok"] += 1

        # convergence stats
        try:
            imp_all = _load_json(impacted_all_path).get("impacted", [])
            imp_code = _load_json(impacted_code_path).get("impacted", [])
            conv = {
                "impacted_all": len(imp_all),
                "impacted_code_only": len(imp_code),
                "impacted_code_only_ratio": (len(imp_code) / len(imp_all)) if imp_all else 0.0,
            }
        except Exception:
            conv = {}

        per_commit.append(
            {
                "commit": c,
                "num_targets": len(changed_targets),
                "settings": setting_ok,
                "notes": {k: v[:5] for k, v in setting_notes.items()},
                "convergence": conv,
                "duration_ms": int((time.monotonic() - t0) * 1000),
            }
        )
        print(f"DONE {c}")

    # Build table
    table_rows = []
    for name, note in settings:
        total = totals[name]["total"]
        ok = totals[name]["ok"]
        rate = (ok / total) if total else 0.0
        table_rows.append({"Setting": name, "Verifier ok": f"{rate:.3f}", "Notes": note})

    result = {
        "repo": repo,
        "num_commits_requested": len(commits),
        "num_commits_with_targets": totals["Ours"]["total"],
        "settings": {name: {"ok": totals[name]["ok"], "total": totals[name]["total"]} for name, _ in settings},
        "table": table_rows,
        "per_commit": per_commit,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": out_dir,
    }

    json_path = os.path.join(out_dir, "negative_control.json")
    md_path = os.path.join(out_dir, "negative_control_table.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md_lines = [
        "Setting\tVerifier ok\tNotes",
        *[f"{r['Setting']}\t{r['Verifier ok']}\t{r['Notes']}" for r in table_rows],
        "",
        f"num_commits_with_targets\t{totals['Ours']['total']}",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("=== Negative control summary ===")
    for r in table_rows:
        print(f"{r['Setting']}\t{r['Verifier ok']}\t{r['Notes']}")
    print("wrote:", md_path)
    print("wrote:", json_path)


if __name__ == "__main__":
    main()

