#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
agent.py

输入：repo + commit
流程：
  1) extract_seeds.py 生成 seeds
  2) process_impacted_set.py 生成 impacted set
  3) 选 targets（hop<=1 且 internal 且由 code_change seeds 推导）
  4) 生成 docstring patch（仅补缺失 docstring；内容为参数列表模板）
  5) 跑 docstring verifier（AST 级校验）
输出：patch + verifier report + history
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_COUNTERS: Dict[str, int] = {
    "subprocess_calls": 0,
    "git_show_calls": 0,
    "ast_parse_calls": 0,
}


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> str:
    _COUNTERS["subprocess_calls"] += 1
    out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def _repo_name(repo_path: str) -> str:
    return os.path.basename(os.path.abspath(repo_path).rstrip("\\/")) or "repo"


def _runs_root() -> str:
    # <locbench>/analyzer/agents/template_agent.py -> <locbench>/analyzer
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _posix(path: str) -> str:
    return path.replace("\\", "/")


def _git_show_text(repo: str, commit: str, rel_path: str) -> str:
    _COUNTERS["git_show_calls"] += 1
    return _run(["git", "-C", repo, "show", f"{commit}:{_posix(rel_path)}"])


def _is_test_relpath(rel_path: str) -> bool:
    p = _posix(rel_path).lower()
    base = os.path.basename(p)
    return ("/tests/" in p) or p.startswith("tests/") or base.startswith("test_")


def _resolve_qualname_to_file(repo_path: str, qualname: str, ext: str = ".py") -> Optional[Tuple[str, str]]:
    """
    Return (abs_file_path, rel_file_path) for best-effort module resolution.
    Tries longest prefix of qualname as module name, mapping:
      a.b.c -> <repo>/a/b/c.py  OR  <repo>/a/b/c/__init__.py
    """
    parts = [p for p in qualname.split(".") if p]
    for i in range(len(parts), 0, -1):
        module_qual = ".".join(parts[:i])
        rel_mod = module_qual.replace(".", os.sep)
        cand1 = os.path.join(repo_path, f"{rel_mod}{ext}")
        cand2 = os.path.join(repo_path, rel_mod, f"__init__{ext}")
        if os.path.isfile(cand1):
            rel = os.path.relpath(cand1, repo_path)
            return cand1, rel
        if os.path.isfile(cand2):
            rel = os.path.relpath(cand2, repo_path)
            return cand2, rel
    return None


def _qualname_suffix_after_module(qualname: str, rel_file_path: str, ext: str = ".py") -> List[str]:
    """
    Given qualname and resolved rel_file_path, return the nested name stack after module.
    Example:
      rel_file_path = unidiff/patch.py => module_qual = unidiff.patch
      qualname = unidiff.patch.PatchSet._parse => suffix = [PatchSet, _parse]
    """
    rel = _posix(rel_file_path)
    if rel.endswith(ext):
        rel = rel[: -len(ext)]
    module_qual = rel.replace("/", ".")
    if module_qual.endswith(".__init__"):
        module_qual = module_qual[: -len(".__init__")]
    if qualname == module_qual:
        return []
    if qualname.startswith(module_qual + "."):
        tail = qualname[len(module_qual) + 1 :]
        return [p for p in tail.split(".") if p]
    # fallback: try from the front by removing common prefix
    qparts = [p for p in qualname.split(".") if p]
    mparts = [p for p in module_qual.split(".") if p]
    k = 0
    while k < min(len(qparts), len(mparts)) and qparts[k] == mparts[k]:
        k += 1
    return qparts[k:]


def _collect_target_nodes(source: str, module_qual: str) -> Dict[str, ast.AST]:
    """
    Build mapping qualname -> node (ClassDef/FunctionDef/AsyncFunctionDef).
    """
    # Some repos contain regex strings like "\s" in normal literals which triggers SyntaxWarning
    # when compiling/parsing source. We ignore that noise here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=SyntaxWarning,
            message=r".*invalid escape sequence.*",
        )
        _COUNTERS["ast_parse_calls"] += 1
        tree = ast.parse(source)
    stack: List[str] = []
    out: Dict[str, ast.AST] = {}

    class V(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            stack.append(node.name)
            out[".".join([p for p in [module_qual, *stack] if p])] = node
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            stack.append(node.name)
            out[".".join([p for p in [module_qual, *stack] if p])] = node
            self.generic_visit(node)
            stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            stack.append(node.name)
            out[".".join([p for p in [module_qual, *stack] if p])] = node
            self.generic_visit(node)
            stack.pop()

    V().visit(tree)
    return out


def _module_qual_from_relpath(rel_file_path: str, ext: str = ".py") -> str:
    rel = _posix(rel_file_path)
    if rel.endswith(ext):
        rel = rel[: -len(ext)]
    module_qual = rel.replace("/", ".")
    if module_qual.endswith(".__init__"):
        module_qual = module_qual[: -len(".__init__")]
    return module_qual


def _func_param_names(node: ast.AST) -> List[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    a = node.args
    names: List[str] = []
    names += [x.arg for x in getattr(a, "posonlyargs", [])]
    names += [x.arg for x in a.args]
    if a.vararg is not None:
        names.append("*" + a.vararg.arg)
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg is not None:
        names.append("**" + a.kwarg.arg)
    # drop common implicit receiver
    names = [n for n in names if n not in ("self", "cls")]
    return names


def _indent_of_line(line: str) -> str:
    m = re.match(r"^(\s*)", line)
    return m.group(1) if m else ""


def _make_docstring(params: List[str]) -> List[str]:
    if not params:
        return ['"""TODO: docstring"""']
    lines = ['"""TODO: docstring', "", "Args:"]
    for p in params:
        lines.append(f"    {p}:")
    lines.append('"""')
    return lines


@dataclass
class PatchEdit:
    rel_path: str
    old: List[str]
    new: List[str]


def _insert_docstring_into_source(
    source: str,
    qualname: str,
    module_qual: str,
) -> Tuple[bool, str, Optional[str]]:
    """
    Returns (changed, new_source, message).
    Supports both inserting new docstring and updating existing one.
    """
    try:
        nodes = _collect_target_nodes(source, module_qual)
    except SyntaxError as e:
        return False, source, f"syntax_error: {e}"

    node = nodes.get(qualname)
    if node is None:
        return False, source, "node_not_found"
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False, source, "not_a_function"

    if not node.body:
        return False, source, "empty_body"

    lines = source.splitlines()
    existing_doc = ast.get_docstring(node)

    if existing_doc is not None:
        # Update existing docstring: find and replace it
        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, (ast.Str, ast.Constant)):
            doc_start = first_stmt.lineno - 1  # 0-based
            doc_end = getattr(first_stmt, "end_lineno", first_stmt.lineno)  # 1-based end
            body_indent = _indent_of_line(lines[doc_start]) if doc_start < len(lines) else "    "
            doc_lines = _make_docstring(_func_param_names(node))
            doc_block = [body_indent + dl if dl else "" for dl in doc_lines]
            new_lines = lines[:doc_start] + doc_block + lines[doc_end:]
            return True, "\n".join(new_lines) + ("\n" if source.endswith("\n") else ""), "updated"
        else:
            return False, source, "docstring_node_not_found"
    else:
        # Insert new docstring
        first_stmt = node.body[0]
        insert_at = getattr(first_stmt, "lineno", None)
        if not isinstance(insert_at, int) or insert_at <= 0:
            insert_at = node.lineno + 1
        # insert before first statement (1-based -> 0-based)
        idx = max(insert_at - 1, 0)

        # determine indentation from first statement line if exists
        body_indent = _indent_of_line(lines[idx]) if idx < len(lines) else "    "
        doc_lines = _make_docstring(_func_param_names(node))
        doc_block = [body_indent + dl if dl else "" for dl in doc_lines]

        new_lines = lines[:idx] + doc_block + lines[idx:]
        return True, "\n".join(new_lines) + ("\n" if source.endswith("\n") else ""), "inserted"


def _verify_docstrings(
    repo_path: str,
    commit: str,
    targets: List[Dict],
    file_to_new_source: Dict[str, str],
) -> Dict:
    """
    Very lightweight verifier:
      - file parses
      - each target qualname that was modified has a docstring
      - docstring mentions each param name (string containment)
    """
    results = []
    ok = True

    for t in targets:
        qn = t["qualname"]
        rel_path = t["rel_path"]
        if rel_path not in file_to_new_source:
            continue
        src = file_to_new_source[rel_path]
        try:
            module_qual = _module_qual_from_relpath(rel_path, ext=".py")
            nodes = _collect_target_nodes(src, module_qual)
            node = nodes.get(qn)
            if node is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                results.append({"qualname": qn, "rel_path": rel_path, "ok": False, "errors": ["node_not_found"]})
                ok = False
                continue
            doc = ast.get_docstring(node)
            if not doc:
                results.append({"qualname": qn, "rel_path": rel_path, "ok": False, "errors": ["missing_docstring"]})
                ok = False
                continue
            missing = []
            for p in t.get("params", []):
                if p not in doc:
                    missing.append(p)
            if missing:
                results.append(
                    {"qualname": qn, "rel_path": rel_path, "ok": False, "errors": ["missing_params"], "missing": missing}
                )
                ok = False
            else:
                results.append({"qualname": qn, "rel_path": rel_path, "ok": True, "errors": []})
        except SyntaxError as e:
            results.append({"qualname": qn, "rel_path": rel_path, "ok": False, "errors": ["syntax_error"], "detail": str(e)})
            ok = False

    return {
        "ok": ok,
        "num_targets_checked": len(results),
        "results": results,
        "commit": commit,
        "repo": repo_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="待分析 git repo 根目录")
    ap.add_argument("--commit", required=True, help="commit sha (short ok)")
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--parser", choices=["auto", "ast", "treesitter"], default="ast")
    ap.add_argument("--seed-scope", choices=["all", "code_only"], default="code_only")
    ap.add_argument("--limit-targets", type=int, default=50, help="最多生成 docstring 的 targets 数（默认 50）")
    ap.add_argument(
        "--run-parent",
        default=None,
        help="可选：指定输出父目录（便于批量跑）；默认写到 analyzer/runs/<repo>/agent/",
    )
    args = ap.parse_args()

    t0 = time.monotonic()
    repo_path = os.path.abspath(args.repo)
    commit = args.commit
    if args.run_parent:
        run_parent = os.path.abspath(args.run_parent)
    else:
        run_parent = os.path.join(_runs_root(), "runs", _repo_name(repo_path), "agent")
    run_dir = os.path.join(run_parent, f"{commit}_{_now_tag()}")
    os.makedirs(run_dir, exist_ok=True)

    history = {
        "repo": repo_path,
        "commit": commit,
        "run_dir": run_dir,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
    }

    seeds_path = os.path.join(run_dir, f"impacted_seeds_{commit}.json")
    impacted_path = os.path.join(run_dir, f"impacted_set_{commit}.json")
    patch_path = os.path.join(run_dir, f"docstring_patch_{commit}.diff")
    report_path = os.path.join(run_dir, f"verifier_report_{commit}.json")
    history_path = os.path.join(run_dir, "history.json")

    # 1) seeds
    history["steps"].append({"step": "extract_seeds", "out": seeds_path})
    from ..core.seed_extractor import extract_seeds as extract_seeds_func
    extract_seeds_func(repo_path, commit, out_path=seeds_path)

    # 2) impacted
    history["steps"].append({"step": "process_impacted_set", "out": impacted_path})
    import json
    from ..core import build_call_graph, propagate_impacts
    from ..core.impact_propagator import _classify_qualname
    
    with open(seeds_path, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)
    seeds = seeds_data["seeds"]
    if args.seed_scope == "code_only":
        seeds = [s for s in seeds if s.get("seed_type") == "code_change"]
    
    call_graph = build_call_graph(repo_path, parser_mode=args.parser)
    impacted = propagate_impacts(seeds, call_graph, max_hops=args.max_hops, seed_scope=args.seed_scope)
    
    # Classify nodes
    for qn, item in impacted.items():
        is_internal, is_test, is_external = _classify_qualname(repo_path, qn)
        item["is_internal"] = is_internal
        item["is_test"] = is_test
        item["is_external"] = is_external
    
    output = {
        "commit": seeds_data.get("commit", ""),
        "parent": seeds_data.get("parent", ""),
        "repo": repo_path,
        "num_seeds": len(seeds_data["seeds"]),
        "num_seeds_used": len(seeds),
        "seed_scope": args.seed_scope,
        "num_impacted": len(impacted),
        "max_hops": args.max_hops,
        "impacted": list(impacted.values()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(impacted_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    seeds_data = json.load(open(seeds_path, "r", encoding="utf-8"))
    impacted_data = json.load(open(impacted_path, "r", encoding="utf-8"))

    seeds = seeds_data["seeds"]
    code_change_seeds = {s["qualname"] for s in seeds if s.get("seed_type") == "code_change"}

    impacted_items = impacted_data["impacted"]

    # 3) pick targets
    targets: List[Dict] = []
    # Cache parsed module nodes per file at this commit
    file_nodes_cache: Dict[str, Dict[str, ast.AST]] = {}
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

        resolved = _resolve_qualname_to_file(repo_path, qn, ext=".py")
        if not resolved:
            continue
        _, rel_path = resolved
        if _is_test_relpath(rel_path):
            continue
        rel_path = _posix(rel_path)

        # Keep real function defs that exist in this commit (with or without docstring).
        if rel_path not in file_nodes_cache:
            base = _git_show_text(repo_path, commit, rel_path)
            file_source_cache[rel_path] = base
            module_qual = _module_qual_from_relpath(rel_path, ext=".py")
            try:
                file_nodes_cache[rel_path] = _collect_target_nodes(base, module_qual)
            except SyntaxError:
                file_nodes_cache[rel_path] = {}

        module_qual = _module_qual_from_relpath(rel_path, ext=".py")
        node = file_nodes_cache[rel_path].get(qn)
        if node is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        existing_doc = ast.get_docstring(node)
        has_docstring = existing_doc is not None
        target_action = "update" if has_docstring else "insert"

        targets.append(
            {
                "qualname": qn,
                "hop": hop,
                "source": src,
                "rel_path": rel_path,
                "params": _func_param_names(node),
                "has_docstring": has_docstring,
                "target_action": target_action,
            }
        )

    targets = targets[: max(args.limit_targets, 0)]
    history["steps"].append({"step": "select_targets", "num_targets": len(targets)})

    # 4) generate patch edits (in-memory)
    edits: List[PatchEdit] = []
    file_to_old: Dict[str, List[str]] = {}
    file_to_new: Dict[str, str] = {}

    per_target_records = []
    for t in targets:
        rel_path = t["rel_path"]
        qn = t["qualname"]
        if rel_path not in file_to_old:
            base = file_source_cache.get(rel_path)
            if base is None:
                base = _git_show_text(repo_path, commit, rel_path)
            file_to_old[rel_path] = base.splitlines()
            file_to_new[rel_path] = base  # start from base

        base_src = file_to_new[rel_path]
        module_qual = _module_qual_from_relpath(rel_path, ext=".py")
        changed, new_src, msg = _insert_docstring_into_source(base_src, qn, module_qual)
        if changed:
            file_to_new[rel_path] = new_src
        per_target_records.append({"qualname": qn, "rel_path": rel_path, "action": msg, "changed": changed})

    # build unified diffs
    patch_lines: List[str] = []
    for rel_path, old_lines in file_to_old.items():
        new_src = file_to_new[rel_path]
        new_lines = new_src.splitlines()
        if old_lines == new_lines:
            continue
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{_posix(rel_path)}",
            tofile=f"b/{_posix(rel_path)}",
            lineterm="",
        )
        patch_lines.extend(list(diff))
        patch_lines.append("")  # blank line between files
        edits.append(PatchEdit(rel_path=rel_path, old=old_lines, new=new_lines))

    # Patch stats (rough)
    patch_added = 0
    patch_removed = 0
    for ln in patch_lines:
        if ln.startswith(("---", "+++", "@@")):
            continue
        if ln.startswith("+"):
            patch_added += 1
        elif ln.startswith("-"):
            patch_removed += 1

    patch_text = "\n".join(patch_lines).rstrip() + "\n"
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch_text)
    history["steps"].append({"step": "generate_patch", "patch_path": patch_path, "num_files_changed": len(edits)})

    # 5) verifier
    changed_targets = [t for t in targets if any(r["qualname"] == t["qualname"] and r["changed"] for r in per_target_records)]
    report = _verify_docstrings(repo_path, commit, changed_targets, file_to_new)
    report["per_target"] = per_target_records
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    history["steps"].append({"step": "verifier", "report_path": report_path, "ok": report.get("ok")})

    history["finished_at"] = datetime.now().isoformat(timespec="seconds")
    history["duration_ms"] = int((time.monotonic() - t0) * 1000)
    history["counters"] = dict(_COUNTERS)
    history["summary"] = {
        "num_targets": len(targets),
        "num_changed_targets": len(changed_targets),
        "num_files_changed": len(edits),
        "patch_added_lines": patch_added,
        "patch_removed_lines": patch_removed,
        "patch_total_lines": patch_added + patch_removed,
        "verifier_ok": bool(report.get("ok")),
    }
    history["outputs"] = {"patch": patch_path, "verifier_report": report_path, "impacted": impacted_path, "seeds": seeds_path}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("=== agent outputs ===")
    print("patch:", patch_path)
    print("verifier:", report_path)
    print("history:", history_path)
    print("seeds:", seeds_path)
    print("impacted:", impacted_path)
    print("num_targets:", len(targets), "num_files_changed:", len(edits), "verifier_ok:", report.get("ok"))
    print(
        f"[DONE] commit={commit} targets={len(targets)} changed_targets={len(changed_targets)} "
        f"patch_lines={patch_added + patch_removed} verifier_ok={bool(report.get('ok'))} run_dir={run_dir}"
    )


if __name__ == "__main__":
    main()

