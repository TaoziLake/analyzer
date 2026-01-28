#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
agent_llm.py

使用 LLM 生成 docstring 的 agent。

流程：
  1) extract_seeds.py 生成 seeds
  2) process_impacted_set.py 生成 impacted set
  3) 選 targets（hop<=1 且 internal 且由 code_change seeds 推導）
  4) 自動檢測 repo 的 docstring 風格
  5) 使用 LLM 生成 docstring（失敗時 fallback 到模板）
  6) 跑 docstring verifier（AST 級校驗）

輸出：patch + verifier report + history（含 LLM 統計）
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
from typing import Dict, List, Optional, Sequence, Tuple

# 本地模組
from docstring_style import detect_repo_style_cached

# LLM 模組（優先從新路徑導入）
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from llm.llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient  # fallback 舊路徑


_COUNTERS: Dict[str, int] = {
    "subprocess_calls": 0,
    "git_show_calls": 0,
    "ast_parse_calls": 0,
    "llm_calls": 0,
    "llm_fallbacks": 0,
}


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> str:
    _COUNTERS["subprocess_calls"] += 1
    out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def _repo_name(repo_path: str) -> str:
    return os.path.basename(os.path.abspath(repo_path).rstrip("\\/")) or "repo"


def _runs_root() -> str:
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
    qparts = [p for p in qualname.split(".") if p]
    mparts = [p for p in module_qual.split(".") if p]
    k = 0
    while k < min(len(qparts), len(mparts)) and qparts[k] == mparts[k]:
        k += 1
    return qparts[k:]


def _collect_target_nodes(source: str, module_qual: str) -> Dict[str, ast.AST]:
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
    names = [n for n in names if n not in ("self", "cls")]
    return names


def _indent_of_line(line: str) -> str:
    m = re.match(r"^(\s*)", line)
    return m.group(1) if m else ""


def _make_docstring_template(params: List[str]) -> List[str]:
    """Fallback 模板方式生成 docstring"""
    if not params:
        return ['"""TODO: docstring"""']
    lines = ['"""TODO: docstring', "", "Args:"]
    for p in params:
        lines.append(f"    {p}:")
    lines.append('"""')
    return lines


def _format_llm_docstring(docstring: str) -> List[str]:
    """
    將 LLM 生成的 docstring 格式化為多行列表
    """
    lines = ['"""' + docstring.split("\n")[0]]
    rest = docstring.split("\n")[1:]
    if rest:
        lines.extend(rest)
    if not lines[-1].endswith('"""'):
        lines.append('"""')
    else:
        # 確保最後一行是單獨的 """
        if lines[-1] != '"""':
            last = lines[-1]
            if last.endswith('"""'):
                lines[-1] = last[:-3].rstrip()
                if lines[-1]:
                    lines.append('"""')
                else:
                    lines[-1] = '"""'
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
    docstring_lines: List[str],
) -> Tuple[bool, str, Optional[str]]:
    """
    將 docstring 插入或更新到源碼中

    Args:
        source: 原始源碼
        qualname: 函式的完整限定名
        module_qual: 模組限定名
        docstring_lines: docstring 行列表（含三引號）

    Returns:
        (changed, new_source, message)
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
            doc_block = [body_indent + dl if dl else "" for dl in docstring_lines]
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
        idx = max(insert_at - 1, 0)

        body_indent = _indent_of_line(lines[idx]) if idx < len(lines) else "    "
        doc_block = [body_indent + dl if dl else "" for dl in docstring_lines]

        new_lines = lines[:idx] + doc_block + lines[idx:]
        return True, "\n".join(new_lines) + ("\n" if source.endswith("\n") else ""), "inserted"


def _verify_docstrings(
    repo_path: str,
    commit: str,
    targets: List[Dict],
    file_to_new_source: Dict[str, str],
) -> Dict:
    """
    輕量級 verifier：
      - 檔案能解析
      - 每個被修改的 target 有 docstring
      - docstring 包含每個參數名
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
                # 忽略 * 和 ** 前綴
                param_name = p.lstrip("*")
                if param_name and param_name not in doc:
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
    ap = argparse.ArgumentParser(description="使用 LLM 生成 docstring 的 agent")
    ap.add_argument("--repo", required=True, help="待分析 git repo 根目錄")
    ap.add_argument("--commit", required=True, help="commit sha (short ok)")
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--parser", choices=["auto", "ast", "treesitter"], default="ast")
    ap.add_argument("--seed-scope", choices=["all", "code_only"], default="code_only")
    ap.add_argument("--limit-targets", type=int, default=50, help="最多生成 docstring 的 targets 數")
    ap.add_argument("--style", choices=["auto", "google", "numpy", "sphinx"], default="auto",
                    help="docstring 風格（auto 會自動檢測）")
    ap.add_argument("--no-llm", action="store_true", help="禁用 LLM，只用模板（用於測試）")
    ap.add_argument(
        "--run-parent",
        default=None,
        help="指定輸出父目錄；默認寫到 analyzer/runs/<repo>/agent_llm/",
    )
    args = ap.parse_args()

    t0 = time.monotonic()
    repo_path = os.path.abspath(args.repo)
    commit = args.commit
    if args.run_parent:
        run_parent = os.path.abspath(args.run_parent)
    else:
        run_parent = os.path.join(_runs_root(), "runs", _repo_name(repo_path), "agent_llm")
    run_dir = os.path.join(run_parent, f"{commit}_{_now_tag()}")
    os.makedirs(run_dir, exist_ok=True)

    history = {
        "repo": repo_path,
        "commit": commit,
        "run_dir": run_dir,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
        "llm_enabled": not args.no_llm,
    }

    seeds_path = os.path.join(run_dir, f"impacted_seeds_{commit}.json")
    impacted_path = os.path.join(run_dir, f"impacted_set_{commit}.json")
    patch_path = os.path.join(run_dir, f"docstring_patch_{commit}.diff")
    report_path = os.path.join(run_dir, f"verifier_report_{commit}.json")
    history_path = os.path.join(run_dir, "history.json")

    # 1) seeds
    print(f"[1/6] Extracting seeds for {commit}...")
    history["steps"].append({"step": "extract_seeds", "out": seeds_path})
    _run(
        [
            os.environ.get("PYTHON", "python"),
            os.path.join(os.path.dirname(__file__), "extract_seeds.py"),
            "--repo",
            repo_path,
            "--commit",
            commit,
            "--out",
            seeds_path,
        ]
    )

    # 2) impacted
    print(f"[2/6] Processing impacted set...")
    history["steps"].append({"step": "process_impacted_set", "out": impacted_path})
    _run(
        [
            os.environ.get("PYTHON", "python"),
            os.path.join(os.path.dirname(__file__), "process_impacted_set.py"),
            "--repo",
            repo_path,
            "--seeds",
            seeds_path,
            "--max-hops",
            str(args.max_hops),
            "--parser",
            args.parser,
            "--seed-scope",
            args.seed_scope,
            "--out",
            impacted_path,
        ]
    )

    seeds_data = json.load(open(seeds_path, "r", encoding="utf-8"))
    impacted_data = json.load(open(impacted_path, "r", encoding="utf-8"))

    seeds = seeds_data["seeds"]
    code_change_seeds = {s["qualname"] for s in seeds if s.get("seed_type") == "code_change"}

    impacted_items = impacted_data["impacted"]

    # 3) 檢測風格
    print(f"[3/6] Detecting docstring style...")
    if args.style == "auto":
        detected_style = detect_repo_style_cached(repo_path, cache_dir=run_dir)
        print(f"    Detected style: {detected_style}")
    else:
        detected_style = args.style
    history["steps"].append({"step": "detect_style", "style": detected_style})

    # 4) pick targets
    print(f"[4/6] Selecting targets...")
    targets: List[Dict] = []
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
                "lineno": node.lineno,
                "has_docstring": has_docstring,
                "target_action": target_action,
            }
        )

    targets = targets[: max(args.limit_targets, 0)]
    print(f"    Selected {len(targets)} targets")
    history["steps"].append({"step": "select_targets", "num_targets": len(targets)})

    # 5) 生成 docstring（LLM 或 fallback）
    print(f"[5/6] Generating docstrings...")
    llm_client: Optional[LLMClient] = None
    if not args.no_llm:
        try:
            llm_client = LLMClient()
            print(f"    LLM client initialized (model: {llm_client.model_name})")
        except Exception as e:
            print(f"    Warning: Failed to initialize LLM client: {e}")
            print(f"    Falling back to template mode")

    edits: List[PatchEdit] = []
    file_to_old: Dict[str, List[str]] = {}
    file_to_new: Dict[str, str] = {}

    per_target_records = []
    for i, t in enumerate(targets):
        rel_path = t["rel_path"]
        qn = t["qualname"]
        params = t["params"]
        lineno = t["lineno"]

        if rel_path not in file_to_old:
            base = file_source_cache.get(rel_path)
            if base is None:
                base = _git_show_text(repo_path, commit, rel_path)
            file_to_old[rel_path] = base.splitlines()
            file_to_new[rel_path] = base

        base_src = file_to_new[rel_path]
        module_qual = _module_qual_from_relpath(rel_path, ext=".py")

        # 嘗試用 LLM 生成
        llm_result = None
        docstring_lines = None
        used_llm = False

        if llm_client is not None:
            _COUNTERS["llm_calls"] += 1
            file_source = file_source_cache.get(rel_path, base_src)
            llm_result = llm_client.generate_docstring(
                file_source=file_source,
                func_qualname=qn,
                func_lineno=lineno,
                style=detected_style,
            )

            if llm_result["success"] and llm_result["docstring"]:
                docstring_lines = _format_llm_docstring(llm_result["docstring"])
                used_llm = True
                print(f"    [{i+1}/{len(targets)}] {qn}: LLM generated ({llm_result['latency_ms']}ms)")
            else:
                _COUNTERS["llm_fallbacks"] += 1
                print(f"    [{i+1}/{len(targets)}] {qn}: LLM failed, using template")

        # Fallback 到模板
        if docstring_lines is None:
            docstring_lines = _make_docstring_template(params)
            if llm_client is None:
                print(f"    [{i+1}/{len(targets)}] {qn}: template (no LLM)")

        changed, new_src, msg = _insert_docstring_into_source(base_src, qn, module_qual, docstring_lines)
        if changed:
            file_to_new[rel_path] = new_src

        record = {
            "qualname": qn,
            "rel_path": rel_path,
            "action": msg,
            "changed": changed,
            "used_llm": used_llm,
        }
        if llm_result:
            record["llm_latency_ms"] = llm_result.get("latency_ms", 0)
            record["llm_prompt_tokens"] = llm_result.get("prompt_tokens", 0)
            record["llm_completion_tokens"] = llm_result.get("completion_tokens", 0)
            if llm_result.get("error"):
                record["llm_error"] = llm_result["error"]

        per_target_records.append(record)

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
        patch_lines.append("")
        edits.append(PatchEdit(rel_path=rel_path, old=old_lines, new=new_lines))

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

    # 6) verifier
    print(f"[6/6] Running verifier...")
    changed_targets = [t for t in targets if any(r["qualname"] == t["qualname"] and r["changed"] for r in per_target_records)]
    report = _verify_docstrings(repo_path, commit, changed_targets, file_to_new)
    report["per_target"] = per_target_records
    report["style"] = detected_style
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    history["steps"].append({"step": "verifier", "report_path": report_path, "ok": report.get("ok")})

    # LLM 統計
    llm_stats = {}
    if llm_client:
        llm_stats = llm_client.get_stats_dict()

    history["finished_at"] = datetime.now().isoformat(timespec="seconds")
    history["duration_ms"] = int((time.monotonic() - t0) * 1000)
    history["counters"] = dict(_COUNTERS)
    history["llm_stats"] = llm_stats
    history["summary"] = {
        "num_targets": len(targets),
        "num_changed_targets": len(changed_targets),
        "num_files_changed": len(edits),
        "patch_added_lines": patch_added,
        "patch_removed_lines": patch_removed,
        "patch_total_lines": patch_added + patch_removed,
        "verifier_ok": bool(report.get("ok")),
        "style": detected_style,
        "llm_success_rate": (
            llm_stats.get("success_calls", 0) / llm_stats.get("total_calls", 1)
            if llm_stats.get("total_calls", 0) > 0 else 0.0
        ),
    }
    history["outputs"] = {"patch": patch_path, "verifier_report": report_path, "impacted": impacted_path, "seeds": seeds_path}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n=== agent_llm outputs ===")
    print("patch:", patch_path)
    print("verifier:", report_path)
    print("history:", history_path)
    print("seeds:", seeds_path)
    print("impacted:", impacted_path)
    print(f"style: {detected_style}")
    print(f"num_targets: {len(targets)}, num_changed: {len(changed_targets)}, verifier_ok: {report.get('ok')}")
    if llm_stats:
        print(f"llm_calls: {llm_stats.get('total_calls', 0)}, success: {llm_stats.get('success_calls', 0)}, "
              f"fallback: {_COUNTERS['llm_fallbacks']}")
    print(
        f"[DONE] commit={commit} targets={len(targets)} changed={len(changed_targets)} "
        f"patch_lines={patch_added + patch_removed} verifier_ok={bool(report.get('ok'))} run_dir={run_dir}"
    )


if __name__ == "__main__":
    main()
