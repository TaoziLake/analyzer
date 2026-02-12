#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
agent_llm.py

使用 LLM 生成 docstring 的 agent（分層處理版本）。

流程：
  1) extract_seeds.py 生成 seeds（含 old_code / new_code）
  2) process_impacted_set.py 生成 impacted set
  3) 自動檢測 repo 的 docstring 風格
  4) 選 targets（按 hop 分層，internal 且由 code_change seeds 推導）
  5) 分層生成 docstring：
     - hop 0: 將 diff 源碼（舊代碼 + 新代碼）發給 LLM，生成 docstring + change_analysis
     - hop 1+: 將受影響函式全文（非整個檔案）+ 上層分析結果發給 LLM
     失敗時 fallback 到模板
  6) 跑 docstring verifier（AST 級校驗）

輸出：patch + verifier report + history（含 LLM 統計）
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import random
import re
import string
import subprocess
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

# 本地模組
from ..core.style_detector import detect_repo_style_cached
from ..llm.llm_client import LLMClient


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
    # <locbench>/analyzer/agents/llm_agent.py -> <locbench>/analyzer
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


def _extract_function_text(source: str, node: ast.AST) -> str:
    """從源碼中提取函式全文（使用 AST 節點的行號範圍）。"""
    lines = source.splitlines()
    start = node.lineno - 1  # 0-based
    end = getattr(node, "end_lineno", node.lineno)  # 1-based inclusive
    return "\n".join(lines[start:end])


@dataclass
class PatchEdit:
    rel_path: str
    old: List[str]
    new: List[str]


def _select_targets(
    impacted_items: List[Dict],
    code_change_seeds: set,
    max_hops: int,
    repo_path: str,
    commit: str,
    limit_targets: int,
    file_nodes_cache: Dict[str, Dict[str, ast.AST]],
    file_source_cache: Dict[str, str],
) -> List[Dict]:
    """
    從 impacted items 中挑選可生成 docstring 的函式級 targets。

    此函式封裝了 target selection 的核心邏輯，可被主流程和 LLM fallback 共用。

    Args:
        impacted_items: impacted set 中的條目列表
        code_change_seeds: code_change 類型 seed 的 qualname 集合
        max_hops: 最大跳數
        repo_path: git repo 根目錄絕對路徑
        commit: commit SHA
        limit_targets: 最多選擇的 target 數量
        file_nodes_cache: 檔案 AST 節點快取（會被就地修改）
        file_source_cache: 檔案源碼快取（會被就地修改）

    Returns:
        targets 列表，每個 target 是一個 dict
    """
    targets: List[Dict] = []

    # 按 hop 分組 impacted items，逐層檢查 lineage
    impacted_by_hop: Dict[int, List[Dict]] = defaultdict(list)
    for it in impacted_items:
        impacted_by_hop[int(it.get("hop", 999))].append(it)

    # 追蹤已接受的 qualname（具有 code_change lineage）
    accepted_lineage = set(code_change_seeds)

    for hop_level in sorted(impacted_by_hop.keys()):
        if hop_level > max_hops:
            break
        for it in impacted_by_hop[hop_level]:
            if not it.get("is_internal", False):
                continue
            qn = it.get("qualname", "")
            src = it.get("source")

            # Lineage 檢查：hop 0 必須是 code_change seed，hop 1+ 的 source 必須在已接受集合中
            if hop_level == 0:
                if qn not in code_change_seeds:
                    continue
            else:
                if src not in accepted_lineage:
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

            accepted_lineage.add(qn)
            targets.append(
                {
                    "qualname": qn,
                    "hop": hop_level,
                    "source": src,
                    "rel_path": rel_path,
                    "params": _func_param_names(node),
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "has_docstring": has_docstring,
                    "target_action": target_action,
                }
            )

    targets = targets[: max(limit_targets, 0)]
    return targets


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

    # README 文件名：commit_timestamp_randomstring.md
    rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    readme_filename = f"README_{commit}_{_now_tag()}_{rand_suffix}.md"
    readme_path = os.path.join(run_dir, readme_filename)

    # 1) seeds
    print(f"[1/7] Extracting seeds for {commit}...")
    history["steps"].append({"step": "extract_seeds", "out": seeds_path})
    from ..core.seed_extractor import extract_seeds as extract_seeds_func
    extract_seeds_func(repo_path, commit, out_path=seeds_path)

    # 2) impacted
    print(f"[2/7] Processing impacted set...")
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

    # 3) 檢測風格
    print(f"[3/7] Detecting docstring style...")
    if args.style == "auto":
        detected_style = detect_repo_style_cached(repo_path, cache_dir=run_dir)
        print(f"    Detected style: {detected_style}")
    else:
        detected_style = args.style
    history["steps"].append({"step": "detect_style", "style": detected_style})

    # Initialize LLM client early (needed for both step 4.5 fallback and step 5 generation)
    llm_client: Optional[LLMClient] = None
    if not args.no_llm:
        try:
            llm_client = LLMClient()
            print(f"    LLM client initialized (model: {llm_client.model_name})")
        except Exception as e:
            print(f"    Warning: Failed to initialize LLM client: {e}")
            print(f"    Falling back to template mode")

    # 4) pick targets（按 hop 分層，支持 max_hops 層）
    print(f"[4/7] Selecting targets...")
    file_nodes_cache: Dict[str, Dict[str, ast.AST]] = {}
    file_source_cache: Dict[str, str] = {}

    targets = _select_targets(
        impacted_items=impacted_items,
        code_change_seeds=code_change_seeds,
        max_hops=args.max_hops,
        repo_path=repo_path,
        commit=commit,
        limit_targets=args.limit_targets,
        file_nodes_cache=file_nodes_cache,
        file_source_cache=file_source_cache,
    )
    print(f"    Selected {len(targets)} targets")
    hop_counts = defaultdict(int)
    for t in targets:
        hop_counts[t["hop"]] += 1
    print(f"    By hop: {dict(sorted(hop_counts.items()))}")
    history["steps"].append({"step": "select_targets", "num_targets": len(targets), "by_hop": dict(sorted(hop_counts.items()))})

    # 4.5) LLM module-level fallback
    # 當所有 seed 都是模塊級別（kind="module"）且 target selection 沒找到任何
    # 函式級 target 時，讓 LLM 分析 diff 來識別受影響的函式。
    module_seeds = [s for s in seeds_data["seeds"] if s.get("kind") == "module"]
    llm_fallback_used = False

    if len(targets) == 0 and len(module_seeds) > 0 and llm_client is not None:
        print(f"[4.5/7] LLM module-level fallback (0 targets from {len(module_seeds)} module seed(s))...")

        # 1. 取得 diff 文本
        diff_text = _run(["git", "-C", repo_path, "show", commit, "--unified=3", "--format="])

        # 2. 收集每個變動 .py 檔案的源碼和函式列表
        fallback_file_sources: Dict[str, str] = {}
        fallback_file_functions: Dict[str, list] = {}
        for s in module_seeds:
            rel = _posix(s["path"])
            if rel in fallback_file_sources:
                continue
            try:
                src = _git_show_text(repo_path, commit, rel)
            except Exception:
                continue
            fallback_file_sources[rel] = src
            mqual = _module_qual_from_relpath(rel, ext=".py")
            try:
                nodes = _collect_target_nodes(src, mqual)
                fallback_file_functions[rel] = list(nodes.keys())
            except SyntaxError:
                fallback_file_functions[rel] = []

        if fallback_file_functions and any(fallback_file_functions.values()):
            # 3. 呼叫 LLM 分析模塊級變更
            _COUNTERS["llm_calls"] += 1
            llm_analysis = llm_client.analyze_module_level_changes(
                diff_text=diff_text,
                file_sources=fallback_file_sources,
                file_functions=fallback_file_functions,
            )

            if llm_analysis["success"] and llm_analysis["affected_functions"]:
                llm_fallback_used = True
                affected_qns = []
                for af in llm_analysis["affected_functions"]:
                    qn = af["qualname"]
                    affected_qns.append(qn)
                    # 注入到 code_change_seeds
                    code_change_seeds.add(qn)
                    # 注入到 impacted_items（如果尚未存在）
                    if not any(it.get("qualname") == qn for it in impacted_items):
                        impacted_items.append({
                            "qualname": qn,
                            "hop": 0,
                            "reason": "llm_module_analysis",
                            "source": None,
                            "is_internal": True,
                            "is_test": False,
                            "is_external": False,
                        })

                print(f"    LLM identified {len(affected_qns)} affected function(s): {affected_qns}")
                print(f"    ({llm_analysis['latency_ms']}ms)")

                # 4. 用更新後的 seeds + impacted 重新選擇 targets
                targets = _select_targets(
                    impacted_items=impacted_items,
                    code_change_seeds=code_change_seeds,
                    max_hops=args.max_hops,
                    repo_path=repo_path,
                    commit=commit,
                    limit_targets=args.limit_targets,
                    file_nodes_cache=file_nodes_cache,
                    file_source_cache=file_source_cache,
                )
                # 標記 fallback 來源
                for t in targets:
                    t["source_step"] = "llm_module_fallback"

                print(f"    Re-selected {len(targets)} targets after fallback")
                hop_counts = defaultdict(int)
                for t in targets:
                    hop_counts[t["hop"]] += 1
                print(f"    By hop: {dict(sorted(hop_counts.items()))}")

                history["steps"].append({
                    "step": "llm_module_fallback",
                    "num_module_seeds": len(module_seeds),
                    "llm_affected_functions": [af["qualname"] for af in llm_analysis["affected_functions"]],
                    "num_targets_after": len(targets),
                    "by_hop": dict(sorted(hop_counts.items())),
                    "latency_ms": llm_analysis["latency_ms"],
                    "prompt_tokens": llm_analysis["prompt_tokens"],
                    "completion_tokens": llm_analysis["completion_tokens"],
                })
            else:
                reason = llm_analysis.get("error") or "no affected functions identified"
                print(f"    LLM fallback: {reason}")
                history["steps"].append({
                    "step": "llm_module_fallback",
                    "num_module_seeds": len(module_seeds),
                    "llm_affected_functions": [],
                    "num_targets_after": 0,
                    "error": reason,
                })
        else:
            print(f"    No functions found in changed files, skipping LLM fallback")
            history["steps"].append({
                "step": "llm_module_fallback",
                "num_module_seeds": len(module_seeds),
                "skipped": True,
                "reason": "no_functions_in_changed_files",
            })

    # 5) 分層生成 docstring（LLM 或 fallback）
    print(f"[5/7] Generating docstrings (layered)...")

    # 構建 seeds_by_qualname 以便 hop 0 查找 old_code / new_code
    seeds_by_qualname: Dict[str, Dict] = {}
    for s in seeds:
        if s.get("seed_type") == "code_change":
            qn_key = s.get("qualname", "")
            if qn_key and qn_key not in seeds_by_qualname:
                seeds_by_qualname[qn_key] = s

    edits: List[PatchEdit] = []
    file_to_old: Dict[str, List[str]] = {}
    file_to_new: Dict[str, str] = {}

    # 分層分析結果：qualname -> analysis text（傳遞給下一層）
    layer_analyses: Dict[str, str] = {}

    # 按 hop 分組 targets
    targets_by_hop: Dict[int, List[Dict]] = defaultdict(list)
    for t in targets:
        targets_by_hop[t["hop"]].append(t)

    per_target_records = []
    target_idx = 0

    for hop_level in sorted(targets_by_hop.keys()):
        hop_targets = targets_by_hop[hop_level]
        print(f"    --- Hop {hop_level}: {len(hop_targets)} targets ---")

        for t in hop_targets:
            target_idx += 1
            rel_path = t["rel_path"]
            qn = t["qualname"]
            params = t["params"]

            if rel_path not in file_to_old:
                base = file_source_cache.get(rel_path)
                if base is None:
                    base = _git_show_text(repo_path, commit, rel_path)
                file_to_old[rel_path] = base.splitlines()
                file_to_new[rel_path] = base

            base_src = file_to_new[rel_path]
            module_qual = _module_qual_from_relpath(rel_path, ext=".py")

            # 嘗試用 LLM 生成（分層策略）
            llm_result = None
            docstring_lines = None
            used_llm = False

            if llm_client is not None:
                _COUNTERS["llm_calls"] += 1

                if hop_level == 0:
                    # ---- Hop 0: diff-based prompt（舊代碼 + 新代碼）----
                    seed_info = seeds_by_qualname.get(qn, {})
                    old_code = seed_info.get("old_code") or ""
                    new_code = seed_info.get("new_code") or ""

                    # 若 seed 中沒有 code，回退到從 file_source 提取
                    if not new_code:
                        orig_source = file_source_cache.get(rel_path, base_src)
                        node = file_nodes_cache.get(rel_path, {}).get(qn)
                        if node:
                            new_code = _extract_function_text(orig_source, node)

                    llm_result = llm_client.generate_docstring_with_diff(
                        old_code=old_code,
                        new_code=new_code,
                        func_qualname=qn,
                        style=detected_style,
                        commit_hash=commit,
                    )

                    if llm_result["success"] and llm_result["docstring"]:
                        docstring_lines = _format_llm_docstring(llm_result["docstring"])
                        used_llm = True
                        layer_analyses[qn] = llm_result.get("change_analysis", "")
                        print(f"    [{target_idx}/{len(targets)}] {qn} (hop={hop_level}): "
                              f"LLM diff-mode ({llm_result['latency_ms']}ms)")
                    else:
                        _COUNTERS["llm_fallbacks"] += 1
                        print(f"    [{target_idx}/{len(targets)}] {qn} (hop={hop_level}): "
                              f"LLM diff-mode failed, using template"
                              f" | error={llm_result.get('error')} prompt_tokens={llm_result.get('prompt_tokens')} completion_tokens={llm_result.get('completion_tokens')}")

                else:
                    # ---- Hop 1+: impact-based prompt（函式全文 + 上層分析）----
                    orig_source = file_source_cache.get(rel_path, base_src)
                    node = file_nodes_cache.get(rel_path, {}).get(qn)
                    if node:
                        func_source = _extract_function_text(orig_source, node)
                    else:
                        func_source = ""

                    parent_qn = t.get("source", "") or ""
                    parent_analysis = layer_analyses.get(parent_qn, "")
                    relationship = f"this function calls {parent_qn}" if parent_qn else "unknown"

                    llm_result = llm_client.generate_docstring_with_impact(
                        func_source=func_source,
                        func_qualname=qn,
                        parent_qualname=parent_qn,
                        parent_analysis=parent_analysis,
                        relationship=relationship,
                        style=detected_style,
                        commit_hash=commit,
                    )

                    if llm_result["success"] and llm_result["docstring"]:
                        docstring_lines = _format_llm_docstring(llm_result["docstring"])
                        used_llm = True
                        layer_analyses[qn] = llm_result.get("impact_analysis", "")
                        print(f"    [{target_idx}/{len(targets)}] {qn} (hop={hop_level}): "
                              f"LLM impact-mode ({llm_result['latency_ms']}ms)")
                    else:
                        _COUNTERS["llm_fallbacks"] += 1
                        print(f"    [{target_idx}/{len(targets)}] {qn} (hop={hop_level}): "
                              f"LLM impact-mode failed, using template"
                              f" | error={llm_result.get('error')} prompt_tokens={llm_result.get('prompt_tokens')} completion_tokens={llm_result.get('completion_tokens')}")

            # Fallback 到模板
            if docstring_lines is None:
                docstring_lines = _make_docstring_template(params)
                if llm_client is None:
                    print(f"    [{target_idx}/{len(targets)}] {qn} (hop={hop_level}): template (no LLM)")

            changed, new_src, msg = _insert_docstring_into_source(base_src, qn, module_qual, docstring_lines)
            if changed:
                file_to_new[rel_path] = new_src

            record = {
                "qualname": qn,
                "rel_path": rel_path,
                "hop": hop_level,
                "action": msg,
                "changed": changed,
                "used_llm": used_llm,
            }
            if llm_result:
                record["llm_latency_ms"] = llm_result.get("latency_ms", 0)
                record["llm_prompt_tokens"] = llm_result.get("prompt_tokens", 0)
                record["llm_completion_tokens"] = llm_result.get("completion_tokens", 0)
                record["llm_mode"] = "diff" if hop_level == 0 else "impact"
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
    print(f"[6/7] Running verifier...")
    changed_targets = [t for t in targets if any(r["qualname"] == t["qualname"] and r["changed"] for r in per_target_records)]
    report = _verify_docstrings(repo_path, commit, changed_targets, file_to_new)
    report["per_target"] = per_target_records
    report["style"] = detected_style
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    history["steps"].append({"step": "verifier", "report_path": report_path, "ok": report.get("ok")})

    # 7) 生成 README 文件
    readme_generated = False
    if llm_client is not None and len(changed_targets) > 0:
        print(f"[7/7] Generating commit README...")

        # 構建 seeds 摘要
        seeds_summary_parts = []
        for s in seeds:
            if s.get("seed_type") == "code_change":
                qn = s.get("qualname", "")
                path = s.get("path", "")
                kind = s.get("kind", "")
                seeds_summary_parts.append(f"- `{qn}` ({kind}) in `{path}`")
        seeds_summary_text = "\n".join(seeds_summary_parts) if seeds_summary_parts else "(no direct code changes)"

        # 構建 targets 摘要
        targets_summary_parts = []
        for t in targets:
            qn = t["qualname"]
            hop = t["hop"]
            rel = t["rel_path"]
            action = "directly modified" if hop == 0 else f"indirectly affected (hop {hop})"
            targets_summary_parts.append(f"- `{qn}` in `{rel}` — {action}")
        targets_summary_text = "\n".join(targets_summary_parts) if targets_summary_parts else "(no targets)"

        # 構建每個 target 的詳細信息（docstring 更新 + 分析）
        per_target_detail_parts = []
        for rec in per_target_records:
            qn = rec["qualname"]
            hop = rec.get("hop", "?")
            action = rec.get("action", "?")
            used_llm = rec.get("used_llm", False)
            analysis = layer_analyses.get(qn, "")
            detail = f"### `{qn}` (hop {hop})\n"
            detail += f"- Action: {action}\n"
            detail += f"- Used LLM: {used_llm}\n"
            if analysis:
                detail += f"- Analysis: {analysis}\n"
            per_target_detail_parts.append(detail)
        per_target_details_text = "\n".join(per_target_detail_parts) if per_target_detail_parts else "(no details)"

        # 獲取 diff 文本
        try:
            diff_for_readme = _run(["git", "-C", repo_path, "show", commit, "--unified=3", "--format="])
        except Exception:
            diff_for_readme = patch_text

        _COUNTERS["llm_calls"] += 1
        readme_result = llm_client.generate_readme(
            commit_hash=commit,
            repo_name=_repo_name(repo_path),
            seeds_summary=seeds_summary_text,
            targets_summary=targets_summary_text,
            per_target_details=per_target_details_text,
            diff_text=diff_for_readme,
        )

        if readme_result["success"] and readme_result["readme_content"]:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_result["readme_content"])
            readme_generated = True
            print(f"    README generated: {readme_path} ({readme_result['latency_ms']}ms)")
            history["steps"].append({
                "step": "generate_readme",
                "readme_path": readme_path,
                "latency_ms": readme_result["latency_ms"],
                "prompt_tokens": readme_result["prompt_tokens"],
                "completion_tokens": readme_result["completion_tokens"],
            })
        else:
            print(f"    README generation failed: {readme_result.get('error', 'unknown')}")
            history["steps"].append({
                "step": "generate_readme",
                "readme_path": None,
                "error": readme_result.get("error", "unknown"),
            })
    else:
        if len(changed_targets) == 0:
            print(f"[7/7] Skipping README (no changed targets)")
        else:
            print(f"[7/7] Skipping README (no LLM client)")
        history["steps"].append({"step": "generate_readme", "skipped": True})

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
        "llm_module_fallback_used": llm_fallback_used,
        "readme_generated": readme_generated,
        "llm_success_rate": (
            llm_stats.get("success_calls", 0) / llm_stats.get("total_calls", 1)
            if llm_stats.get("total_calls", 0) > 0 else 0.0
        ),
    }
    history["outputs"] = {
        "patch": patch_path,
        "verifier_report": report_path,
        "impacted": impacted_path,
        "seeds": seeds_path,
        "readme": readme_path if readme_generated else None,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n=== agent_llm outputs ===")
    print("patch:", patch_path)
    print("verifier:", report_path)
    if readme_generated:
        print("readme:", readme_path)
    print("history:", history_path)
    print("seeds:", seeds_path)
    print("impacted:", impacted_path)
    print(f"style: {detected_style}")
    print(f"num_targets: {len(targets)}, num_changed: {len(changed_targets)}, verifier_ok: {report.get('ok')}")
    if readme_generated:
        print(f"readme: generated ({readme_filename})")
    if llm_fallback_used:
        print(f"module_fallback: used (injected targets from LLM module-level analysis)")
    if llm_stats:
        print(f"llm_calls: {llm_stats.get('total_calls', 0)}, success: {llm_stats.get('success_calls', 0)}, "
              f"fallback: {_COUNTERS['llm_fallbacks']}")
    print(
        f"[DONE] commit={commit} targets={len(targets)} changed={len(changed_targets)} "
        f"patch_lines={patch_added + patch_removed} verifier_ok={bool(report.get('ok'))} run_dir={run_dir}"
    )


if __name__ == "__main__":
    main()
