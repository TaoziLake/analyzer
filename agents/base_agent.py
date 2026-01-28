#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base agent class with shared functionality for docstring generation agents.
"""

from __future__ import annotations

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

from ..core import extract_seeds, build_call_graph, propagate_impacts, propagate_impacts_typed, PropagationConfig
from ..core.impact_propagator import _classify_qualname


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
    # <locbench>/analyzer/agents/base_agent.py -> <locbench>/analyzer
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


def _module_qual_from_relpath(rel_file_path: str, ext: str = ".py") -> str:
    rel = _posix(rel_file_path)
    if rel.endswith(ext):
        rel = rel[: -len(ext)]
    module_qual = rel.replace("/", ".")
    if module_qual.endswith(".__init__"):
        module_qual = module_qual[: -len(".__init__")]
    return module_qual


def _collect_target_nodes(source: str, module_qual: str) -> Dict[str, ast.AST]:
    """
    Build mapping qualname -> node (ClassDef/FunctionDef/AsyncFunctionDef).
    """
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


@dataclass
class PatchEdit:
    rel_path: str
    old: List[str]
    new: List[str]


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


class BaseAgent:
    """Base class for docstring generation agents."""
    
    def __init__(self, repo_path: str, commit: str, run_parent: Optional[str] = None):
        self.repo_path = os.path.abspath(repo_path)
        self.commit = commit
        self.run_parent = run_parent or os.path.join(_runs_root(), "runs", _repo_name(repo_path), "agent")
        self.run_dir = os.path.join(self.run_parent, f"{commit}_{_now_tag()}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.history = {
            "repo": self.repo_path,
            "commit": commit,
            "run_dir": self.run_dir,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "steps": [],
        }
        
        self.seeds_path = os.path.join(self.run_dir, f"impacted_seeds_{commit}.json")
        self.impacted_path = os.path.join(self.run_dir, f"impacted_set_{commit}.json")
        self.patch_path = os.path.join(self.run_dir, f"docstring_patch_{commit}.diff")
        self.report_path = os.path.join(self.run_dir, f"verifier_report_{commit}.json")
        self.history_path = os.path.join(self.run_dir, "history.json")
    
    def extract_seeds_step(self):
        """Step 1: Extract seeds from git diff."""
        self.history["steps"].append({"step": "extract_seeds", "out": self.seeds_path})
        extract_seeds(self.repo_path, self.commit, out_path=self.seeds_path)
    
    def build_impacted_set_step(self, max_hops: int = 2, parser: str = "ast", seed_scope: str = "code_only", use_typed: bool = False):
        """Step 2: Build call graph and propagate impacts."""
        self.history["steps"].append({"step": "process_impacted_set", "out": self.impacted_path})
        
        # Import here to avoid circular imports
        import subprocess
        import sys
        
        python_cmd = os.environ.get("PYTHON", sys.executable)
        
        if use_typed:
            # Use typed propagation
            from ..core import build_call_graph, propagate_impacts_typed, PropagationConfig
            import json
            
            with open(self.seeds_path, "r", encoding="utf-8") as f:
                seeds_data = json.load(f)
            
            seeds = seeds_data["seeds"]
            if seed_scope == "code_only":
                seeds = [s for s in seeds if s.get("seed_type") == "code_change"]
            
            call_graph = build_call_graph(self.repo_path, parser_mode=parser)
            config = PropagationConfig(max_hops=max_hops)
            impacted_nodes = propagate_impacts_typed(seeds, call_graph, config)
            
            # Convert to dict format
            impacted_dict = {}
            for qn, node in impacted_nodes.items():
                is_internal, is_test, is_external = _classify_qualname(self.repo_path, qn)
                impacted_dict[qn] = {
                    "qualname": qn,
                    "hop": node.hop,
                    "reason": node.reason,
                    "source": node.source,
                    "callers": node.callers,
                    "callees": node.callees,
                    "is_internal": is_internal,
                    "is_test": is_test,
                    "is_external": is_external,
                    "score": node.score,
                }
            
            output = {
                "commit": seeds_data.get("commit", ""),
                "parent": seeds_data.get("parent", ""),
                "repo": self.repo_path,
                "num_seeds": len(seeds_data["seeds"]),
                "num_seeds_used": len(seeds),
                "seed_scope": seed_scope,
                "num_impacted": len(impacted_dict),
                "max_hops": max_hops,
                "impacted": list(impacted_dict.values()),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            
            with open(self.impacted_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        else:
            # Use simple propagation via subprocess
            from ..core import process_impacted_set
            # This will be handled by the old script for now
            _run([
                python_cmd,
                os.path.join(os.path.dirname(__file__), "..", "unidiff_extract", "process_impacted_set.py"),
                "--repo", self.repo_path,
                "--seeds", self.seeds_path,
                "--max-hops", str(max_hops),
                "--parser", parser,
                "--seed-scope", seed_scope,
                "--out", self.impacted_path,
            ])
    
    def select_targets(self, limit_targets: int = 50) -> List[Dict]:
        """Step 3: Select targets from impacted set."""
        import json
        
        with open(self.seeds_path, "r", encoding="utf-8") as f:
            seeds_data = json.load(f)
        with open(self.impacted_path, "r", encoding="utf-8") as f:
            impacted_data = json.load(f)
        
        seeds = seeds_data["seeds"]
        code_change_seeds = {s["qualname"] for s in seeds if s.get("seed_type") == "code_change"}
        impacted_items = impacted_data["impacted"]
        
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
            
            resolved = _resolve_qualname_to_file(self.repo_path, qn, ext=".py")
            if not resolved:
                continue
            _, rel_path = resolved
            if _is_test_relpath(rel_path):
                continue
            rel_path = _posix(rel_path)
            
            if rel_path not in file_nodes_cache:
                base = _git_show_text(self.repo_path, self.commit, rel_path)
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
            
            targets.append({
                "qualname": qn,
                "hop": hop,
                "source": src,
                "rel_path": rel_path,
                "params": _func_param_names(node),
                "lineno": node.lineno,
                "has_docstring": has_docstring,
                "target_action": target_action,
            })
        
        targets = targets[:max(limit_targets, 0)]
        self.history["steps"].append({"step": "select_targets", "num_targets": len(targets)})
        return targets, file_source_cache
    
    def generate_patch(self, targets: List[Dict], file_source_cache: Dict[str, str], docstring_lines_fn) -> Tuple[List[PatchEdit], Dict[str, str]]:
        """Step 4: Generate patch edits."""
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
                    base = _git_show_text(self.repo_path, self.commit, rel_path)
                file_to_old[rel_path] = base.splitlines()
                file_to_new[rel_path] = base
            
            base_src = file_to_new[rel_path]
            module_qual = _module_qual_from_relpath(rel_path, ext=".py")
            
            # Get docstring lines from subclass
            docstring_lines = docstring_lines_fn(t, base_src, module_qual)
            
            changed, new_src, msg = self._insert_docstring_into_source(base_src, qn, module_qual, docstring_lines)
            if changed:
                file_to_new[rel_path] = new_src
            per_target_records.append({"qualname": qn, "rel_path": rel_path, "action": msg, "changed": changed})
        
        # Build unified diffs
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
        
        patch_text = "\n".join(patch_lines).rstrip() + "\n"
        with open(self.patch_path, "w", encoding="utf-8") as f:
            f.write(patch_text)
        
        self.history["steps"].append({"step": "generate_patch", "patch_path": self.patch_path, "num_files_changed": len(edits)})
        return edits, file_to_new, per_target_records
    
    def _insert_docstring_into_source(
        self,
        source: str,
        qualname: str,
        module_qual: str,
        docstring_lines: List[str],
    ) -> Tuple[bool, str, Optional[str]]:
        """Insert or update docstring in source."""
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
            # Update existing docstring
            first_stmt = node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, (ast.Str, ast.Constant)):
                doc_start = first_stmt.lineno - 1
                doc_end = getattr(first_stmt, "end_lineno", first_stmt.lineno)
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
    
    def verify_step(self, targets: List[Dict], file_to_new: Dict[str, str], per_target_records: List[Dict]):
        """Step 5: Verify docstrings."""
        changed_targets = [t for t in targets if any(r["qualname"] == t["qualname"] and r["changed"] for r in per_target_records)]
        report = _verify_docstrings(self.repo_path, self.commit, changed_targets, file_to_new)
        report["per_target"] = per_target_records
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.history["steps"].append({"step": "verifier", "report_path": self.report_path, "ok": report.get("ok")})
        return report
    
    def save_history(self, summary: Dict):
        """Save execution history."""
        self.history["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.history["duration_ms"] = int((time.monotonic() - self.history.get("_start_time", time.monotonic())) * 1000)
        self.history["counters"] = dict(_COUNTERS)
        self.history["summary"] = summary
        self.history["outputs"] = {
            "patch": self.patch_path,
            "verifier_report": self.report_path,
            "impacted": self.impacted_path,
            "seeds": self.seeds_path,
        }
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
