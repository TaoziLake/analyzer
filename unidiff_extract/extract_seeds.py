#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set

HUNK_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


@dataclass(frozen=True)
class LineRange:
    start: int
    count: int  # 0 allowed

    @property
    def end_inclusive(self) -> int:
        if self.count <= 0:
            return self.start - 1
        return self.start + self.count - 1


def run_git(repo: str, args: List[str]) -> str:
    out = subprocess.check_output(["git", "-C", repo] + args, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def get_parent_commit(repo: str, commit: str) -> str:
    return run_git(repo, ["rev-parse", f"{commit}^"]).strip()


def get_diff_text(repo: str, commit: str) -> str:
    # unified=0 gives precise line ranges w/out context; --format= removes commit message header
    return run_git(repo, ["show", commit, "--unified=0", "--format="])


def get_file_at_commit(repo: str, commit: str, path: str) -> Optional[str]:
    try:
        return run_git(repo, ["show", f"{commit}:{path}"])
    except subprocess.CalledProcessError:
        return None


def parse_unified0_diff(diff_text: str) -> Dict[str, Dict[str, List[LineRange]]]:
    """
    Returns:
      { path: { "new": [LineRange...], "old": [LineRange...] } }
    """
    result: Dict[str, Dict[str, List[LineRange]]] = {}
    cur_path_new: Optional[str] = None
    cur_path_old: Optional[str] = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            cur_path_new = None
            cur_path_old = None
            continue

        if line.startswith("--- "):
            token = line[4:].strip()
            cur_path_old = None if token == "/dev/null" else token[2:] if token.startswith("a/") else token
            continue

        if line.startswith("+++ "):
            token = line[4:].strip()
            cur_path_new = None if token == "/dev/null" else token[2:] if token.startswith("b/") else token
            key = cur_path_new or cur_path_old
            if key and key not in result:
                result[key] = {"new": [], "old": []}
            continue

        m = HUNK_RE.match(line)
        if m:
            old_start = int(m.group(1))
            old_count = int(m.group(2) or "1")
            new_start = int(m.group(3))
            new_count = int(m.group(4) or "1")

            key = cur_path_new or cur_path_old
            if not key:
                continue
            if key not in result:
                result[key] = {"new": [], "old": []}

            result[key]["old"].append(LineRange(old_start, old_count))
            result[key]["new"].append(LineRange(new_start, new_count))

    return result


@dataclass
class DefSpan:
    kind: str  # function | async_function | class
    qualname: str
    lineno: int
    end_lineno: int


def safe_end_lineno(node: ast.AST) -> Optional[int]:
    end = getattr(node, "end_lineno", None)
    return end if isinstance(end, int) else None


def collect_def_spans(source: str, module_qual: str = "") -> List[DefSpan]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=SyntaxWarning,
            message=r".*invalid escape sequence.*",
        )
        tree = ast.parse(source)
    spans: List[DefSpan] = []
    stack: List[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            stack.append(node.name)
            end = safe_end_lineno(node)
            if end is not None:
                qual = ".".join([p for p in [module_qual, *stack] if p])
                spans.append(DefSpan("class", qual, node.lineno, end))
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            stack.append(node.name)
            end = safe_end_lineno(node)
            if end is not None:
                qual = ".".join([p for p in [module_qual, *stack] if p])
                spans.append(DefSpan("function", qual, node.lineno, end))
            self.generic_visit(node)
            stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            stack.append(node.name)
            end = safe_end_lineno(node)
            if end is not None:
                qual = ".".join([p for p in [module_qual, *stack] if p])
                spans.append(DefSpan("async_function", qual, node.lineno, end))
            self.generic_visit(node)
            stack.pop()

    Visitor().visit(tree)
    return spans


def choose_innermost(spans: List[DefSpan], line_no: int) -> Optional[DefSpan]:
    candidates = [s for s in spans if s.lineno <= line_no <= s.end_lineno]
    if not candidates:
        return None
    candidates.sort(key=lambda s: (s.end_lineno - s.lineno, s.lineno))
    return candidates[0]


def ranges_to_lines(ranges: List[LineRange]) -> Set[int]:
    lines: Set[int] = set()
    for r in ranges:
        if r.count <= 0:
            continue
        for ln in range(r.start, r.start + r.count):
            lines.add(ln)
    return lines


def is_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return ("/tests/" in p) or p.startswith("tests/") or os.path.basename(p).startswith("test_")


def _runs_root() -> str:
    # <locbench>/analyzer/unidiff_extract/extract_seeds.py -> <locbench>/analyzer
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_output_path(repo: str, commit: str, out_arg: Optional[str]) -> str:
    """
    If out_arg is provided, use it as-is (absolute or relative to cwd).
    Otherwise write to:
      <analyzer>/runs/<repo_name>/impacted_seeds/impacted_seeds_<commit>_<timestamp>.json
    """
    if out_arg:
        return os.path.abspath(out_arg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_name = os.path.basename(os.path.abspath(repo).rstrip("\\/")) or "repo"
    out_dir = os.path.join(_runs_root(), "runs", repo_name, "impacted_seeds")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"impacted_seeds_{commit}_{ts}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to git repo (root directory)")
    ap.add_argument("--commit", required=True, help="Commit SHA (short ok)")
    ap.add_argument("--ext", default=".py", help="Target file extension (default .py)")
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path. If omitted, auto-saves under <analyzer>/runs/<repo_name>/impacted_seeds/ with timestamped filename.",
    )
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    commit = args.commit
    out_path = resolve_output_path(repo, commit, args.out)

    parent = get_parent_commit(repo, commit)
    diff_text = get_diff_text(repo, commit)
    file_ranges = parse_unified0_diff(diff_text)

    seeds = []
    for path, ro in file_ranges.items():
        if not path.endswith(args.ext):
            continue

        test_flag = is_test_path(path)
        seed_type = "test_change" if test_flag else "code_change"

        new_src = get_file_at_commit(repo, commit, path)
        old_src = get_file_at_commit(repo, parent, path)

        new_lines = ranges_to_lines(ro["new"])
        old_lines = ranges_to_lines(ro["old"])

        module_qual = path.replace("/", ".").replace("\\", ".")
        if module_qual.endswith(args.ext):
            module_qual = module_qual[: -len(args.ext)]

        if new_src is not None and len(new_lines) > 0:
            spans = collect_def_spans(new_src, module_qual=module_qual)
            for ln in sorted(new_lines):
                hit = choose_innermost(spans, ln)
                if hit:
                    seeds.append(
                        {
                            "path": path,
                            "version": "new",
                            "line": ln,
                            "kind": hit.kind,
                            "qualname": hit.qualname,
                            "span": [hit.lineno, hit.end_lineno],
                            "reason": "diff_new_line_in_def",
                            "is_test": test_flag,
                            "seed_type": seed_type,
                        }
                    )
                else:
                    seeds.append(
                        {
                            "path": path,
                            "version": "new",
                            "line": ln,
                            "kind": "module",
                            "qualname": module_qual,
                            "span": None,
                            "reason": "diff_new_line_outside_defs",
                            "is_test": test_flag,
                            "seed_type": seed_type,
                        }
                    )

        if old_src is not None and (len(old_lines) > 0) and (new_src is None or len(new_lines) == 0):
            spans = collect_def_spans(old_src, module_qual=module_qual)
            for ln in sorted(old_lines):
                hit = choose_innermost(spans, ln)
                if hit:
                    seeds.append(
                        {
                            "path": path,
                            "version": "old",
                            "line": ln,
                            "kind": hit.kind,
                            "qualname": hit.qualname,
                            "span": [hit.lineno, hit.end_lineno],
                            "reason": "diff_old_line_in_def",
                            "is_test": test_flag,
                            "seed_type": seed_type,
                        }
                    )
                else:
                    seeds.append(
                        {
                            "path": path,
                            "version": "old",
                            "line": ln,
                            "kind": "module",
                            "qualname": module_qual,
                            "span": None,
                            "reason": "diff_old_line_outside_defs",
                            "is_test": test_flag,
                            "seed_type": seed_type,
                        }
                    )

    uniq = {}
    for s in seeds:
        key = (s["path"], s["version"], s["qualname"], s["kind"], s["seed_type"])
        if key not in uniq or s["line"] < uniq[key]["line"]:
            uniq[key] = s

    payload = {
        "commit": commit,
        "parent": parent,
        "repo": repo,
        "num_files_in_diff": len(file_ranges),
        "num_py_files_in_diff": sum(1 for p in file_ranges.keys() if p.endswith(args.ext)),
        "num_seeds": len(uniq),
        "seeds": list(uniq.values()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} with {len(uniq)} unique seeds")


if __name__ == "__main__":
    main()

