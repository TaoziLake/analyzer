#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    from tree_sitter_languages import get_parser  # type: ignore[import-not-found]

    _TS_PARSER = get_parser("python")
except Exception:
    _TS_PARSER = None

# Fallback for Python 3.13+ on Windows where `tree_sitter_languages` wheels may be unavailable.
# Uses `tree_sitter` + `tree_sitter_python` to create a Parser.
if _TS_PARSER is None:
    try:
        from tree_sitter import Language, Parser  # type: ignore[import-not-found]
        import tree_sitter_python as _tsp  # type: ignore[import-not-found]

        _TS_PARSER = Parser()
        _TS_PARSER.language = Language(_tsp.language())
    except Exception:
        _TS_PARSER = None


@dataclass
class FunctionCall:
    caller: str  # qualname of calling function
    callee: str  # qualname of called function (best-effort)
    line: int    # line number of the call (1-based)


@dataclass
class CallGraph:
    functions: Dict[str, Set[str]]  # caller -> set(callees)
    reverse_functions: Dict[str, Set[str]]  # callee -> set(callers)
    all_functions: Set[str]  # all discovered qualnames (caller+callee+defs)


def _module_package_parts(module_qual: str) -> List[str]:
    parts = [p for p in module_qual.split(".") if p]
    if not parts:
        return []
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def _resolve_import_from_base(module_qual: str, module: Optional[str], level: int) -> Optional[str]:
    """
    Best-effort resolution for `from ... import ...`:
      - absolute: level == 0  => base = node.module
      - relative: level > 0   => base = <pkg up> + node.module
    """
    if level and level > 0:
        pkg = _module_package_parts(module_qual)
        up = max(level - 1, 0)  # level=1 means current package
        if up:
            pkg = pkg[:-up] if up <= len(pkg) else []
        base = ".".join([p for p in pkg if p])
        if module:
            return ".".join([p for p in [base, module] if p])
        return base or None
    return module or None


def _collect_import_aliases_ast(tree: ast.AST, module_qual: str) -> Dict[str, str]:
    """
    Collect import alias mapping (module-scope best-effort):
      - import x as y           => y -> x
      - import x.y as z         => z -> x.y
      - from a.b import c as d  => d -> a.b.c
      - from .a import b as c   => c -> <resolved>.a.b
    """
    aliases: Dict[str, str] = {}

    class V(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name

        def visit_ImportFrom(self, node: ast.ImportFrom):
            base = _resolve_import_from_base(module_qual, node.module, node.level)
            for a in node.names:
                if a.name == "*":
                    continue
                bound = a.asname or a.name
                if base:
                    aliases[bound] = f"{base}.{a.name}"
                else:
                    aliases[bound] = a.name

    V().visit(tree)
    return aliases


def _apply_alias_to_name(name: str, aliases: Dict[str, str]) -> str:
    if not aliases or not name:
        return name
    if "." not in name:
        return aliases.get(name, name)
    first, rest = name.split(".", 1)
    if first in aliases:
        return f"{aliases[first]}.{rest}"
    return name


def _normalize_self_cls_callee(
    callee: str,
    caller: Optional[str],
    current_class_qual: Optional[str],
) -> str:
    if not callee:
        return callee
    if callee.startswith("self.") or callee.startswith("cls."):
        suffix = callee.split(".", 1)[1]
        if current_class_qual:
            return f"{current_class_qual}.{suffix}"
        # fallback: derive class qual from caller if caller looks like <...>.<Class>.<method>
        if caller and "." in caller:
            class_qual = ".".join(caller.split(".")[:-1])
            if class_qual:
                return f"{class_qual}.{suffix}"
    return callee


def extract_functions_and_calls_ast(source: str, module_qual: str = "") -> Tuple[Set[str], List[FunctionCall]]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=SyntaxWarning,
                message=r".*invalid escape sequence.*",
            )
            tree = ast.parse(source)
    except SyntaxError:
        return set(), []

    defined: Set[str] = set()
    calls: List[FunctionCall] = []

    if module_qual:
        defined.add(module_qual)

    aliases = _collect_import_aliases_ast(tree, module_qual)

    stack: List[str] = []
    current_function: Optional[str] = module_qual or None
    current_class_qual: Optional[str] = None

    class DefVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            stack.append(node.name)
            defined.add(".".join([p for p in [module_qual, *stack] if p]))
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            stack.append(node.name)
            defined.add(".".join([p for p in [module_qual, *stack] if p]))
            self.generic_visit(node)
            stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            stack.append(node.name)
            defined.add(".".join([p for p in [module_qual, *stack] if p]))
            self.generic_visit(node)
            stack.pop()

    def _extract_callee_name(func_node) -> Optional[str]:
        if isinstance(func_node, ast.Name):
            name = func_node.id
            if name in aliases:
                return aliases[name]
            if module_qual and f"{module_qual}.{name}" in defined:
                return f"{module_qual}.{name}"
            return name
        if isinstance(func_node, ast.Attribute):
            parts: List[str] = []
            cur = func_node
            while isinstance(cur, ast.Attribute):
                parts.insert(0, cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.insert(0, cur.id)
                return _apply_alias_to_name(".".join(parts), aliases)
        return None

    class CallVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            nonlocal current_class_qual
            stack.append(node.name)
            old = current_class_qual
            current_class_qual = ".".join([p for p in [module_qual, *stack] if p])
            self.generic_visit(node)
            current_class_qual = old
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            nonlocal current_function
            stack.append(node.name)
            old = current_function
            current_function = ".".join([p for p in [module_qual, *stack] if p])
            self.generic_visit(node)
            current_function = old
            stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            nonlocal current_function
            stack.append(node.name)
            old = current_function
            current_function = ".".join([p for p in [module_qual, *stack] if p])
            self.generic_visit(node)
            current_function = old
            stack.pop()

        def visit_Call(self, node: ast.Call):
            callee = _extract_callee_name(node.func)
            if callee:
                callee = _apply_alias_to_name(callee, aliases)
                callee = _normalize_self_cls_callee(callee, current_function, current_class_qual)
                if current_function:
                    calls.append(FunctionCall(caller=current_function, callee=callee, line=node.lineno))
            self.generic_visit(node)

    DefVisitor().visit(tree)
    CallVisitor().visit(tree)
    return defined, calls


def extract_functions_and_calls_treesitter(source: str, module_qual: str = "") -> Tuple[Set[str], List[FunctionCall]]:
    if _TS_PARSER is None:
        return set(), []

    # tree-sitter parse
    source_bytes = source.encode("utf-8", errors="ignore")
    tree = _TS_PARSER.parse(source_bytes)

    defined: Set[str] = set()
    calls: List[FunctionCall] = []

    if module_qual:
        defined.add(module_qual)

    # alias mapping via ast (safe)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=SyntaxWarning,
                message=r".*invalid escape sequence.*",
            )
            aliases = _collect_import_aliases_ast(ast.parse(source), module_qual)
    except SyntaxError:
        aliases = {}

    stack: List[str] = []
    current_function: Optional[str] = module_qual or None

    def node_text(node) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def add_defined(name: str) -> str:
        stack.append(name)
        qual = ".".join([p for p in [module_qual, *stack] if p])
        defined.add(qual)
        return qual

    def extract_callee(node) -> Optional[str]:
        if node is None:
            return None
        if node.type == "identifier":
            name = node_text(node)
            if name in aliases:
                return aliases[name]
            if module_qual and f"{module_qual}.{name}" in defined:
                return f"{module_qual}.{name}"
            return name
        if node.type == "attribute":
            obj = node.child_by_field_name("object")
            attr = node.child_by_field_name("attribute")
            if obj is None or attr is None:
                return node_text(node).replace(" ", "")
            obj_name = extract_callee(obj)
            attr_name = node_text(attr)
            if obj_name:
                return _apply_alias_to_name(f"{obj_name}.{attr_name}", aliases)
            return attr_name
        return None

    def walk(node, current_class_qual: Optional[str]):
        nonlocal current_function

        if node.type in ("class_definition", "function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = node_text(name_node)
                qualname = add_defined(name)
                if node.type in ("function_definition", "async_function_definition"):
                    old_fn = current_function
                    current_function = qualname
                    for child in node.children:
                        walk(child, current_class_qual)
                    current_function = old_fn
                else:
                    # class_definition
                    new_class_qual = qualname
                    for child in node.children:
                        walk(child, new_class_qual)
                stack.pop()
                return

        if node.type == "call":
            callee_node = node.child_by_field_name("function")
            callee = extract_callee(callee_node)
            if callee:
                callee = _normalize_self_cls_callee(callee, current_function, current_class_qual)
                if current_function:
                    calls.append(
                        FunctionCall(caller=current_function, callee=callee, line=node.start_point[0] + 1)
                    )

        for child in node.children:
            walk(child, current_class_qual)

    walk(tree.root_node, None)
    return defined, calls


def build_call_graph(repo_path: str, ext: str = ".py", parser_mode: str = "auto") -> CallGraph:
    functions: Dict[str, Set[str]] = {}
    reverse_functions: Dict[str, Set[str]] = {}
    all_functions: Set[str] = set()

    use_treesitter = parser_mode in ("auto", "treesitter") and _TS_PARSER is not None
    if parser_mode == "treesitter" and _TS_PARSER is None:
        print("Warning: tree-sitter not available, falling back to ast parser")

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "node_modules", ".git"]]
        for file in files:
            if not file.endswith(ext):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
            except (UnicodeDecodeError, IOError):
                continue

            rel_path = os.path.relpath(file_path, repo_path)
            module_qual = rel_path.replace("/", ".").replace("\\", ".")
            if module_qual.endswith(ext):
                module_qual = module_qual[:-len(ext)]

            if use_treesitter:
                defined, calls = extract_functions_and_calls_treesitter(source, module_qual)
            else:
                defined, calls = extract_functions_and_calls_ast(source, module_qual)

            all_functions.update(defined)
            for c in calls:
                all_functions.add(c.caller)
                all_functions.add(c.callee)
                functions.setdefault(c.caller, set()).add(c.callee)
                reverse_functions.setdefault(c.callee, set()).add(c.caller)

    return CallGraph(functions=functions, reverse_functions=reverse_functions, all_functions=all_functions)


def propagate_impacts(seeds: List[Dict], call_graph: CallGraph, max_hops: int = 2) -> Dict[str, Dict]:
    impacted: Dict[str, Dict] = {}
    visited: Set[str] = set()
    queue: List[Tuple[str, int]] = []

    def add_impacted(qualname: str, hop: int, reason: str, source_qualname: Optional[str]):
        if qualname not in impacted:
            impacted[qualname] = {
                "qualname": qualname,
                "hop": hop,
                "reason": reason,
                "source": source_qualname,
                "callers": list(call_graph.reverse_functions.get(qualname, set())),
                "callees": list(call_graph.functions.get(qualname, set())),
            }
        elif hop < impacted[qualname]["hop"]:
            impacted[qualname]["hop"] = hop
            impacted[qualname]["reason"] = reason
            impacted[qualname]["source"] = source_qualname

    for seed in seeds:
        qn = seed.get("qualname")
        if not qn:
            continue
        if qn in call_graph.all_functions:
            add_impacted(qn, 0, "direct_change", None)
            queue.append((qn, 0))
            visited.add(qn)

    while queue:
        cur, hop = queue.pop(0)
        if hop >= max_hops:
            continue
        nxt = hop + 1

        for caller in call_graph.reverse_functions.get(cur, set()):
            if caller not in visited:
                add_impacted(caller, nxt, "calls_changed_function", cur)
                queue.append((caller, nxt))
                visited.add(caller)

        for callee in call_graph.functions.get(cur, set()):
            if callee not in visited:
                add_impacted(callee, nxt, "called_by_changed_function", cur)
                queue.append((callee, nxt))
                visited.add(callee)

    return impacted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to git repo (root directory)")
    ap.add_argument("--seeds", required=True, help="Path to seeds JSON file from extract_seeds.py")
    ap.add_argument("--ext", default=".py", help="Target file extension (default .py)")
    ap.add_argument(
        "--parser",
        choices=["auto", "ast", "treesitter"],
        default="auto",
        help="Call graph parser: auto|ast|treesitter (default auto)",
    )
    ap.add_argument("--max-hops", type=int, default=2, help="Maximum propagation hops (default 2)")
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    repo_path = os.path.abspath(args.repo)
    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)
    seeds = seeds_data["seeds"]

    print(f"Building call graph for {repo_path}...")
    cg = build_call_graph(repo_path, args.ext, parser_mode=args.parser)
    print(f"Found {len(cg.all_functions)} functions in call graph")
    print(f"Propagating impacts from {len(seeds)} seeds...")

    impacted = propagate_impacts(seeds, cg, args.max_hops)

    output = {
        "commit": seeds_data.get("commit", ""),
        "parent": seeds_data.get("parent", ""),
        "num_seeds": len(seeds),
        "num_impacted": len(impacted),
        "max_hops": args.max_hops,
        "call_graph_stats": {
            "total_functions": len(cg.all_functions),
            "total_call_relations": sum(len(v) for v in cg.functions.values()),
        },
        "impacted": list(impacted.values()),
        "generated_at": seeds_data.get("generated_at", ""),
    }

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        seeds_dir = os.path.dirname(args.seeds)
        seeds_basename = os.path.basename(args.seeds)
        out_basename = seeds_basename.replace("impacted_seeds_", "impacted_set_")
        out_path = os.path.join(seeds_dir, out_basename)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} with {len(impacted)} impacted functions")


if __name__ == "__main__":
    main()

