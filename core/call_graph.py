#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
import warnings
from collections import deque
from dataclasses import dataclass, field
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
    # P1: class inheritance map (class_qualname -> [base_qualnames])
    inheritance: Dict[str, List[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 内置函数 / 标准库过滤（P0: 避免 print, len 等污染调用图）
# ---------------------------------------------------------------------------

_PYTHON_BUILTINS: frozenset = frozenset({
    # built-in functions
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
    "bytes", "callable", "chr", "classmethod", "compile", "complex",
    "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec",
    "filter", "float", "format", "frozenset", "getattr", "globals",
    "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance",
    "issubclass", "iter", "len", "list", "locals", "map", "max",
    "memoryview", "min", "next", "object", "oct", "open", "ord", "pow",
    "print", "property", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
    "tuple", "type", "vars", "zip",
    # common exception constructors
    "Exception", "BaseException",
    "ValueError", "TypeError", "KeyError", "AttributeError",
    "IndexError", "RuntimeError", "NotImplementedError", "StopIteration",
    "OSError", "IOError", "FileNotFoundError", "FileExistsError",
    "ImportError", "ModuleNotFoundError", "PermissionError",
    "AssertionError", "ArithmeticError", "LookupError",
    "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeError",
    "OverflowError", "ZeroDivisionError", "RecursionError",
    "StopAsyncIteration", "GeneratorExit", "SystemExit", "KeyboardInterrupt",
})

_STDLIB_TOP_MODULES: frozenset = frozenset({
    "os", "sys", "re", "json", "math", "datetime", "collections",
    "itertools", "functools", "pathlib", "typing", "typing_extensions",
    "abc", "io", "copy", "glob", "shutil", "tempfile", "logging",
    "warnings", "argparse", "unittest", "pytest", "subprocess",
    "threading", "multiprocessing", "concurrent", "asyncio",
    "hashlib", "hmac", "base64", "secrets",
    "urllib", "http", "email", "string", "textwrap",
    "struct", "enum", "dataclasses", "contextlib",
    "inspect", "traceback", "pprint", "operator", "heapq",
    "bisect", "array", "queue", "socket", "ssl", "signal",
    "time", "calendar", "locale", "codecs", "unicodedata",
    "difflib", "csv", "configparser", "xml", "html",
    "importlib", "pkgutil", "ast", "dis", "token", "tokenize",
    "platform", "ctypes", "weakref", "gc", "atexit",
    "pickle", "shelve", "sqlite3", "zipfile", "tarfile", "gzip", "bz2",
    "lzma", "zlib", "pdb", "cProfile", "profile",
    # common third-party (not repo code)
    "numpy", "np", "pandas", "pd", "scipy", "sklearn",
    "requests", "flask", "django", "fastapi",
    "click", "rich", "tqdm", "yaml", "toml",
})


def _is_external_callee(callee: str) -> bool:
    """判断 callee 是否为 Python 内置函数或标准库调用，不应加入调用图。"""
    if not callee:
        return False
    # 裸名匹配内置函数 / 异常
    if "." not in callee:
        return callee in _PYTHON_BUILTINS
    # 带模块前缀：检查顶层模块名
    top = callee.split(".", 1)[0]
    return top in _STDLIB_TOP_MODULES


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


# ---------------------------------------------------------------------------
# P1: 继承关系解析辅助函数
# ---------------------------------------------------------------------------

def _resolve_base_class(base_name: str, all_functions: Set[str]) -> Optional[str]:
    """
    尝试将基类名称解析为 all_functions 中的完整 qualname。
    优先精确匹配，其次后缀匹配。
    """
    if base_name in all_functions:
        return base_name
    # 后缀匹配：找到以 .BaseName 结尾的 qualname
    candidates = [q for q in all_functions if q.endswith(f".{base_name}") or q == base_name]
    if len(candidates) == 1:
        return candidates[0]
    # 多个候选时，优先选择看起来像类（不含再下一层点号）的
    if candidates:
        # 倾向选择层级最浅的
        candidates.sort(key=lambda c: c.count("."))
        return candidates[0]
    return None


def _inject_inheritance_edges(
    all_functions: Set[str],
    functions: Dict[str, Set[str]],
    reverse_functions: Dict[str, Set[str]],
    inheritance: Dict[str, List[str]],
) -> None:
    """
    P1: 为继承关系注入虚拟边（OVERRIDES / INHERITS）。

    对于每个子类-父类对:
      - 若 Child.method 和 Parent.method 都存在 → 添加 OVERRIDES 双向边
      - 若 Parent.method 存在但 Child 未重写，且有人调用了 Child.method → 添加继承解析边
    这确保父方法变更能传播到子类调用者。
    """
    if not inheritance:
        return

    # 收集所有已知类的 qualname（继承图的 key 和 value 都是类）
    known_classes: Set[str] = set(inheritance.keys())
    for bases in inheritance.values():
        known_classes.update(bases)

    # 解析可能不完整的基类名称
    base_resolution: Dict[str, str] = {}
    for _child, bases in inheritance.items():
        for base in bases:
            if base not in all_functions and base not in base_resolution:
                resolved = _resolve_base_class(base, all_functions)
                if resolved:
                    base_resolution[base] = resolved
                    known_classes.add(resolved)

    # 收集每个类的直接方法名（qualname 以 class_qual + "." 开头且只多一层）
    class_methods: Dict[str, Set[str]] = {}
    for qn in all_functions:
        for cls in known_classes:
            if qn.startswith(cls + "."):
                remainder = qn[len(cls) + 1:]
                # 只取直接方法，跳过嵌套类/函数
                if "." not in remainder:
                    class_methods.setdefault(cls, set()).add(remainder)

    for child_class, bases in inheritance.items():
        child_meths = class_methods.get(child_class, set())
        for base in bases:
            resolved_base = base_resolution.get(base, base)
            if resolved_base not in all_functions:
                continue
            base_meths = class_methods.get(resolved_base, set())

            # 重写方法: 子类和父类都定义了同名方法
            for method in child_meths & base_meths:
                child_m = f"{child_class}.{method}"
                parent_m = f"{resolved_base}.{method}"
                # 添加边: Child.method → Parent.method (OVERRIDES)
                functions.setdefault(child_m, set()).add(parent_m)
                reverse_functions.setdefault(parent_m, set()).add(child_m)

            # 继承方法: 仅父类定义，但可能通过 child.method 被调用
            for method in base_meths - child_meths:
                child_m = f"{child_class}.{method}"
                parent_m = f"{resolved_base}.{method}"
                # 只有当有人引用了 child.method 时才注入边
                if child_m in reverse_functions or child_m in functions:
                    functions.setdefault(child_m, set()).add(parent_m)
                    reverse_functions.setdefault(parent_m, set()).add(child_m)
                    all_functions.add(child_m)


# ---------------------------------------------------------------------------
# P1: Seed qualname 后缀匹配辅助函数
# ---------------------------------------------------------------------------

def _build_suffix_index(all_functions: Set[str]) -> Dict[str, Set[str]]:
    """
    构建后缀索引: 最后 N 段 dot 分隔的后缀 → 匹配的完整 qualname 集合。
    用于 seed 模糊匹配。
    """
    index: Dict[str, Set[str]] = {}
    for qn in all_functions:
        parts = qn.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            index.setdefault(suffix, set()).add(qn)
    return index


def _fuzzy_match_seed(
    qn: str,
    all_functions: Set[str],
    suffix_index: Dict[str, Set[str]],
) -> Tuple[Optional[str], str]:
    """
    P1: 模糊匹配 seed qualname 到调用图。

    匹配策略（按优先级）:
      1. 精确匹配
      2. seed 是某个 qualname 的后缀 → 唯一则命中
      3. 某个 qualname 是 seed 的后缀 → 唯一则命中

    Returns:
        (matched_qualname | None, match_type)
        match_type: "exact" | "suffix" | "prefix_strip" | "none"
    """
    # 1. 精确匹配
    if qn in all_functions:
        return qn, "exact"

    # 2. seed 本身作为后缀在索引中查找
    candidates = suffix_index.get(qn, set())
    if len(candidates) == 1:
        return next(iter(candidates)), "suffix"

    # 3. 逐步截短 seed 的前缀，尝试后缀匹配
    parts = qn.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        cands = suffix_index.get(suffix, set())
        if len(cands) == 1:
            return next(iter(cands)), "prefix_strip"

    # 4. seed 作为前缀：可能 call graph 比 seed 更长（嵌套）
    prefix_matches = [f for f in all_functions if f.startswith(qn + ".")]
    if len(prefix_matches) == 1:
        return prefix_matches[0], "prefix_strip"

    return None, "none"


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


def extract_functions_and_calls_treesitter(
    source: str, module_qual: str = "",
) -> Tuple[Set[str], List[FunctionCall], Dict[str, List[str]]]:
    """
    使用 tree-sitter 提取函数定义、调用关系和类继承。

    改进点:
      - P1: 提取 class 基类列表，返回 class_inheritance
      - P1: 处理 super().method() 调用，解析到父类方法
      - P2: 基于构造函数赋值和参数类型注解的轻量类型推断

    Returns:
        (defined_functions, calls, class_inheritance)
        class_inheritance: {class_qualname: [base_class_qualnames]}
    """
    if _TS_PARSER is None:
        return set(), [], {}

    # tree-sitter parse
    source_bytes = source.encode("utf-8", errors="ignore")
    tree = _TS_PARSER.parse(source_bytes)

    defined: Set[str] = set()
    calls: List[FunctionCall] = []
    class_inheritance: Dict[str, List[str]] = {}  # P1: 继承关系

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

    # P2: 每个函数作用域的局部类型推断栈
    local_types_stack: List[Dict[str, str]] = [{}]

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
        # P1: 处理 super() 调用 → 返回占位符以便后续解析到父类
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node is not None and func_node.type == "identifier":
                if node_text(func_node) == "super":
                    return "__super__"
        return None

    # ------------------------------------------------------------------
    # P2: 类型推断辅助函数
    # ------------------------------------------------------------------

    def _extract_type_name(type_node) -> Optional[str]:
        """从类型注解节点提取类型名称。"""
        if type_node is None:
            return None
        # tree-sitter 的 type 节点包裹实际类型表达式
        if type_node.type == "type":
            for child in type_node.children:
                if child.type in ("identifier", "attribute"):
                    return extract_callee(child)
            return None
        if type_node.type in ("identifier", "attribute"):
            return extract_callee(type_node)
        return None

    def _extract_param_types(params_node) -> Dict[str, str]:
        """从函数参数列表中提取类型注解。"""
        types: Dict[str, str] = {}
        if params_node is None:
            return types
        for child in params_node.children:
            if child.type in ("typed_parameter", "typed_default_parameter"):
                name_node = child.child_by_field_name("name")
                type_node = child.child_by_field_name("type")
                if name_node and type_node:
                    pname = node_text(name_node)
                    ptype = _extract_type_name(type_node)
                    # 跳过 self / cls，它们已有专门的解析逻辑
                    if pname and ptype and pname not in ("self", "cls"):
                        ptype = _apply_alias_to_name(ptype, aliases)
                        if not _is_external_callee(ptype):
                            types[pname] = ptype
        return types

    def _try_extract_assignment_type(node) -> Optional[Tuple[str, str]]:
        """
        从赋值语句中推断变量类型。

        处理模式:
          - x = Foo()           → ("x", "Foo")  (构造函数调用，类名首字母大写)
          - x: Foo = ...        → ("x", "Foo")  (类型注解)
          - x = module.Bar()    → ("x", "module.Bar")
        """
        if node.type != "assignment":
            return None
        left = node.child_by_field_name("left")
        if left is None or left.type != "identifier":
            return None
        var_name = node_text(left)

        # 检查类型注解: x: Foo = ...
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            type_name = _extract_type_name(type_node)
            if type_name:
                type_name = _apply_alias_to_name(type_name, aliases)
                if not _is_external_callee(type_name):
                    return (var_name, type_name)

        # 检查构造函数调用: x = Foo()
        right = node.child_by_field_name("right")
        if right is not None and right.type == "call":
            func_node = right.child_by_field_name("function")
            if func_node is not None:
                callee_name = extract_callee(func_node)
                if callee_name and not _is_external_callee(callee_name):
                    # 启发式: 类名首字母大写
                    bare = callee_name.rsplit(".", 1)[-1]
                    if bare and bare[0].isupper():
                        callee_name = _apply_alias_to_name(callee_name, aliases)
                        return (var_name, callee_name)
        return None

    def _resolve_callee_with_types(
        callee: str,
        local_types: Dict[str, str],
    ) -> str:
        """
        P2: 利用局部类型信息解析 callee。
        例如: callee = "obj.method", local_types["obj"] = "pkg.Foo"
              → 解析为 "pkg.Foo.method"
        """
        if not callee or "." not in callee:
            return callee
        obj_name, rest = callee.split(".", 1)
        # self/cls 已有专门逻辑，跳过
        if obj_name in ("self", "cls"):
            return callee
        if obj_name in local_types:
            return f"{local_types[obj_name]}.{rest}"
        return callee

    # ------------------------------------------------------------------
    # 主遍历
    # ------------------------------------------------------------------

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

                    # P2: 推入新的局部类型作用域，附带参数类型注解
                    params_node = node.child_by_field_name("parameters")
                    param_types = _extract_param_types(params_node)
                    local_types_stack.append(param_types)

                    for child in node.children:
                        walk(child, current_class_qual)

                    local_types_stack.pop()  # P2: 弹出作用域
                    current_function = old_fn
                else:
                    # class_definition
                    new_class_qual = qualname

                    # P1: 提取基类列表
                    superclasses_node = node.child_by_field_name("superclasses")
                    if superclasses_node is not None:
                        bases: List[str] = []
                        for child in superclasses_node.children:
                            if child.type in ("identifier", "attribute"):
                                base_name = extract_callee(child)
                                if base_name:
                                    base_name = _apply_alias_to_name(base_name, aliases)
                                    if not _is_external_callee(base_name):
                                        bases.append(base_name)
                        if bases:
                            class_inheritance[qualname] = bases

                    for child in node.children:
                        walk(child, new_class_qual)
                stack.pop()
                return

        # P2: 跟踪赋值语句中的类型信息
        if node.type == "assignment":
            result = _try_extract_assignment_type(node)
            if result:
                var_name, type_name = result
                local_types_stack[-1][var_name] = type_name

        if node.type == "call":
            callee_node = node.child_by_field_name("function")
            callee = extract_callee(callee_node)
            if callee:
                # P1: 解析 super() 调用到父类方法
                if "__super__" in callee:
                    if current_class_qual and current_class_qual in class_inheritance:
                        bases = class_inheritance[current_class_qual]
                        if bases:
                            callee = callee.replace("__super__", bases[0])
                        else:
                            callee = callee.replace("__super__.", "")
                    elif current_class_qual:
                        callee = callee.replace("__super__.", "")
                    else:
                        callee = callee.replace("__super__.", "")

                # P2: 利用局部类型推断解析 obj.method()
                callee = _resolve_callee_with_types(callee, local_types_stack[-1])

                callee = _normalize_self_cls_callee(callee, current_function, current_class_qual)
                if current_function:
                    calls.append(
                        FunctionCall(caller=current_function, callee=callee, line=node.start_point[0] + 1)
                    )

        for child in node.children:
            walk(child, current_class_qual)

    walk(tree.root_node, None)
    return defined, calls, class_inheritance


def build_call_graph(repo_path: str, ext: str = ".py", parser_mode: str = "auto") -> CallGraph:
    functions: Dict[str, Set[str]] = {}
    reverse_functions: Dict[str, Set[str]] = {}
    all_functions: Set[str] = set()
    all_inheritance: Dict[str, List[str]] = {}  # P1: 聚合所有文件的继承关系

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
                defined, calls, file_inheritance = extract_functions_and_calls_treesitter(source, module_qual)
                all_inheritance.update(file_inheritance)
            else:
                defined, calls = extract_functions_and_calls_ast(source, module_qual)

            all_functions.update(defined)
            for c in calls:
                # P0: 跳过内置函数 / 标准库调用，避免污染调用图
                if _is_external_callee(c.callee):
                    continue
                all_functions.add(c.caller)
                all_functions.add(c.callee)
                functions.setdefault(c.caller, set()).add(c.callee)
                reverse_functions.setdefault(c.callee, set()).add(c.caller)

    # P1: 后处理 — 根据继承关系注入虚拟边
    _inject_inheritance_edges(all_functions, functions, reverse_functions, all_inheritance)

    return CallGraph(
        functions=functions,
        reverse_functions=reverse_functions,
        all_functions=all_functions,
        inheritance=all_inheritance,
    )


def propagate_impacts(seeds: List[Dict], call_graph: CallGraph, max_hops: int = 2) -> Dict[str, Dict]:
    impacted: Dict[str, Dict] = {}
    visited: Set[str] = set()
    queue: deque = deque()

    # P1: 构建后缀索引用于模糊匹配
    suffix_index = _build_suffix_index(call_graph.all_functions)
    unmatched_seeds: List[str] = []

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
        # P1: 模糊匹配 seed qualname
        matched, match_type = _fuzzy_match_seed(qn, call_graph.all_functions, suffix_index)
        if matched:
            if match_type != "exact":
                print(f"  [seed-match] fuzzy ({match_type}): '{qn}' -> '{matched}'")
            add_impacted(matched, 0, "direct_change", None)
            queue.append((matched, 0))
            visited.add(matched)
        else:
            unmatched_seeds.append(qn)

    if unmatched_seeds:
        print(f"  [seed-match] WARNING: {len(unmatched_seeds)} seed(s) unmatched in call graph:")
        for s in unmatched_seeds[:10]:
            print(f"    - {s}")
        if len(unmatched_seeds) > 10:
            print(f"    ... and {len(unmatched_seeds) - 10} more")

    while queue:
        cur, hop = queue.popleft()
        if hop >= max_hops:
            continue
        nxt = hop + 1

        for caller in call_graph.reverse_functions.get(cur, set()):
            if caller not in visited:
                add_impacted(caller, nxt, "calls_changed_function", cur)
                queue.append((caller, nxt))
                visited.add(caller)

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
    if cg.inheritance:
        print(f"  Inheritance relations: {len(cg.inheritance)} classes with bases")
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
            "inheritance_classes": len(cg.inheritance),
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
