#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Docstring style detection module.
"""

from __future__ import annotations

import ast
import os
import random
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class StyleVotes:
    """風格投票統計"""
    google: int = 0
    numpy: int = 0
    sphinx: int = 0
    unknown: int = 0

    def total(self) -> int:
        return self.google + self.numpy + self.sphinx + self.unknown

    def winner(self) -> str:
        """返回票數最多的風格，平手時優先 Google"""
        scores = [
            (self.google, "google"),
            (self.numpy, "numpy"),
            (self.sphinx, "sphinx"),
        ]
        scores.sort(key=lambda x: (-x[0], x[1]))  # 降序，同分時字母序
        if scores[0][0] == 0:
            return "google"  # 默認
        return scores[0][1]

    def confidence(self) -> float:
        """返回最高票佔比"""
        total = self.google + self.numpy + self.sphinx
        if total == 0:
            return 0.0
        winner_votes = max(self.google, self.numpy, self.sphinx)
        return winner_votes / total


# 風格特徵正則表達式
_GOOGLE_PATTERNS = [
    re.compile(r"^\s*Args:\s*$", re.MULTILINE),
    re.compile(r"^\s*Returns:\s*$", re.MULTILINE),
    re.compile(r"^\s*Raises:\s*$", re.MULTILINE),
    re.compile(r"^\s*Yields:\s*$", re.MULTILINE),
    re.compile(r"^\s*Examples?:\s*$", re.MULTILINE),
    re.compile(r"^\s*Attributes:\s*$", re.MULTILINE),
]

_NUMPY_PATTERNS = [
    re.compile(r"^\s*Parameters\s*$", re.MULTILINE),
    re.compile(r"^\s*-{3,}\s*$", re.MULTILINE),  # NumPy 用 --- 分隔
    re.compile(r"^\s*Returns\s*$", re.MULTILINE),
    re.compile(r"^\s*Raises\s*$", re.MULTILINE),
    re.compile(r"^\s*See Also\s*$", re.MULTILINE),
    re.compile(r"^\s*Notes\s*$", re.MULTILINE),
]

_SPHINX_PATTERNS = [
    re.compile(r":param\s+\w+:", re.MULTILINE),
    re.compile(r":type\s+\w+:", re.MULTILINE),
    re.compile(r":returns?:", re.MULTILINE),
    re.compile(r":rtype:", re.MULTILINE),
    re.compile(r":raises?\s+\w+:", re.MULTILINE),
]


def detect_docstring_style(docstring: str) -> str:
    """
    檢測單個 docstring 的風格

    Returns:
        "google" | "numpy" | "sphinx" | "unknown"
    """
    if not docstring:
        return "unknown"

    google_score = sum(1 for p in _GOOGLE_PATTERNS if p.search(docstring))
    numpy_score = sum(1 for p in _NUMPY_PATTERNS if p.search(docstring))
    sphinx_score = sum(1 for p in _SPHINX_PATTERNS if p.search(docstring))

    # NumPy 需要 underline（---）才算
    if numpy_score > 0 and not re.search(r"^\s*-{3,}\s*$", docstring, re.MULTILINE):
        numpy_score = max(0, numpy_score - 2)  # 降低權重

    if google_score == 0 and numpy_score == 0 and sphinx_score == 0:
        return "unknown"

    scores = [
        (google_score, "google"),
        (numpy_score, "numpy"),
        (sphinx_score, "sphinx"),
    ]
    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores[0][1]


def extract_docstrings_from_source(source: str) -> List[str]:
    """
    從 Python 源碼中提取所有 docstring
    """
    docstrings: List[str] = []

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=SyntaxWarning,
                message=r".*invalid escape sequence.*",
            )
            tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 20:  # 忽略太短的 docstring
                docstrings.append(doc)

    return docstrings


def scan_repo_for_docstrings(
    repo_path: str,
    max_files: int = 50,
    ext: str = ".py",
    exclude_tests: bool = True,
    seed: Optional[int] = None,
) -> List[str]:
    """
    掃描 repo 收集 docstring 樣本

    Args:
        repo_path: 倉庫根目錄
        max_files: 最多掃描的檔案數
        ext: 檔案擴展名
        exclude_tests: 是否排除測試檔案
        seed: 隨機種子（用於可重現的抽樣）

    Returns:
        docstring 列表
    """
    py_files: List[str] = []

    for root, dirs, files in os.walk(repo_path):
        # 排除隱藏目錄和常見非源碼目錄
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in ["__pycache__", "node_modules", ".git", "venv", ".venv", "env", "build", "dist"]
        ]

        for file in files:
            if not file.endswith(ext):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path).replace("\\", "/").lower()

            # 排除測試檔案
            if exclude_tests:
                if "/tests/" in rel_path or rel_path.startswith("tests/") or file.startswith("test_"):
                    continue

            py_files.append(file_path)

    # 隨機抽樣
    if seed is not None:
        random.seed(seed)

    if len(py_files) > max_files:
        py_files = random.sample(py_files, max_files)

    # 提取 docstring
    all_docstrings: List[str] = []
    for file_path in py_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            docstrings = extract_docstrings_from_source(source)
            all_docstrings.extend(docstrings)
        except (IOError, UnicodeDecodeError):
            continue

    return all_docstrings


def detect_repo_style(
    repo_path: str,
    max_files: int = 50,
    min_samples: int = 5,
    seed: Optional[int] = 42,
) -> Tuple[str, StyleVotes, float]:
    """
    檢測整個 repo 的主要 docstring 風格

    Args:
        repo_path: 倉庫根目錄
        max_files: 最多掃描的檔案數
        min_samples: 最少需要的 docstring 樣本數（不足時返回 google）
        seed: 隨機種子

    Returns:
        (style, votes, confidence)
        - style: "google" | "numpy" | "sphinx"
        - votes: StyleVotes 統計
        - confidence: 置信度 0.0-1.0
    """
    docstrings = scan_repo_for_docstrings(
        repo_path,
        max_files=max_files,
        seed=seed,
    )

    votes = StyleVotes()

    for doc in docstrings:
        style = detect_docstring_style(doc)
        if style == "google":
            votes.google += 1
        elif style == "numpy":
            votes.numpy += 1
        elif style == "sphinx":
            votes.sphinx += 1
        else:
            votes.unknown += 1

    # 樣本不足時默認 Google
    effective_total = votes.google + votes.numpy + votes.sphinx
    if effective_total < min_samples:
        return "google", votes, 0.0

    return votes.winner(), votes, votes.confidence()


def detect_repo_style_cached(
    repo_path: str,
    cache_dir: Optional[str] = None,
    max_files: int = 50,
) -> str:
    """
    帶緩存的風格檢測（結果寫入 cache_dir）

    Args:
        repo_path: 倉庫根目錄
        cache_dir: 緩存目錄（如果提供，會讀取/寫入緩存）
        max_files: 最多掃描的檔案數

    Returns:
        風格名稱
    """
    import json

    # 嘗試讀取緩存
    if cache_dir:
        cache_file = os.path.join(cache_dir, "docstring_style_cache.json")
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("repo_path") == os.path.abspath(repo_path):
                    return cached.get("style", "google")
            except (IOError, json.JSONDecodeError):
                pass

    # 執行檢測
    style, votes, confidence = detect_repo_style(repo_path, max_files=max_files)

    # 寫入緩存
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_data = {
            "repo_path": os.path.abspath(repo_path),
            "style": style,
            "votes": {
                "google": votes.google,
                "numpy": votes.numpy,
                "sphinx": votes.sphinx,
                "unknown": votes.unknown,
            },
            "confidence": confidence,
        }
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except IOError:
            pass

    return style


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python style_detector.py <repo_path>")
        sys.exit(1)

    repo = sys.argv[1]
    style, votes, conf = detect_repo_style(repo)

    print(f"Detected style: {style}")
    print(f"Confidence: {conf:.2%}")
    print(f"Votes: google={votes.google}, numpy={votes.numpy}, sphinx={votes.sphinx}, unknown={votes.unknown}")
